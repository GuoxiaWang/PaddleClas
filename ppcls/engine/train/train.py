# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import time
import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info

from collections import defaultdict
@paddle.no_grad()
def clip_grad_norm_(grads_fp32,
                    grads_fp16,
                    grad_norm_clip=2.0,
                    grad_norm_clip_max=2.0):

    if len(grads_fp32) <= 0 and len(grads_fp16) <= 0:
        print('grads_fp32 and grads_fp16 are empty')
        return None

    if len(grads_fp32) > 0:
        norm_fp32 = paddle.sum(
            paddle.stack([
                paddle.matmul(g.detach().reshape((1, -1)),
                              g.detach().reshape((-1, 1))) for g in grads_fp32
            ]))
    if len(grads_fp16) > 0:
        norm_fp16 = paddle.sum(
            paddle.stack([
                paddle.matmul(g.detach().reshape((1, -1)),
                              g.detach().reshape((-1, 1))) for g in grads_fp16
            ]))

    if len(grads_fp32) > 0 and len(grads_fp16) > 0:
        global_norm = paddle.sqrt(norm_fp32 + paddle.cast(norm_fp16,
                                                          'float32'))
    elif len(grads_fp32) > 0:
        global_norm = paddle.sqrt(norm_fp32)
    elif len(grads_fp16) > 0:
        global_norm = paddle.cast(norm_fp16, 'float32')

    clip_coef_fp32 = paddle.clip(
        grad_norm_clip / (global_norm + 1e-6), max=grad_norm_clip_max)

    if len(grads_fp32) > 0:
        grads_fp32 = [g.scale_(clip_coef_fp32) for g in grads_fp32]

    if len(grads_fp16) > 0:
        clip_coef_fp16 = paddle.cast(clip_coef_fp32, 'float16')
        grads_fp16 = [g.scale_(clip_coef_fp16) for g in grads_fp16]

    return global_norm

@paddle.no_grad()
def clip_grad_norm(optimizer, grad_norm_clip=1.0, grad_norm_clip_max=1.0):

    # data parallel
    param_grads_dict = defaultdict(list)
    # model parallel
    dist_param_grads_dict = defaultdict(list)

    if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict):
        for group in optimizer._param_groups:
            for param in group['params']:
                if not param.is_distributed:
                    if param._grad_ivar() is not None:
                        param_grads_dict[param._grad_ivar().dtype].append(
                            param._grad_ivar())
                else:
                    if param._grad_ivar() is not None:
                        dist_param_grads_dict[param._grad_ivar(
                        ).dtype].append(param._grad_ivar())
                    elif getattr(param, 'sparse_grad', None) is not None:
                        grad = getattr(param, 'sparse_grad')
                        dist_param_grads_dict[grad.dtype].append(grad)
    else:
        for param in optimizer._parameter_list:
            if not param.is_distributed:
                if param._grad_ivar() is not None:
                    param_grads_dict[param._grad_ivar().dtype].append(
                        param._grad_ivar())
            else:
                if param._grad_ivar() is not None:
                    dist_param_grads_dict[param._grad_ivar().dtype].append(
                        param._grad_ivar())
                elif getattr(param, 'sparse_grad', None) is not None:
                    grad = getattr(param, 'sparse_grad')
                    dist_param_grads_dict[grad.dtype].append(grad)

    grads_fp32 = []
    grads_fp16 = []
    if len(param_grads_dict[paddle.float32]) > 0:
        coalesced_grads_and_vars_fp32 = \
            paddle.fluid.dygraph.parallel.build_groups(param_grads_dict[paddle.float32], 128 * 1024 * 1024)
        for coalesced_grad, _, _ in coalesced_grads_and_vars_fp32:
            grads_fp32.append(coalesced_grad)

    if len(param_grads_dict[paddle.float16]) > 0:
        coalesced_grads_and_vars_fp16 = \
            paddle.fluid.dygraph.parallel.build_groups(param_grads_dict[paddle.float16], 128 * 1024 * 1024)
        for coalesced_grad, _, _ in coalesced_grads_and_vars_fp16:
            grads_fp16.append(coalesced_grad)

    clip_grad_norm_(grads_fp32, grads_fp16, grad_norm_clip, grad_norm_clip_max)

    if len(param_grads_dict[paddle.float16]) > 0:
        paddle.fluid.dygraph.parallel._split_tensors(
            coalesced_grads_and_vars_fp16)
    if len(param_grads_dict[paddle.float32]) > 0:
        paddle.fluid.dygraph.parallel._split_tensors(
            coalesced_grads_and_vars_fp32)

def train_epoch(engine, epoch_id, print_batch_step):
    tic = time.time()
    for iter_id, batch in enumerate(engine.train_dataloader):
        if iter_id >= engine.max_iter:
            break
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch_size = batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        engine.global_step += 1

        # image input
        if engine.amp:
            with paddle.amp.auto_cast(custom_black_list={
                    "flatten_contiguous_range", "greater_than"
            }):
                out = forward(engine, batch)
                loss_dict = engine.train_loss_func(out, batch[1])
        else:
            out = forward(engine, batch)

        # calc loss
        if engine.config["DataLoader"]["Train"]["dataset"].get(
                "batch_transform_ops", None):
            loss_dict = engine.train_loss_func(out, batch[1:])
        else:
            loss_dict = engine.train_loss_func(out, batch[1])

        # step opt and lr
        if engine.amp:
            scaled = engine.scaler.scale(loss_dict["loss"])
            scaled.backward()
            engine.scaler.step(engine.optimizer)
        else:
            loss_dict["loss"].backward()
            if engine.global_norm_clip:
                assert engine.global_norm_clip_max is not None
                clip_grad_norm(engine.optimizer, engine.global_norm_clip,
                    engine.global_norm_clip_max)
            engine.optimizer.step()
        engine.optimizer.clear_grad()
        
        if engine.lr_sch_unit == 'step':
            engine.lr_sch.step()

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
        tic = time.time()

    if engine.lr_sch_unit == 'epoch':
        engine.lr_sch.step(epoch_id-1)

def forward(engine, batch):
    if not engine.is_rec:
        return engine.model(batch[0])
    else:
        return engine.model(batch[0], batch[1])
