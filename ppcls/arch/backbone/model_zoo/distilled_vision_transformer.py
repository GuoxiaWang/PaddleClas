# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
import paddle.nn as nn
from .vision_transformer import VisionTransformer, Identity, trunc_normal_, zeros_

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "DeiT_tiny_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_patch16_224_pretrained.pdparams",
    "DeiT_small_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_patch16_224_pretrained.pdparams",
    "DeiT_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams",
    "DeiT_tiny_distilled_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_distilled_patch16_224_pretrained.pdparams",
    "DeiT_small_distilled_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_distilled_patch16_224_pretrained.pdparams",
    "DeiT_base_distilled_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams",
    "DeiT_base_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_384_pretrained.pdparams",
    "DeiT_base_distilled_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_num=class_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs)
        self.pos_embed = self.create_parameter(
            shape=(1, self.patch_embed.num_patches + 2, self.embed_dim),
            default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)

        self.dist_token = self.create_parameter(
            shape=(1, 1, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        self.head_dist = nn.Linear(
            self.embed_dim,
            self.class_num) if self.class_num > 0 else Identity()

        trunc_normal_(self.dist_token)
        trunc_normal_(self.pos_embed)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )
        
def _load_for_finetune(path, model):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
        
    state_dict = model.state_dict()
    param_state_dict = paddle.load(path + ".pdparams")
    for k in ['head.weight', 'head.bias']:
        if k in param_state_dict and param_state_dict[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del param_state_dict[k] 
            
    # interpolate position embedding
    pos_embed_checkpoint = param_state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = paddle.transpose(pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]), perm=[0, 3, 1, 2])
    pos_tokens = paddle.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)           
    pos_tokens = paddle.transpose(pos_tokens, perm=[0, 2, 3, 1]).flatten(1, 2)
    new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
    param_state_dict['pos_embed'] = new_pos_embed    
    
    model.set_dict(param_state_dict)    
    return

def DeiT_tiny_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if kwargs.get('finetune', False):
        assert isinstance(pretrained, str), "pretrained type is not available. Please use `string`."
        _load_for_finetune(
            pretrained,
            model)
    else:
        _load_pretrained(
            pretrained,
            model,
            MODEL_URLS["DeiT_tiny_patch16_224"],
            use_ssld=use_ssld)
    return model

def DeiT_small_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if kwargs.get('finetune', False):
        assert isinstance(pretrained, str), "pretrained type is not available. Please use `string`."
        _load_for_finetune(
            pretrained,
            model)
    else:
        _load_pretrained(
            pretrained,
            model,
            MODEL_URLS["DeiT_small_patch16_224"],
            use_ssld=use_ssld)
    return model


def DeiT_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if kwargs.get('finetune', False):
        assert isinstance(pretrained, str), "pretrained type is not available. Please use `string`."
        _load_for_finetune(
            pretrained,
            model)
    else:
        _load_pretrained(
            pretrained,
            model,
            MODEL_URLS["DeiT_base_patch16_224"],
            use_ssld=use_ssld)
    return model


def DeiT_tiny_distilled_patch16_224(pretrained=False, use_ssld=False,
                                    **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_tiny_distilled_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_small_distilled_patch16_224(pretrained=False,
                                     use_ssld=False,
                                     **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_small_distilled_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_base_distilled_patch16_224(pretrained=False, use_ssld=False,
                                    **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_distilled_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_base_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_patch16_384"],
        use_ssld=use_ssld)
    return model


def DeiT_base_distilled_patch16_384(pretrained=False, use_ssld=False,
                                    **kwargs):
    model = DistilledVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_distilled_patch16_384"],
        use_ssld=use_ssld)
    return model
