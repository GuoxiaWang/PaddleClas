#!/usr/bin/env bash

# for single card train
# python3.7 tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
python -m paddle.distributed.launch --ips=$TRAINER_IP_LIST --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ./ppcls/configs/ImageNet/VisionTransformer/ViT_base_patch16_224.yaml
