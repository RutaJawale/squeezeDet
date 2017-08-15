#!/bin/bash

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/eval.py \
  --dataset=KITTI \
  --data_path=./data/KITTI \
  --image_set=val_100 \
  --eval_dir=/rscratch/ruta/logs/squeezeDet/8.8/eval_val \
  --checkpoint_path=/rscratch/ruta/squeezeDet/data/model_checkpoints/squeezeDet/model.ckpt-87000 \
  --run_once=True \
  --net=squeezeDet \
  --gpu=0

# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=val \
#   --eval_dir=/tmp/bichen/logs/SqueezeDetPlus/eval_val \
#   --checkpoint_path=/tmp/bichen/logs/SqueezeDetPlus/train \
#   --net=squeezeDet+ \
#   --gpu=0

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=val \
#   --eval_dir=/tmp/bichen/logs/vgg16/eval_val \
#   --checkpoint_path=/tmp/bichen/logs/vgg16/train \
#   --net=squeezeDet+ \
#   --gpu=0

# =========================================================================== #
# command for resnet50:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=val \
#   --eval_dir=/tmp/bichen/logs/resnet50/eval_train \
#   --checkpoint_path=/tmp/bichen/logs/resnet50/train \
#   --net=resnet50 \
#   --gpu=0
