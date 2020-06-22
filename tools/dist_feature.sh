#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/save_feat.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}