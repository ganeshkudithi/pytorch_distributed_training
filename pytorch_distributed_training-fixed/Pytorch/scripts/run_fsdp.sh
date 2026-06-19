#!/bin/bash
# Run FSDP training with torchrun

set -e

CONFIG=${1:-"configs/qwen14b_fsdp.yaml"}
NPROC_PER_NODE=${2:-$(nvidia-smi --list-gpus | wc -l)}

echo "Launching FSDP training"
echo "  Config         : $CONFIG"
echo "  GPUs per node  : $NPROC_PER_NODE"

torchrun \
    --standalone \
    --nproc_per_node=$NPROC_PER_NODE \
    src/trainer/train.py \
    --config $CONFIG
