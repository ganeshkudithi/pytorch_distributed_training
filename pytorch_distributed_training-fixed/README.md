# pytorch_distributed_training
Distributed training framework using pytorch


#Single GPU
python train.py --config configs/qwen_single.yaml

#DDP
torchrun --nproc_per_node=4 train.py \
    --config configs/qwen_ddp.yaml

#FSDP2
torchrun --nproc_per_node=4 train.py \
    --config configs/qwen_fsdp.yaml
