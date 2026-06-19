import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp():

    dist.init_process_group(
        backend="nccl"
    )

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank


def wrap_ddp(model, local_rank, config):

    model = model.to(local_rank)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=config["distributed"]["find_unused_parameters"],
        static_graph=config["distributed"]["static_graph"],
    )

    return model


def cleanup():

    dist.destroy_process_group()
