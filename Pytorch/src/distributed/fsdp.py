import os
import functools

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
)


def setup_fsdp():

    dist.init_process_group(
        backend="nccl"
    )

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank


def wrap_fsdp(model, config):

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2DecoderLayer,
        },
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        sync_module_states=True,
        forward_prefetch=False,
        limit_all_gathers=True,
    )

    return model


def cleanup():

    dist.destroy_process_group()
