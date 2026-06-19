
from src.distributed.ddp import setup_ddp, wrap_ddp

local_rank = setup_ddp()

model = load_model(config)

model = apply_liger(model, config)

model = apply_compile(model, config)

model = wrap_ddp(
    model,
    local_rank,
    config,
)

from src.distributed.fsdp import setup_fsdp, wrap_fsdp

local_rank = setup_fsdp()

model = load_model(config)

model = apply_liger(model, config)

model = apply_compile(model, config)

model = wrap_fsdp(
    model,
    config,
)
