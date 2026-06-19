import torch


def apply_liger(model, config):
    """
    Apply Liger Kernel if enabled.
    """

    if config["model"]["liger_kernel"]:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        apply_liger_kernel_to_qwen2()

    return model


def apply_compile(model, config):
    """
    Compile model using torch.compile().
    """

    if config["model"]["torch_compile"]:
        model = torch.compile(model)

    return model
