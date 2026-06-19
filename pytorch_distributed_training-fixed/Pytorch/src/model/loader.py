from transformers import AutoModelForCausalLM
import torch


def load_model(config):
    """
    Load Hugging Face causal language model.
    """

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        trust_remote_code=config["model"]["trust_remote_code"],
        attn_implementation=(
            "flash_attention_2"
            if config["model"]["flash_attention"]
            else "eager"
        ),
    )

    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    return model
