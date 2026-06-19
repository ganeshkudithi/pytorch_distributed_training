from datasets import load_dataset

from .tokenizer import tokenize_function


def load_sft_dataset(config, tokenizer):
    """
    Load and tokenize SFT dataset.
    """

    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
    )

    # Convert chat messages into a training string
    def formatting(example):

        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    dataset = dataset.map(
        formatting,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda x: tokenize_function(
            x,
            tokenizer,
            config["dataset"]["max_seq_length"],
        ),
        batched=True,
        num_proc=config["dataset"]["preprocessing_workers"],
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
        ],
    )

    return dataset
