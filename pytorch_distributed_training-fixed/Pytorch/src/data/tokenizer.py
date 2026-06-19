from transformers import AutoTokenizer


def load_tokenizer(model_name: str):
    """
    Load Hugging Face tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Qwen tokenizer doesn't always have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def tokenize_function(example, tokenizer, max_seq_length):
    """
    Tokenize a single training example.
    """

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
