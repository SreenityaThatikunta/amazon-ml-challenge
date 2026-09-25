"""Shared helpers for the Kaggle LLM scripts (kept free of the `ber` package so the
scripts run from a single Kaggle dataset upload)."""

import torch

DIGITS = [str(i) for i in range(10)]


def digit_token_ids(tok):
    """Token id of each answer digit "0".."9" (Qwen tokenises digits one by one)."""
    ids = [tok.convert_tokens_to_ids(d) for d in DIGITS]
    assert all(i is not None and i != tok.unk_token_id for i in ids), ids
    return torch.tensor(ids)


def format_prompt(prompt: str) -> str:
    # the answer digit is the very next token after this newline
    return prompt.rstrip() + "\n"


def batch_encode(tok, prompts, max_len):
    enc = tok([format_prompt(p) for p in prompts], return_tensors="pt", padding=True,
              truncation=True, max_length=max_len)
    return enc


def last_logits(model, enc, digit_ids):
    """Logits over the answer digits at each sequence's last real token
    (left padding => the last position)."""
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return out.logits[:, -1, :].float()[:, digit_ids.to(out.logits.device)]
