"""Batched inference for the select-matcher (task T5.5). Runs on Kaggle GPU.

    python infer.py --data /kaggle/input/ber-llm/llm_test.parquet \
        --lora /kaggle/input/ber-lora --out /kaggle/working/llm_scores_test.parquet

Writes q_id, cands, and probabilities over "0" (none) and each candidate
position (softmax over the digits 0..N). Prompts are sorted by length to
minimise padding. Throughput is printed so the GPU budget can be checked.
"""

import argparse
import json
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import batch_encode, digit_token_ids, last_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (benchmark)")
    args = ap.parse_args()

    info = json.load(open(f"{args.lora}/train_info.json"))
    tok = AutoTokenizer.from_pretrained(args.lora)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(info["base_model"], torch_dtype=torch.float16,
                                                device_map="auto")
    model = PeftModel.from_pretrained(base, args.lora).merge_and_unload().eval()
    digits = digit_token_ids(tok)
    dev = next(model.parameters()).device

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    df = df.assign(_len=df.prompt.str.len()).sort_values("_len")
    probs, n_tok, t0 = [], 0, time.time()
    with torch.no_grad():
        for i in range(0, len(df), args.batch):
            b = df.iloc[i:i + args.batch]
            enc = batch_encode(tok, b.prompt.tolist(), args.max_len).to(dev)
            n_tok += int(enc["attention_mask"].sum())
            with torch.autocast("cuda", dtype=torch.float16):
                lg = last_logits(model, enc, digits)
            for row, k in zip(lg, b.cands.map(len)):
                probs.append(torch.softmax(row[:k + 1], -1).cpu().numpy().round(5).tolist())
            if (i // args.batch) % 200 == 0:
                el = time.time() - t0
                print(f"{i + len(b):,}/{len(df):,} prompts, {n_tok / el:,.0f} tok/s, "
                      f"{(i + len(b)) / el:.1f} prompts/s", flush=True)
    el = time.time() - t0
    print(f"done: {len(df):,} prompts in {el / 60:.1f} min, {n_tok / el:,.0f} tok/s")
    out = df[["q_id", "cands"]].copy()
    out["probs"] = probs  # [p_none, p_cand1, ..., p_candN]
    out.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
