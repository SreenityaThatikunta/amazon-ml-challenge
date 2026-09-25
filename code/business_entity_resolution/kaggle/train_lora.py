"""LoRA fine-tuning of the select-matcher (task T5.4). Runs on Kaggle GPU (T4/P100: fp16).

    python train_lora.py --data /kaggle/input/ber-llm/llm_train.parquet \
        --val /kaggle/input/ber-llm/llm_val.parquet --out /kaggle/working/lora

Loss = cross-entropy over the 10 answer digits at the last position (0 = none,
1..N = candidate), which is exactly what inference reads. Only prompts whose
candidate count fits the digits are used.
"""

import argparse
import json
import math
import random
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import batch_encode, digit_token_ids, last_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=200_000)
    ap.add_argument("--max-val", type=int, default=5_000)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj"]))
    model.print_trainable_parameters()
    digits = digit_token_ids(tok)
    dev = next(model.parameters()).device

    tr = pd.read_parquet(args.data).sample(frac=1.0, random_state=args.seed)
    tr = tr[tr.label.notna()].head(args.max_train)
    va = pd.read_parquet(args.val)
    va = va[va.label.notna()].sample(min(args.max_val, len(va)), random_state=args.seed)
    print(f"train {len(tr):,} val {len(va):,}; label none: {(tr.label == 0).mean():.3f}")

    # LoRA params in fp32 for stable fp16 mixed precision
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = math.ceil(len(tr) * args.epochs / (args.batch * args.grad_accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()

    def evaluate():
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for i in range(0, len(va), args.batch * 2):
                b = va.iloc[i:i + args.batch * 2]
                enc = batch_encode(tok, b.prompt.tolist(), args.max_len).to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = last_logits(model, enc, digits)
                correct += (lg.argmax(-1).cpu() == torch.tensor(b.label.astype(int).values)).sum().item()
                n += len(b)
        model.train()
        return correct / max(n, 1)

    print(f"val accuracy before training: {evaluate():.4f}")
    model.train()
    t0, step, micro = time.time(), 0, 0
    rows = tr.to_dict("records")
    n_micro = math.ceil(len(rows) * args.epochs / args.batch)
    for m in range(n_micro):
        b = [rows[(m * args.batch + j) % len(rows)] for j in range(args.batch)]
        enc = batch_encode(tok, [r["prompt"] for r in b], args.max_len).to(dev)
        y = torch.tensor([int(r["label"]) for r in b], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(last_logits(model, enc, digits), y)
        scaler.scale(loss / args.grad_accum).backward()
        micro += 1
        if micro % args.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % 1000 == 0:
                print(f"val accuracy @ {step}: {evaluate():.4f}", flush=True)
                model.save_pretrained(args.out)
        if step >= steps:
            break
    acc = evaluate()
    print(f"final val accuracy: {acc:.4f}; {(time.time() - t0) / 3600:.2f} h")
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"base_model": args.model, "val_accuracy": acc, "train_rows": len(tr),
               "steps": step, "args": vars(args)}, open(f"{args.out}/train_info.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
