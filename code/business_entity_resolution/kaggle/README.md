# Kaggle: LLM matcher (tasks T5.4 / T5.5)

The LLM stage runs on Kaggle GPUs (T4 ×2 or P100, fp16). Everything else runs
locally. The round trip:

```
local  ber llm-export --run main          -> artifacts/main/llm-export/llm_{train,val,test}.parquet
        upload those 3 files + this kaggle/ folder as a private Kaggle dataset "ber-llm"
kaggle train_lora.py                       -> /kaggle/working/lora   (save as dataset "ber-lora")
kaggle infer.py  (val and test)            -> llm_scores_{val,test}.parquet
        download them into artifacts/llm_scores/
local  ber llm-import --run main           -> blended scores, blend weight chosen on val
local  ber decide --run main --set decide.scores_stage=llm-import
local  ber submit --run main
```

## One-time setup (you)
1. Kaggle account with a **verified phone** (needed for GPU and internet).
2. Notebook settings: *Accelerator* = GPU T4 ×2 (or P100), *Internet* = on (to
   download `Qwen/Qwen3-1.7B` from Hugging Face; Apache-2.0).
3. Create a **private** dataset `ber-llm` with `llm_train.parquet`, `llm_val.parquet`,
   `llm_test.parquet`, `common.py`, `train_lora.py`, `infer.py`.

## Notebook cells
```python
!pip install -q -r /kaggle/input/ber-llm/requirements.txt
import shutil, os
for f in ["common.py", "train_lora.py", "infer.py"]:
    shutil.copy(f"/kaggle/input/ber-llm/{f}", "/kaggle/working/")
os.chdir("/kaggle/working")

# 1) quick throughput check before committing GPU hours (T5.5 exit criterion)
!python train_lora.py --data /kaggle/input/ber-llm/llm_train.parquet \
    --val /kaggle/input/ber-llm/llm_val.parquet --out lora --max-train 2000 --max-val 500
!python infer.py --data /kaggle/input/ber-llm/llm_test.parquet --lora lora \
    --out bench.parquet --limit 2000

# 2) full training (~budget: see the step time printed above)
!python train_lora.py --data /kaggle/input/ber-llm/llm_train.parquet \
    --val /kaggle/input/ber-llm/llm_val.parquet --out lora

# 3) inference
!python infer.py --data /kaggle/input/ber-llm/llm_val.parquet  --lora lora --out llm_scores_val.parquet
!python infer.py --data /kaggle/input/ber-llm/llm_test.parquet --lora lora --out llm_scores_test.parquet
```

Sessions are limited to 12 h: if inference on test does not fit, split
`llm_test.parquet` into shards (e.g. by row ranges) and run one per session.

## Budget notes
- Prompts: ~5 candidates, median ≈ 1.5k characters (≈ 400–600 tokens).
- Test prompts ≈ 10% of test queries (route band [0.05, 0.95]).
- `infer.py` prints tokens/s: multiply by the test prompt count to check the
  GPU budget before the full run. If it's too slow: narrow the band
  (`llm.route_low/high`), use `--model Qwen/Qwen3-0.6B`, or reduce
  `llm.n_candidates`.
