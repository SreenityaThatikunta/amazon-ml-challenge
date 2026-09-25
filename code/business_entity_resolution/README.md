# business_entity_resolution

Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.
The design is in [../../plan.md](../../plan.md); status is in [../../progress.md](../../progress.md).

## Setup

From the `student_resource/` directory (Python 3.12):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
.venv/bin/pip install -e code/business_entity_resolution
```

Check the environment:

```bash
cd code/business_entity_resolution && ../../.venv/bin/pytest
../../.venv/bin/python tools/check_licenses.py
```

## Layout

| Path | Contents |
|---|---|
| `src/ber/normalize/` | Stage 0: text/address normalisation, transliteration, lookup tables (`tables/*.tsv`) |
| `src/ber/candidates/` | Stage 1: per-country char-TF-IDF top-k candidate generation |
| `src/ber/features/`, `src/ber/scorer/` | Stage 2: pair features, LightGBM scorer, uncertain-band router |
| `src/ber/llm/` | Stage 3: select-prompt LLM matcher (training/inference run on Kaggle) |
| `src/ber/decide/` | Stage 4: one-owner assignment, expected-F0.5 selection |
| `src/ber/france/` | Synthetic French training pairs |
| `src/ber/eval/` | Metrics, geographic split, reports |
| `configs/` | YAML configs (`default.yaml`, `dev.yaml`) |
| `kaggle/` | Kaggle notebooks for LoRA training / inference |
| `tools/` | Dev utilities (license check) |
| `tests/` | pytest suite |

End-to-end run instructions will be added as the stages land (T6.3).
