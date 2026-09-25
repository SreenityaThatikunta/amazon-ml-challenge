# Progress

Single source of truth for task status, runs, metrics and decisions.
Plan and task definitions: [plan.md](plan.md).

**Status keys:** ⬜ todo · 🔄 in progress · ✅ done · ⏸ blocked · ❌ dropped

---

## Task board

| ID | Task | Status | Notes |
|---|---|---|---|
| — | EDA (eda/, FINDINGS.md) | ✅ | 2026-09-25 |
| — | Local F0.5 scorer (utils/score.py) | ✅ | 2026-09-25 |
| — | Research notes (docs/RESEARCH_NOTES.md) | ✅ | 2026-09-25 |
| T0.1 | Project scaffold, venv, requirements, license check | ✅ | 2026-09-25; `.venv` (Py 3.12), all deps permissive, pytest green |
| T0.2 | Config, logging, artifact IO, CLI skeleton | ✅ | 2026-09-25; `ber --help` lists all stages; noop writes manifest; 23 tests |
| T0.3 | Geographic train/val split + dev slice | ⬜ | |
| T0.4 | eval/metrics.py matching utils/score.py | ⬜ | |
| T1.1 | Text normaliser | ⬜ | |
| T1.2 | Address parser + tables | ⬜ | |
| T1.3 | Learned Indic→Latin dictionary + fallback | ⬜ | |
| T1.4 | `normalize` stage (train + test) | ⬜ | |
| T2.1 | Chunked TF-IDF top-k per country | ⬜ | |
| T2.2 | Combine candidates + recall report | ⬜ | |
| T2.3 | Tune k / index mix to recall targets | ⬜ | |
| T2.4 | (conditional) Dense ANN fallback | ⬜ | |
| T3.1 | Number-relation feature | ⬜ | |
| T3.2 | Edit-type / initials / gibberish features | ⬜ | |
| T3.3 | String + context features | ⬜ | |
| T3.4 | LightGBM train + calibration | ⬜ | |
| T3.5 | Minimal decision + submission (**first LB**) | ⬜ | |
| T3.6 | Router (uncertain band) | ⬜ | |
| T4.1 | Expected-F0.5 subset selection | ⬜ | |
| T4.2 | Duplicate tying / capacity assignment | ⬜ | |
| T4.3 | `ber explain` + error dumps | ⬜ | |
| T5.1 | Synthetic French pairs + French val proxy | ⬜ | |
| T5.2 | Prompt v1 + token budget | ⬜ | |
| T5.3 | LLM training data | ⬜ | |
| T5.4 | Kaggle LoRA fine-tune Qwen3-1.7B | ⬜ | |
| T5.5 | Kaggle batched inference + throughput | ⬜ | |
| T5.6 | Blend LLM + GBM (**LB**) | ⬜ | |
| T5.7 | (optional) Qwen3-4B / wider band | ⬜ | |
| T6.1 | France pseudo-labelling | ⬜ | |
| T6.2 | Error-analysis loop | ⬜ | |
| T6.3 | Clean-checkout reproducibility | ⬜ | |
| T6.4 | Documentation + submission zip | ⬜ | |

---

## Metrics

### Validation runs
| Date | Run id | Stage / change | Cand. recall | Val F0.5 | US | India | Indic | Singletons | Notes |
|---|---|---|---|---|---|---|---|---|---|

### Leaderboard submissions
| Date | Run id | Val F0.5 | Public LB | Δ vs prev | Notes |
|---|---|---|---|---|---|

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-09-25 | Target: beat 0.971 public LB | User goal |
| 2026-09-25 | Build v3 (LLM select matcher) in the first iteration, backed by TF-IDF blocking + GBM router | User choice; see plan §1 |
| 2026-09-25 | All stage recommendations accepted (plan §1) | User approval |
| 2026-09-25 | Compute: Kaggle GPUs for LLM training/inference; local Mac for everything else | User's available compute |
| 2026-09-25 | Unlabelled test data may be used for France (synthetic pairs + pseudo-labels) | User approval; document in methodology |
| 2026-09-25 | Query direction S2/S3 → S1; one owner per record | EDA: all matched ids unique |

---

## Log

### 2026-09-25
- Explored repo, problem statement, data (≈2.4 GB; train 2.2M S1 / 5.0M S2 / 5.3M S3; test adds France).
- Added `utils/score.py` (macro F0.5; tested on the README example and GT vs GT = 1.0).
- Research on prior art → `docs/RESEARCH_NOTES.md`.
- EDA scripts 01–06 → `eda/reports/`, summary in `eda/FINDINGS.md`. Key findings:
  one-owner structure; identical US/India generator; lookalike distractors
  (shifted house numbers, one-letter name edits); name OR address trigram
  similarity keeps 99.9% (Latin) / 98.0% (Indic) of true pairs; France looks
  no harder than US once normalised.
- Wrote `plan.md` and this file.
- **T0.1 done.** Scaffold at `code/business_entity_resolution/` (src layout, `pyproject.toml`, package
  `ber` with stage subpackages, `tests/`, `configs/`, `kaggle/`, `tools/`). Root `.venv` on Python 3.12.0
  (arm64); pinned `requirements.txt`: numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, scipy 1.18.1,
  scikit-learn 1.9.1, rapidfuzz 3.14.6, lightgbm 4.7.0, PyYAML 6.0.3, pytest 9.1.1.
  `tools/check_licenses.py` → all permissive. Smoke test passes. `.gitignore`: added `.venv/`, `artifacts/`,
  `*.egg-info/`, `.pytest_cache/` (and fixed a missing newline that merged two entries).
  Notes: pandas 3.x defaults (string dtype, copy-on-write) apply; local machine has **16 GB RAM / 10 cores**,
  so stage 1 must be chunked per country.
- **T0.2 done.** Modules in `src/ber/`:
  - `config.py`: typed dataclass config from YAML; `extends:`; `--set a.b=v` overrides (YAML-typed);
    **strict** (unknown keys / wrong scalar types → `ConfigError`); root auto-detected (dir containing
    `dataset/`, or `BER_ROOT`); `cfg.hash()` excludes run_id.
  - `log.py`: `ber.*` loggers, `timed()` blocks with duration + **peak RSS**.
  - `io.py`: strict TSV reading (`QUOTE_NONE`, no NA parsing, column/prefix checks), GT reader,
    `write_id_lists` enforcing the submission rules (every key once, S2/S3 ids only, dedup).
  - `artifacts.py`: `StageRun` context manager → `artifacts/<run>/<stage>/{log.txt, manifest.json, outputs}`;
    the manifest is written **even on failure** (status + traceback); downstream stages refuse failed/missing
    inputs with an actionable message and warn on config-hash mismatch.
  - `cli.py`: `ber <stage> [-c cfg.yaml] [-s k=v] [-r run]`; `config` and `noop` implemented, the other
    12 stages are registered and name the task that implements them.
  - `configs/default.yaml` (every key documented), `configs/dev.yaml` (`extends`, run_id=dev, slice=dev).
  - Tests: config (extends, overrides, strictness, circular extends, hash), io (id-list format and
    rules, quote handling), artifacts (manifest, failure recording, missing upstream), CLI. **23 passed.**
  - Bug caught by tests: a stage run outside the CLI wrote an empty log.txt (logger level never set);
    `StageRun` now calls `setup_logging` itself.
- `.gitignore` reorganised into sections: data, pipeline outputs (`artifacts/`, `output/`, submission zip),
  EDA caches/reports, `docs/`, model files (`*.safetensors`, `*.bin`, `*.pt`, `*.gguf`, `checkpoints/`,
  `models/`), secrets (`.env`, `kaggle.json`), Python/tool caches, editors/OS. Removed a duplicate `.venv/`.
  Next: T0.3.

---

## Open questions
- Local RAM is 16 GB: full-train TF-IDF top-k (T2.1) must stream per country/chunk; watch peak memory.
- Kaggle specifics to confirm at T5.4: GPU type available (P100 vs 2×T4), session RAM, dataset size limits.
