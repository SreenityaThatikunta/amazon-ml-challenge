# EDA

Exploratory analysis of the Business Entity Resolution data. Each script
answers questions that feed the pipeline design (see
[../docs/RESEARCH_NOTES.md](../docs/RESEARCH_NOTES.md)). It prints to stdout and
writes the same output to `reports/<script>.txt`. The key takeaways are summarised
in [FINDINGS.md](FINDINGS.md).

| Script | What it answers | Feeds |
|---|---|---|
| `01_overview.py` | Sizes, country split, empty/duplicate fields, name scripts & letter case, record lengths (LLM prompt sizing) | everything |
| `02_ground_truth.py` | Cluster sizes, S2 vs S3 matches per S1, singletons by country, one-owner check, distractor rate | decision step, validation split |
| `03_pair_noise.py` | How true pairs differ: exact-match rates, suffix changes, reorders, script changes, added/dropped/replaced tokens, single-char typo map, address number relations | normalisation tables, features, LLM prompt design |
| `04_distractors.py` | Do unmatched records mimic S1 entities? Shared names/addresses inside S1; hard-negative examples | precision, hard-negative mining |
| `05_blocking_keys.py` | Recall and block size of candidate keys (name tokens, postcode, house numbers, trigram similarity) incl. for Indic names | candidate generation |
| `06_test_france.py` | Train→test shift, French name/address conventions, unlabelled pseudo-pairs showing French noise | generalisation to France |

## Run

From `student_resource/`, with pandas and numpy installed (`pip install -r eda/requirements.txt`):

```bash
cd eda && python3 01_overview.py
```

or all scripts in order:

```bash
eda/run_all.sh
```

The first run parses the TSVs and caches them as pickles in `eda/.cache/`
(gitignored, a few GB), so later runs start faster. Scripts 03–05 sample pairs
(`N_SAMPLE` at the top of each file). Increase it for tighter estimates.
