# Research Notes: Prior Art for Business Entity Resolution

Notes on existing work that informs our pipeline. The target is to beat the public
leaderboard top of **0.971 macro F0.5** (as of 2026-09-25).

Claims marked *(unverified)* come from memory or secondary sources and were not
checked against the primary source. Verify them before relying on them, especially
licenses.

---

## 1. This challenge (Amazon ML Challenge 2026)

- The problem is new. Participant repos exist, but no public scores or winning
  write-ups were found as of 2026-09-25:
  - https://github.com/Sanyam2005/amazonMl
  - https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution
  - https://github.com/ItsDemonic/amazon-business-entity-resolution-challenge
- Registration: https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604
- Past Amazon ML Challenges were different tasks, but winners follow the same
  pattern: strong open models within the size limit, plus heavy ensembling and
  post-processing.
  - 2024: entity value extraction from product images. Reported winning repo
    ensembled Qwen2-VL-7B and MiniCPM-V 2.6 (https://github.com/KhadgaA/Amazon-ML-Challenge).
  - 2025: price prediction (SMAPE), about 23k teams. Solutions combined text and
    image encoders (DeBERTa, CLIP).

## 2. Closest past competition: Kaggle Foursquare Location Matching (2022)

Matching points of interest by noisy multilingual name, address, category and
coordinates. This is the most similar public competition to ours.

- 1st place (team re:waiwai):
  https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution
- Organizer's summary of top solutions:
  https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/
- Chinese-language roundup of solutions: https://zhuanlan.zhihu.com/p/555976163

**Common structure of the top solutions:**
1. **Candidate generation from several sources combined:** spatial nearest
   neighbours, TF-IDF nearest neighbours on name, embedding nearest neighbours.
   Each record keeps top-k from each source, with recall measured explicitly.
2. **Pruning with a cheap model:** a light LightGBM on a few features cuts
   candidates down (1st place kept the top ~40 per record) before the expensive
   stage.
3. **Main model:** LightGBM/XGBoost on many string-similarity features.
   - Features: Levenshtein, Jaro-Winkler, longest common substring, token set
     ratios, TF-IDF cosine, SVD-embedding cosine.
   - Computed on name, address and city.
4. **Multilingual transformer cross-encoder** (xlm-roberta) on the remaining pairs,
   blended with the GBDT.
5. **Graph post-processing:** add or drop edges using transitivity or connected
   components.

**What we take from it:**
- Combine several blocking sources and score recall on its own.
- Filter in two stages, cheap then expensive.
- Feed the GBDT features about the candidate's context: its rank among the
  record's candidates, and the gap to the best candidate.
- Save the transformer for last and use it only on uncertain pairs.

## 3. Other benchmarks and contests

- **SIGMOD Programming Contest 2021/2022** *(unverified)*: entity resolution and
  blocking on product data under time and recall limits. Strong teams used
  hand-crafted normalisation plus token blocking.
- Standard entity-matching benchmarks: WDC Products, Abt-Buy, Amazon-Google,
  Walmart-Amazon, DBLP-ACM. Scores on the structured ones (DBLP-ACM, ~99 F1) are
  close to ours, because the noise is synthetic or structured.

## 4. State of the art in entity matching (2023–2026)

| Method | Idea | Reported result | Usable here? |
|---|---|---|---|
| **Ditto** *(details unverified)* | RoBERTa cross-encoder on serialised records, plus data augmentation and injecting domain knowledge (e.g. marking numbers or suffixes) | F1 ~85 (WDC) to ~99 (DBLP-ACM) | Yes, as a small cross-encoder for v2 |
| **Peeters, Steiner & Bizer, EDBT 2025** (https://arxiv.org/abs/2310.11244) | LLMs as matchers; zero-shot vs fine-tuned | GPT-4 zero-shot F1 76–98; **Llama-3.1-8B + LoRA beat GPT-4 zero-shot on 4 of 6 datasets** | Idea yes; check the Llama license (it is not MIT/Apache) |
| **AnyMatch** (https://arxiv.org/abs/2409.04073) | Small fine-tuned language model; AutoML-selected hard training pairs; label balancing | Within ~4.4% of GPT-4 at a fraction of the cost | Yes: **mine hard negatives for training** |
| **Jellyfish-8B** | Llama-3-8B fine-tuned on data-processing tasks | F1 Abt-Buy 88.8, DBLP-ACM 98.8 | **No: CC-BY-NC license** |
| **Unicorn, HierGAT, Sudowoodo** *(unverified)* | Mixture-of-experts matcher; graph attention; contrastive self-supervised learning | Competitive on the standard benchmarks | Maybe: contrastive embeddings for blocking |
| **"Match, Compare, or Select?"** (https://arxiv.org/abs/2405.16884) | Asks the LLM to **select the best of N candidates** rather than judge pairs one at a time | Selection beats pairwise matching | **Yes: matches our one-owner-per-record setup** |

**Blocking:**
- **Sparkly** *(unverified)*: TF-IDF/BM25 top-k is a very strong, simple
  baseline, often beating learned blockers.
- **DeepBlocker** *(unverified)*: learned embedding blocking.
- FAISS / ANN on sentence embeddings.

## 5. Tools and models (check licenses before use)

The rule is that the **final model** must be MIT/Apache-2.0 and at most 8B
parameters. Libraries should still be permissive so the submission can be
reproduced cleanly.

| Tool / model | Use | License *(unverified unless noted)* |
|---|---|---|
| libpostal / pypostal | Offline address parsing and normalisation (US, India, France) | MIT |
| rapidfuzz | Fast string similarity (Levenshtein, Jaro-Winkler, token ratios) | MIT |
| Splink | Probabilistic record linkage (Fellegi-Sunter); ideas for m/u weights | MIT |
| dedupe | Active-learning deduplication | MIT |
| Zingg | ER framework | **AGPL: avoid** |
| multilingual-e5 (small/base) | Multilingual embeddings for blocking and cross-encoding | MIT |
| bge-m3 | Multilingual embeddings, many scripts | MIT |
| LaBSE | Cross-lingual sentence embeddings (Hindi and English share a space) | Apache-2.0 |
| xlm-roberta-base | Cross-encoder backbone | MIT |
| Qwen2.5 (≤7B) | LLM matcher option | Apache-2.0 for most sizes; check each one |
| indic-transliteration / AI4Bharat IndicXlit | Indian scripts → Latin | MIT |
| Unidecode | Accent / script transliteration | **GPL: avoid**; use `unicodedata` NFKD or `anyascii` (ISC) |

---

## 6. How this shapes our pipeline

Each design choice and where it comes from:

| Pipeline stage | Choice | Inspired by |
|---|---|---|
| Normalisation | Expand abbreviations, separate legal suffixes, map states, strip accents; **learn an Indic→Latin word dictionary from training pairs**, falling back to IndicXlit | Ditto's domain-knowledge injection; our EDA (5–9% Indian-script names) |
| Blocking | Combine name char-3gram TF-IDF top-k, address TF-IDF top-k, and exact keys (city+house number, postcode+rarest word); **measure recall on its own** | Foursquare 1st place; Sparkly |
| Blocking v2 | Multilingual embedding ANN (e5 / bge-m3 / LaBSE) to catch transliterated and heavily rewritten names | Foursquare embedding stage; DeepBlocker |
| Pruning | Cheap LightGBM to cut candidates before expensive features | Foursquare 1st place |
| Matcher v1 | LightGBM on name/address similarity plus **context features** (candidate rank, gap to best, candidate count, name frequency); no country feature | Foursquare top solutions |
| Training data | Mine hard negatives (same address + different name; similar name + nearby address) | AnyMatch |
| Assignment | Each S2/S3 record → its single best S1 (EDA: every S2/S3 record matches at most one S1) | "Match, Compare, or Select?" (select beats pairwise) |
| Decision | Per S1, pick the match set that maximises **expected F0.5** from probabilities, with "no match" as an option | Kaggle F1-optimisation practice; our macro metric |
| Matcher v2 | Small multilingual cross-encoder (xlm-r / e5, MIT) only on uncertain pairs, blended with the GBDT | Ditto; Foursquare transformer stage |
| Matcher v3 (optional) | Qwen2.5 ≤7B + LoRA as a "pick the owner from N candidates" matcher | Peeters et al.; Match/Compare/Select |
| Unseen France | Country-agnostic features; French abbreviation tables (rue, bd, av, SARL/SAS/EURL); optionally pseudo-label high-confidence test matches (**check fair-play rules first**) | General domain-adaptation practice |
| Validation | Hold out whole cities/states so lookalike neighbours stay in; score with `utils/score.py` by country, match/no-match and script | Foursquare CV practice |
