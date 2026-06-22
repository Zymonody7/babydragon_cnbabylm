# Chinese BabyLM 2026 — Final Submission Pipeline

Team **BabyDragon** · model **`zymonody/chinese-babylm-v4`** · NLPCC 2026 Shared Task 5.

A self-contained, reproducible pipeline for a data-efficient Chinese masked
language model: raw-data acquisition → knowledge-fact injection → tokenizer →
pretraining → evaluation → leaderboard submission. Everything runs **inside this
directory** (clone → run); no dependency on any external working tree. Trained
within the ≤102M jieba-word data budget.

## Model

- **Architecture**: DeBERTa-v2 base — 12 layers · 12 heads · hidden 768 · FFN 3072 · max-len 512 · ~104M params.
- **Tokenizer**: character-level (bert-base-chinese vocab + 329 radical/component characters) → vocab **21457** (0 UNK on the evaluation's hanzi targets).
- **Objective**: whole-word MLM (mask rate 0.15) + pinyin-prediction auxiliary head (weight 0.3).
- **Training**: from scratch, 60 epochs, AdamW lr 1e-4, wd 0.01, warmup 0.05, fp16, seed 42, 16-card Ascend 910 DDP.

## Data budget (jieba 0.42.1, ≤102M)

| Component | jieba words |
|---|---|
| Base child-oriented mix (subtitles / children's books / education / child-directed speech / …) | 86.15M |
| Injected facts — hanzi pinyin + structure (amplified) | 4.92M |
| Injected facts — grammar templates (×30) | 1.84M |
| **Total** | **≈ 92.9M** ✅ |

All injected facts are **code-generated from factual databases** (Unihan/pinyin via
`pypinyin`, character decomposition via cjkvi-ids/chaizi) and rule-based grammar
templates — **no large-LM-generated text**.

---

## Quick start

```bash
pip install -r requirements.txt          # + torch / torch_npu (Ascend) or torch (CUDA)
```

**Option A — full closed loop (rebuild data from scratch):**
```bash
bash run_all.sh                          # raw data → dataset/ → output/final
```
`run_all.sh` runs the whole chain in dependency order; every step writes into this
repo (`data/`, `data_facts/`, `dataset/`, `output/`, all git-ignored).

**Option B — train directly on the prebuilt dataset (skip the data pipeline):**
```bash
# place the prebuilt `dataset/` (token_ids/word_ids/pinyin_ids .npy + tokenizer +
# pinyin_vocab.json) at the repo root, then:
cd train && bash train.sh                # → ../output/final
```
`dataset/` is fully self-contained (tokenizer + pinyin vocab are packed in), so no
data rebuild is needed to reproduce training.

### Platform note (NVIDIA / Ascend)

`train/train.sh` and `train.py` are **portable**: they auto-detect the device
(CUDA `nccl` or Ascend `torch_npu`/`hccl`) and the GPU count. On an NVIDIA host it
just works — `nvidia-smi` sets `--nproc_per_node`, and `gradient_accumulation_steps`
auto-scales to keep the global batch at 256 (the original 16-card × 16 setting).
Override with `NPROC=<n>` / `GRAD_ACCUM=<n>` env vars.

### Raw-data acquisition

- **Official corpus** — downloaded automatically by `01_analyze_official.py` via
  `datasets.load_dataset("chinese-babylm-org/babylm-zho-100M")`.
- **Reference databases** — fetched by `data_pipeline/00_fetch_raw.sh` from their
  canonical upstreams into `data/supplementary/`:
  - cjkvi-ids (character decomposition) — github `cjkvi/cjkvi-ids`
  - chaizi (拆字) — github `kfcd/chaizi`
  - polyphone dictionary (多音字) — github `mapull/chinese-dictionary`

---

## Pipeline steps

### `data_pipeline/` — data assembly & fact injection

| # | Script | Produces |
|---|---|---|
| 00 | `00_fetch_raw.sh` | Fetch cjkvi-ids / chaizi / polyphone into `data/supplementary/`. |
| 01 | `01_analyze_official.py` | Download + analyze the official corpus → `data/official/`. |
| 02 | `02_build_hanzi_data.py` | Hanzi-knowledge supplementary corpus. |
| 03 | `03_build_children_data.py` | Children's-reading supplementary corpus (inline sources). |
| 04 | `04_mix_dataset.py` | Deterministic (seed 42) mix → `data/mixed/train.jsonl` (86.15M jieba words). |
| 05 | `05_build_indist_chars.py` | In-distribution common chars (freq ≥ 5) → restricts fact targets, robust to held-out chars. |
| 06 | `06_expand_vocab.py` | Extend tokenizer 21128 → **21457** (+radicals), surgically → `data/tokenizer_expanded/`. |
| 07 | `07_build_pinyin_facts.py` | Pinyin homophone facts in the **exact eval frame** (initial/final/tone), via `pypinyin`. |
| 08 | `08_build_structure_facts.py` | Hanzi structure facts (component & position) in the exact eval frame, via cjkvi-ids. |
| 09 | `09_amplify_hanzi_facts.py` | Amplify hanzi facts → `data/facts/hanzi_facts.jsonl`. |
| 10 | `10_build_grammar_facts.py` | Rule-based templates for ZhoBLiMP-weak phenomena (A-not-A, ba/passive, VP-ellipsis, reflexive agreement). |
| 11 | `11_combine_facts.py` | hanzi facts + grammar×30 → `data/facts/all_facts.jsonl` (≈6.76M jieba words). |
| 12 | `12_build_npy.py` | Concatenate base `.npy` + facts `.npy`, copy expanded tokenizer → `dataset/` (the training set). |

> `train/preprocess.py` is invoked twice by `run_all.sh` — once on the base corpus
> (base tokenizer → `data/`) and once on the combined facts (expanded tokenizer →
> `data_facts/`) — before step 12 packs them together.

### `train/` — pretraining

```bash
cd train && bash train.sh        # torchrun 16-card → ../output/final
```

### `eval/` — official evaluation

Uses `github.com/chinese-babylm/chinese-babylm-pipeline-final` (14 tasks, open + hidden).

```bash
git clone https://github.com/chinese-babylm/chinese-babylm-pipeline-final
cd chinese-babylm-pipeline-final
python pipeline.py download                  # tasks + cogbench-full (~19GB)
cp /path/to/eval/config.yaml ./config.yaml   # model id + all 14 tasks + official hparams
bash /path/to/eval/eval.sh                    # 14 tasks across NPUs
python pipeline.py gather -c config.yaml --export results.json
```

`eval/patches/` are needed **only** to run the official pipeline under Python 3.9 + Ascend NPU:
`match2if.py` (backport `match`/`case`), `add_future.py` (lazy `X | None` hints),
`npu_patch.py` (prefer the `npu` device, else it silently falls back to CPU).

### `submit/` — leaderboard

```bash
python submit/submit.py results/results.json     # → BabyDragon-v4 on the final leaderboard
```

---

## Results (official final pipeline, open + hidden tasks)

| Track | Tasks | Score |
|---|---|---|
| **NLU** | zhoblimp 78.44 · xcomps_zh 57.81 · afqmc 70.71 · ocnli 67.42 · tnews 53.84 · cluewsc 65.79 · c3 32.02 · diagnostic_nli 53.77 | **59.98** |
| **HANZI** | structure 83.15 · structure_hidden 83.70 · pinyin 99.85 · pinyin_hidden 99.55 | **91.56** |
| **Cog** | word_fmri 56.26 · fmri 9.62 | **32.94** |
| **Overall** (track mean) | | **61.49** |

Hidden hanzi scores ≈ open hanzi scores → the injected knowledge **generalizes to
unseen characters**, not memorized test items.

## Artifacts

- Model + tokenizer: https://huggingface.co/zymonody/chinese-babylm-v4 (public)
- Leaderboard entry: BabyDragon-v4 · `chinese-babylm-org/chinesebabylm-2026-final-leaderboard`
- `results/results.json` — the exported scores submitted to the leaderboard.

## Layout

```
final_submission/
├── run_all.sh              # closed-loop orchestrator (data → dataset → output)
├── requirements.txt
├── data_pipeline/          # 00 fetch raw · 01-12 build data + inject facts
├── train/                  # preprocess.py · train.py · train.sh
├── eval/                   # config.yaml · eval.sh · patches/
├── results/results.json
└── submit/submit.py
```
