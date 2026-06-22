#!/bin/bash
# End-to-end closed loop: raw data -> dataset -> pretraining.
# Run from the repo root on the Ascend NPU server (after `pip install -r requirements.txt`).
# Each step writes into this repo (data/, data_facts/, dataset/, output/).
set -eo pipefail
cd "$(dirname "$0")"
PY=python                      # a Python 3.9+ with torch_npu, transformers==4.57.1
DP=data_pipeline

# 0. External raw reference data (cjkvi-ids, chaizi, polyphone). Official corpus
#    is auto-downloaded from Hugging Face by step 01.
( cd $DP && bash 00_fetch_raw.sh )

# 1-4. Corpus assembly -> data/mixed/train.jsonl (base, ~86M jieba words)
$PY $DP/01_analyze_official.py
$PY $DP/02_build_hanzi_data.py
$PY $DP/03_build_children_data.py
$PY $DP/04_mix_dataset.py

# Preprocess the BASE corpus (character-level base tokenizer) -> data/*.npy + data/tokenizer/
$PY train/preprocess.py --input data/mixed/train.jsonl --output_dir data

# 5-6. In-distribution char set + expanded tokenizer (base vocab + radicals -> 21457)
$PY $DP/05_build_indist_chars.py
$PY $DP/06_expand_vocab.py

# 7-11. Knowledge-fact injection -> data/facts/all_facts.jsonl (~6.8M jieba words)
$PY $DP/07_build_pinyin_facts.py
$PY $DP/08_build_structure_facts.py
$PY $DP/09_amplify_hanzi_facts.py
$PY $DP/10_build_grammar_facts.py
$PY $DP/11_combine_facts.py

# Preprocess the FACTS (expanded tokenizer) -> data_facts/*.npy
$PY train/preprocess.py --input data/facts/all_facts.jsonl --output_dir data_facts --tokenizer data/tokenizer_expanded

# 12. Pack base + facts into the final training set -> dataset/
$PY $DP/12_build_npy.py

# Pretrain (16-card DDP) -> output/final
( cd train && bash train.sh )

echo "Done. Trained model in ./output/final. Evaluate with eval/ and submit with submit/."
