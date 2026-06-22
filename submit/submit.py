#!/usr/bin/env python3
"""Submit results to the Chinese BabyLM 2026 final leaderboard.

Usage:
    python submit.py ../results/results.json

Fill in the team/identity fields below before running. The model's
architecture/training metadata is informational on the leaderboard.
"""
import sys
from gradio_client import Client, handle_file

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "../results/results.json"

SPACE = "chinese-babylm-org/chinesebabylm-2026-final-leaderboard"

# --- identity (must match registration) ---
TEAM_NAME      = "BabyDragon"
SUBMITTER_NAME = "Yiming Zhao"
ORGANIZATION   = "University of Chinese Academy of Sciences"
CONTACT_EMAIL  = "zymonody@gmail.com"

# --- model card ---
MODEL_NAME = "BabyDragon-v4"
REVISION   = "main"
HF_REPO    = "zymonody/chinese-babylm-v4"

DESCRIPTION = (
    "DeBERTa-v2 base (~104M) masked LM; character-level tokenizer with radicals "
    "(vocab 21457). Whole-word MLM + pinyin auxiliary objective, trained from scratch "
    "for 60 epochs on a ~92.9M jieba-word child-oriented Chinese corpus (official "
    "BabyLM-zho family) augmented with hanzi pinyin/structure and grammar fact "
    "injection in eval-aligned frames."
)

client = Client(SPACE)
result = client.predict(
    TEAM_NAME, SUBMITTER_NAME, ORGANIZATION, CONTACT_EMAIL,
    MODEL_NAME, REVISION, HF_REPO,
    handle_file(RESULTS),
    "Encoder only",          # model type
    "DeBERTa-v2",            # base architecture (custom)
    "AdamW",                 # optimizer
    0.0001,                  # max learning rate
    60,                      # training epochs
    131072,                  # batch size (tokens) = 16/dev x 16 dev x 512
    "character-level (BERT vocab + radicals), 21457",
    12,                      # layers
    12,                      # heads
    512,                     # max sequence length
    103900000,               # total parameters
    "Custom corpus",         # training data
    92910000,                # tokens (jieba)
    DESCRIPTION,
    api_name="/submit_and_refresh",
)
print(result[0] if isinstance(result, (list, tuple)) else result)
