#!/bin/bash
# Fetch the external raw reference data needed by the pipeline into ./data/.
# Run from the data_pipeline/ directory: bash 00_fetch_raw.sh
#
# The official ≤100M Chinese corpus itself is downloaded automatically by
# 01_analyze_official.py via the `datasets` library
# (chinese-babylm-org/babylm-zho-100M), so it is NOT fetched here.
set -eo pipefail
SUP="../data/supplementary"
mkdir -p "$SUP" "$SUP/chaizi" "$SUP/chinese-dictionary/character"

# 1. CJKVI-IDS — character (ideographic) decomposition database.
#    Source: https://github.com/cjkvi/cjkvi-ids  (public domain / CC)
curl -fSL "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt" \
     -o "$SUP/cjkvi-ids.txt"

# 2. chaizi — Chinese character split (拆字) database.
#    Source: https://github.com/kfcd/chaizi  (open)
curl -fSL "https://raw.githubusercontent.com/kfcd/chaizi/master/chaizi-jt.txt" \
     -o "$SUP/chaizi/chaizi-jt.txt"

# 3. polyphone dictionary (多音字).
#    Source: https://github.com/mapull/chinese-dictionary  (open)
curl -fSL "https://raw.githubusercontent.com/mapull/chinese-dictionary/main/character/polyphone.json" \
     -o "$SUP/chinese-dictionary/character/polyphone.json"

echo "Raw reference data fetched into $SUP"
ls -la "$SUP" "$SUP/chaizi" "$SUP/chinese-dictionary/character"
