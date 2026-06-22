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

# Resilient download: retry transient network/SSL failures.
fetch() {  # url  dest
  curl -fSL --retry 8 --retry-delay 3 --retry-all-errors \
       --connect-timeout 20 --max-time 600 -o "$2" "$1"
}

# 1. CJKVI-IDS — character (ideographic) decomposition database.
#    Source: https://github.com/cjkvi/cjkvi-ids  (public domain / CC)
fetch "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt" \
      "$SUP/cjkvi-ids.txt"

# 2. chaizi — Chinese character split (拆字) database.
#    Source: https://github.com/kfcd/chaizi  (open)
fetch "https://raw.githubusercontent.com/kfcd/chaizi/master/chaizi-jt.txt" \
      "$SUP/chaizi/chaizi-jt.txt"

# 3. polyphone dictionary (多音字).
#    Source: https://github.com/mapull/chinese-dictionary  (open)
fetch "https://raw.githubusercontent.com/mapull/chinese-dictionary/main/character/polyphone.json" \
      "$SUP/chinese-dictionary/character/polyphone.json"

echo "Raw reference data fetched into $SUP"
ls -la "$SUP" "$SUP/chaizi" "$SUP/chinese-dictionary/character"
