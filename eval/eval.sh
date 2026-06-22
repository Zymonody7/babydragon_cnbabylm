#!/bin/bash
# Run the official Chinese BabyLM final evaluation pipeline on one model,
# fanning the 14 tasks across NPUs (zero-shot bundled on one card, each
# fine-tune task and cogbench on their own card). Run from the cloned
# chinese-babylm-pipeline-final repo, with eval/config.yaml copied in.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
CFG=config.yaml
L=logs/eval; mkdir -p $L
ZS="zhoblimp xcomps_zh hanzi_structure hanzi_pinyin hanzi_structure_hidden hanzi_pinyin_hidden"
run(){ local dev=$1 tag=$2; shift 2
  ASCEND_RT_VISIBLE_DEVICES=$dev python pipeline.py eval -c $CFG --tasks "$@" > $L/${tag}.log 2>&1 & }

run 0 zs        $ZS
run 1 afqmc     afqmc
run 2 ocnli     ocnli
run 3 tnews     tnews
run 4 cluewsc   cluewsc2020
run 5 c3        c3
run 6 dnli      diagnostic_nli
run 7 cog       word_fmri fmri
wait
echo "ALL_EVAL_DONE $(date)"

# Export the leaderboard JSON:
#   python pipeline.py gather -c config.yaml --export results.json
