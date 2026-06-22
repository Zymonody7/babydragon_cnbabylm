#!/bin/bash
# Convenience wrapper: data processing + pretraining, end to end.
#   - process: run_process.sh        (raw data -> ./dataset/)
#   - train:   train/train.sh        (./dataset/ -> ./output/final)
# To train on a prebuilt dataset/ (skip processing), run train/train.sh directly.
set -eo pipefail
cd "$(dirname "$0")"
bash run_process.sh
( cd train && bash train.sh )
echo "Done. Trained model in ./output/final. Evaluate with eval/ and submit with submit/."
