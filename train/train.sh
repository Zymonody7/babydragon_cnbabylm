#!/bin/bash
# From-scratch pretraining: DeBERTa-v2 base (~104M), whole-word MLM + pinyin
# auxiliary objective, on the data-efficient Chinese mix + injected facts.
# 16-card Ascend 910 DDP.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MASTER_ADDR=localhost
export MASTER_PORT=29512
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Run from the train/ directory. ../dataset is the packed training set
# (see data_pipeline/13_build_npy.py); ../output holds the checkpoints.
echo "=== pretraining start $(date) ==="
torchrun --nproc_per_node=16 train.py \
    --data_dir ../dataset --output_dir ../output \
    --vocab_size 21457 \
    --hidden_size 768 --num_hidden_layers 12 --num_attention_heads 12 \
    --intermediate_size 3072 \
    --batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.05 \
    --num_epochs 60 --mlm_probability 0.15 --pinyin_weight 0.3 \
    --save_epochs 10 --log_steps 50 --fp16 --seed 42
echo "=== pretraining done $(date) ==="
