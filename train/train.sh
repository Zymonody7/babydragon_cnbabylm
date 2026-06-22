#!/bin/bash
# From-scratch pretraining: DeBERTa-v2 base (~104M), whole-word MLM + pinyin
# auxiliary objective, on the prebuilt dataset (../dataset).
# Portable: runs on NVIDIA CUDA (nccl) or Ascend 910 (torch_npu/hccl). train.py
# auto-selects the device/backend; this launcher only sets up the environment.
set -eo pipefail
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29512}

# Ascend NPU: source its toolkit if present (no-op on NVIDIA hosts).
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Number of processes = one per device. Override with NPROC=<n>.
if [ -z "${NPROC:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi -L | wc -l)
  elif command -v npu-smi >/dev/null 2>&1; then
    NPROC=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo 8)
  else
    NPROC=1
  fi
fi
# Keep the global batch ~ the original 16 cards x 16 = 256 samples:
# per-device batch 16, grad-accum scales inversely with device count.
GRAD_ACCUM=${GRAD_ACCUM:-$(( 16 / NPROC > 0 ? 16 / NPROC : 1 ))}

echo "=== pretraining start $(date) | NPROC=$NPROC grad_accum=$GRAD_ACCUM ==="
torchrun --nproc_per_node=$NPROC train.py \
    --data_dir ../dataset --output_dir ../output \
    --vocab_size 21457 \
    --hidden_size 768 --num_hidden_layers 12 --num_attention_heads 12 \
    --intermediate_size 3072 \
    --batch_size 16 --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.05 \
    --num_epochs 60 --mlm_probability 0.15 --pinyin_weight 0.3 \
    --save_epochs 10 --log_steps 50 --fp16 --seed 42
echo "=== pretraining done $(date) ==="
