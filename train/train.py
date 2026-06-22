#!/usr/bin/env python3
"""Train DeBERTa-v2 with Whole-Word MLM + Pinyin Auxiliary task.

Architecture: DeBERTa-v2 (disentangled attention, relative position encoding)
Primary objective: Masked Language Modeling with whole-word masking
Auxiliary objective: Pinyin prediction for masked CJK character positions

Compatible with: CUDA GPUs, Ascend 910 NPUs (via torch_npu)
Output: HuggingFace-compatible model directory (works with eval pipeline)

Usage:
    # Single device
    python train.py --data_dir data --output_dir output

    # Multi-device (DDP)
    torchrun --nproc_per_node=8 train.py --data_dir data --output_dir output
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoTokenizer,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except ImportError:
    HAS_NPU = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeBERTa-v2 MLM + Pinyin")

    # Data / IO
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("output"))

    # Model architecture
    p.add_argument("--vocab_size", type=int, default=21128)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--num_attention_heads", type=int, default=12)
    p.add_argument("--intermediate_size", type=int, default=3072)
    p.add_argument("--max_position_embeddings", type=int, default=512)

    # Training
    p.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--fp16", action="store_true", help="Use mixed precision")
    p.add_argument("--seed", type=int, default=42)

    # Pinyin auxiliary
    p.add_argument("--pinyin_weight", type=float, default=0.1, help="Weight for pinyin loss (0 to disable)")

    # Logging & checkpointing
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--save_epochs", type=int, default=5, help="Save every N epochs")

    # Resume from checkpoint
    p.add_argument("--resume_from", type=Path, default=None, help="Resume from HF checkpoint dir")

    # DDP (set by torchrun)
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        backend = "hccl" if HAS_NPU else "nccl"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1
    return rank, local_rank, world_size


def get_device(local_rank: int) -> torch.device:
    if HAS_NPU and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset & Collator
# ---------------------------------------------------------------------------

class PackedMLMDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.token_ids = np.load(data_dir / "token_ids.npy", mmap_mode="r")
        self.word_ids = np.load(data_dir / "word_ids.npy", mmap_mode="r")
        pinyin_path = data_dir / "pinyin_ids.npy"
        self.pinyin_ids = np.load(pinyin_path, mmap_mode="r") if pinyin_path.exists() else None

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int):
        item = {
            "token_ids": torch.from_numpy(self.token_ids[idx].copy()).long(),
            "word_ids": torch.from_numpy(self.word_ids[idx].copy()).long(),
        }
        if self.pinyin_ids is not None:
            item["pinyin_ids"] = torch.from_numpy(self.pinyin_ids[idx].copy()).long()
        return item


class WholeWordMaskCollator:
    def __init__(self, pad_id: int, mask_id: int, vocab_size: int, mlm_prob: float = 0.15):
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.mlm_prob = mlm_prob

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        token_ids = torch.stack([b["token_ids"] for b in batch])
        word_ids = torch.stack([b["word_ids"] for b in batch])
        has_pinyin = "pinyin_ids" in batch[0]
        if has_pinyin:
            pinyin_ids = torch.stack([b["pinyin_ids"] for b in batch])

        bsz, seq_len = token_ids.shape
        input_ids = token_ids.clone()
        mlm_labels = torch.full_like(token_ids, -100)
        pinyin_labels = torch.full_like(token_ids, -100) if has_pinyin else None

        for i in range(bsz):
            maskable = word_ids[i] >= 0
            if not maskable.any():
                continue

            unique_words = word_ids[i][maskable].unique()
            n_mask = max(1, int(len(unique_words) * self.mlm_prob))
            selected = set(unique_words[torch.randperm(len(unique_words))[:n_mask]].tolist())

            token_mask = torch.zeros(seq_len, dtype=torch.bool)
            for j in range(seq_len):
                if word_ids[i, j].item() in selected:
                    token_mask[j] = True

            rand = torch.rand(seq_len)
            replace_mask = token_mask & (rand < 0.8)
            random_mask = token_mask & (rand >= 0.8) & (rand < 0.9)

            input_ids[i, replace_mask] = self.mask_id
            input_ids[i, random_mask] = torch.randint(0, self.vocab_size, (random_mask.sum(),))

            mlm_labels[i, token_mask] = token_ids[i, token_mask]

            if has_pinyin:
                pinyin_labels[i, token_mask] = pinyin_ids[i, token_mask]

        # Attention mask from original tokens (before masking)
        attention_mask = (token_ids != self.pad_id).long()

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": mlm_labels,
        }
        if has_pinyin:
            result["pinyin_labels"] = pinyin_labels
        return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PinyinHead(nn.Module):
    def __init__(self, hidden_size: int, pinyin_vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, pinyin_vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(self.act(self.dense(hidden_states))))


class DebertaMLMWithPinyin(nn.Module):
    def __init__(self, config: DebertaV2Config, pinyin_vocab_size: int = 0, pinyin_weight: float = 0.1):
        super().__init__()
        self.deberta_mlm = DebertaV2ForMaskedLM(config)
        self.pinyin_weight = pinyin_weight
        self.has_pinyin = pinyin_vocab_size > 0 and pinyin_weight > 0
        if self.has_pinyin:
            self.pinyin_head = PinyinHead(config.hidden_size, pinyin_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        pinyin_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.deberta_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=self.has_pinyin,
        )
        mlm_loss = outputs.loss

        pinyin_loss = torch.tensor(0.0, device=mlm_loss.device)
        if self.has_pinyin:
            hidden = outputs.hidden_states[-1]
            pinyin_logits = self.pinyin_head(hidden)
            if pinyin_labels is not None:
                pinyin_loss = F.cross_entropy(
                    pinyin_logits.view(-1, pinyin_logits.size(-1)),
                    pinyin_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                pinyin_loss = (pinyin_logits * 0).sum()
            total_loss = mlm_loss + self.pinyin_weight * pinyin_loss
        else:
            total_loss = mlm_loss

        return total_loss, mlm_loss.detach(), pinyin_loss.detach()

    def save_pretrained(self, path: str | Path) -> None:
        self.deberta_mlm.save_pretrained(path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    is_main = rank == 0

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Config: {json.dumps(vars(args), default=str)}")
        logger.info(f"World size: {world_size} | Device: {device} | NPU: {HAS_NPU}")

    # Seed
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # --- Data ---
    dataset = PackedMLMDataset(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.data_dir / "tokenizer")

    pinyin_vocab_path = args.data_dir / "pinyin_vocab.json"
    pinyin_vocab_size = 0
    if pinyin_vocab_path.exists() and args.pinyin_weight > 0:
        with open(pinyin_vocab_path) as f:
            pinyin_info = json.load(f)
        pinyin_vocab_size = pinyin_info["size"]
        if is_main:
            logger.info(f"Pinyin vocab: {pinyin_vocab_size} entries, weight={args.pinyin_weight}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    collator = WholeWordMaskCollator(
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        vocab_size=len(tokenizer),
        mlm_prob=args.mlm_probability,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        logger.info(f"Dataset: {len(dataset):,} sequences | Batches/epoch: {len(dataloader):,}")

    # --- Model ---
    config = DebertaV2Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=0,
        relative_attention=True,
        max_relative_positions=args.max_position_embeddings,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        layer_norm_eps=1e-7,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        position_biased_input=False,
    )

    model = DebertaMLMWithPinyin(config, pinyin_vocab_size, args.pinyin_weight)

    if args.resume_from is not None:
        ckpt_mlm = DebertaV2ForMaskedLM.from_pretrained(str(args.resume_from))
        model.deberta_mlm.load_state_dict(ckpt_mlm.state_dict())
        del ckpt_mlm
        if is_main:
            logger.info(f"Resumed deberta_mlm weights from {args.resume_from}")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M) | Trainable: {n_trainable:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # --- Optimizer ---
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias", "norm.weight", "norm.bias"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-6)

    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        logger.info(f"Steps/epoch: {steps_per_epoch:,} | Total: {total_steps:,} | Warmup: {warmup_steps:,}")

    # --- AMP ---
    use_amp = args.fp16 and (torch.cuda.is_available() or HAS_NPU)
    amp_dtype = torch.float16
    device_type = "npu" if HAS_NPU else ("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler(device_type) if use_amp and device_type != "npu" else None

    # --- Training loop ---
    global_step = 0

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_mlm = 0.0
        epoch_py = 0.0
        epoch_steps = 0
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    total_loss, mlm_loss, py_loss = model(**batch)
            else:
                total_loss, mlm_loss, py_loss = model(**batch)

            loss = total_loss / args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += total_loss.item()
            epoch_mlm += mlm_loss.item()
            epoch_py += py_loss.item()
            epoch_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % args.log_steps == 0:
                    avg = epoch_loss / epoch_steps
                    avg_mlm = epoch_mlm / epoch_steps
                    avg_py = epoch_py / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    sps = epoch_steps / elapsed
                    logger.info(
                        f"E{epoch + 1}/{args.num_epochs} S{global_step}/{total_steps} | "
                        f"loss={avg:.4f} mlm={avg_mlm:.4f} py={avg_py:.4f} | "
                        f"lr={lr:.2e} | {sps:.1f} step/s"
                    )

        # Epoch done
        avg = epoch_loss / max(epoch_steps, 1)
        if is_main:
            logger.info(
                f"Epoch {epoch + 1} done | avg_loss={avg:.4f} | {time.time() - t0:.0f}s"
            )

        # Save checkpoint
        if is_main and (epoch + 1) % args.save_epochs == 0:
            save_dir = args.output_dir / f"checkpoint-epoch-{epoch + 1}"
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"Saved → {save_dir}")

    # Final save
    if is_main:
        save_dir = args.output_dir / "final"
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Training complete. Final model → {save_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
