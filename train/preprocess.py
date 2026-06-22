#!/usr/bin/env python3
"""Preprocess training data for DeBERTa-v2 MLM + Pinyin auxiliary training.

Steps:
  1. Segment texts with jieba (for whole-word masking)
  2. Tokenize with bert-base-chinese tokenizer (character-level for CJK)
  3. Compute pinyin labels for CJK character tokens
  4. Pack into fixed-length sequences
  5. Save as numpy arrays + tokenizer + pinyin vocab

Usage:
    pip install pypinyin jieba transformers
    python preprocess.py --input ../data/mixed/train.jsonl --output_dir data
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import jieba
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import pypinyin
    from pypinyin import Style

    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


def is_cjk_char(c: str) -> bool:
    cp = ord(c)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--input", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--output_dir", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="bert-base-chinese")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--no_pinyin", action="store_true", help="Skip pinyin labels (if pypinyin unavailable)")
    args = parser.parse_args()

    use_pinyin = HAS_PYPINYIN and not args.no_pinyin
    if not use_pinyin:
        print("Pinyin labels: DISABLED" + (" (pypinyin not installed)" if not HAS_PYPINYIN else ""))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    max_content = args.max_length - 2

    # Load texts
    print(f"Loading texts from {args.input} ...")
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text", "").strip()
            if text:
                texts.append(text)
    print(f"  Loaded {len(texts):,} texts")

    # Pass 1: tokenize, segment, collect pinyin
    print("Tokenizing and collecting pinyin ...")
    all_tokens: list[int] = []
    all_word_ids: list[int] = []
    all_pinyin_ids: list[int] = []
    pinyin_vocab: OrderedDict[str, int] = OrderedDict()
    word_counter = 0

    for text in tqdm(texts, desc="Processing"):
        words = list(jieba.cut(text))
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_counter += 1
                continue

            is_pure_cjk = all(is_cjk_char(c) for c in word) and len(word) > 0
            if use_pinyin and is_pure_cjk and len(word_tokens) == len(word):
                word_pinyins = pypinyin.pinyin(word, style=Style.TONE3)
                for token, py_list in zip(word_tokens, word_pinyins):
                    tid = tokenizer.convert_tokens_to_ids(token)
                    py = py_list[0]
                    if py not in pinyin_vocab:
                        pinyin_vocab[py] = len(pinyin_vocab)
                    all_tokens.append(tid)
                    all_word_ids.append(word_counter)
                    all_pinyin_ids.append(pinyin_vocab[py])
            else:
                for token in word_tokens:
                    tid = tokenizer.convert_tokens_to_ids(token)
                    all_tokens.append(tid)
                    all_word_ids.append(word_counter)
                    all_pinyin_ids.append(-100)

            word_counter += 1

    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Pinyin vocab size: {len(pinyin_vocab)}")

    # Pack into fixed-length sequences
    n_sequences = total_tokens // max_content
    print(f"Packing into {n_sequences:,} sequences of length {args.max_length}")

    token_array = np.full((n_sequences, args.max_length), pad_id, dtype=np.int32)
    word_array = np.full((n_sequences, args.max_length), -1, dtype=np.int32)
    pinyin_array = np.full((n_sequences, args.max_length), -100, dtype=np.int32)

    for i in tqdm(range(n_sequences), desc="Packing"):
        start = i * max_content
        end = start + max_content
        content_tokens = all_tokens[start:end]
        content_words = all_word_ids[start:end]
        content_pinyin = all_pinyin_ids[start:end]
        n = len(content_tokens)

        # Remap word IDs to be local within each sequence
        seen: dict[int, int] = {}
        local_words = []
        for w in content_words:
            if w not in seen:
                seen[w] = len(seen)
            local_words.append(seen[w])

        # Fill arrays: [CLS] content... [SEP] [PAD]...
        token_array[i, 0] = cls_id
        token_array[i, 1 : n + 1] = content_tokens
        token_array[i, n + 1] = sep_id

        word_array[i, 1 : n + 1] = local_words

        pinyin_array[i, 1 : n + 1] = content_pinyin

    # Save outputs
    np.save(args.output_dir / "token_ids.npy", token_array)
    np.save(args.output_dir / "word_ids.npy", word_array)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  token_ids.npy  : {token_array.shape}  ({token_array.nbytes / 1e6:.1f} MB)")
    print(f"  word_ids.npy   : {word_array.shape}  ({word_array.nbytes / 1e6:.1f} MB)")

    if use_pinyin:
        np.save(args.output_dir / "pinyin_ids.npy", pinyin_array)
        with open(args.output_dir / "pinyin_vocab.json", "w", encoding="utf-8") as f:
            json.dump({"vocab": pinyin_vocab, "size": len(pinyin_vocab)}, f, ensure_ascii=False, indent=2)
        print(f"  pinyin_ids.npy : {pinyin_array.shape}  ({pinyin_array.nbytes / 1e6:.1f} MB)")
        print(f"  pinyin_vocab   : {len(pinyin_vocab)} entries")

    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    print(f"  tokenizer      : saved to {args.output_dir / 'tokenizer'}")


if __name__ == "__main__":
    main()
