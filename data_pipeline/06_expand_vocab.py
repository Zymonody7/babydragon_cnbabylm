#!/usr/bin/env python3
"""Expand the tokenizer vocab with OOV radical components by SURGICALLY appending
them to tokenizer.json's model.vocab (ids 21128..), preserving ALL original ids
exactly (so v2's npy + resized embeddings stay aligned). CJK chars are split
individually by the WordPiece pre-tokenizer, so appended radicals tokenize fine.

Output: data/tokenizer_expanded/
"""
from __future__ import annotations

import collections
import json
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
IDS = DATA / "supplementary" / "cjkvi-ids.txt"
SRC = DATA / "tokenizer"
OUT = DATA / "tokenizer_expanded"
OPS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def is_cjk(c): return len(c) == 1 and "一" <= c <= "鿿"


def main():
    tj = json.load(open(SRC / "tokenizer.json", encoding="utf-8"))
    vocab = tj["model"]["vocab"]            # {token: id}, original
    orig_size = len(vocab)
    vset = set(vocab.keys())

    comp = collections.Counter()
    for raw in open(IDS, encoding="utf-8"):
        if raw.startswith("#") or "\t" not in raw:
            continue
        cols = raw.rstrip("\n").split("\t")
        if len(cols) < 3:
            continue
        ch = cols[1]
        if not is_cjk(ch) or ch not in vset:
            continue
        for c in cols[2].split("[")[0]:
            if is_cjk(c) and c not in OPS:
                comp[c] += 1
    oov = sorted([c for c in comp if c not in vset], key=lambda c: -comp[c])
    print(f"distinct components: {len(comp)} | OOV radicals to append: {len(oov)}")

    nxt = max(vocab.values()) + 1
    appended = []
    for c in oov:
        vocab[c] = nxt
        appended.append(c)
        nxt += 1

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(tj, open(OUT / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
    for f in ("tokenizer_config.json", "special_tokens_map.json"):
        if (SRC / f).exists():
            shutil.copy(SRC / f, OUT / f)
    # vocab.txt aligned by id order
    inv = {i: t for t, i in vocab.items()}
    with open(OUT / "vocab.txt", "w", encoding="utf-8") as f:
        for i in range(len(vocab)):
            f.write(inv[i] + "\n")
    print(f"vocab {orig_size} -> {len(vocab)} (+{len(appended)}) -> {OUT}")

    # smoke: original ids unchanged + radicals tokenize correctly in a sentence
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(OUT)
    s = "涙字左边的部分是氵。"
    ids = t.convert_tokens_to_ids(t.tokenize(s))
    print(f"sentence tokens: {t.tokenize(s)}")
    print(f"sentence ids:    {ids}")
    print(f"氵->{t.convert_tokens_to_ids('氵')} (expect {vocab['氵']}), 扌->{t.convert_tokens_to_ids('扌')}")
    print(f"original-id check 的->{t.convert_tokens_to_ids('的')} 是->{t.convert_tokens_to_ids('是')}")


if __name__ == "__main__":
    main()
