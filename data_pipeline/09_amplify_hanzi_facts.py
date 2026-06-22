#!/usr/bin/env python3
"""Amplify the hanzi (pinyin+structure) facts to the target token count.

Incremental injection: the ONLY difference from v2's training data is the added
pinyin + structure fact sentences (amplified by repetition). This isolates the
effect of the HANZI knowledge injection for clean attribution.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

DATA = Path(__file__).resolve().parent.parent / "data"
FACT_FILES = [
    DATA / "supplementary" / "hanzi_corpus" / "pinyin_facts.jsonl",
    DATA / "supplementary" / "hanzi_corpus" / "structure_facts.jsonl",
]
OUT = DATA / "facts" / "hanzi_facts.jsonl"
TARGET_FACT_TOKENS = 7_000_000   # target amplified hanzi-fact size (jieba words)
# Reference only: base corpus ≈ 86M jieba words (≈105M char tokens in the npy).
# The authoritative ≤102M-jieba-word budget is checked on the assembled corpus,
# not here — this script just amplifies facts to TARGET_FACT_TOKENS.


def load(p):
    out = []
    bad = 0
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  [warn] skipped {bad} malformed lines in {p.name}")
    return out


def toks(r):
    return r.get("num-tokens", len(r["text"]))


def main():
    facts = []
    for f in FACT_FILES:
        facts += load(f)
    fact_tok = sum(toks(r) for r in facts)

    rep = max(1, TARGET_FACT_TOKENS // max(1, fact_tok))
    amp = facts * rep
    amp_tok = fact_tok * rep
    random.shuffle(amp)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in amp:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"facts (unique): {len(facts):,} recs, {fact_tok:,} jieba words")
    print(f"amplified x{rep}: {len(amp):,} recs, +{amp_tok:,} jieba words")
    print(f"wrote amplified hanzi facts -> {OUT}")
    print("(budget is verified in jieba words on the assembled corpus; "
          "base ≈86M + all facts ≈6.8M ≈ 92.9M ≤ 102M)")


if __name__ == "__main__":
    main()
