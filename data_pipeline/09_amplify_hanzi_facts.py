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
TARGET_FACT_TOKENS = 7_000_000
# v2's REAL training data = data/*.npy = 105.4M char-tokens (the source text mix
# is gone, so we append amplified facts as npy onto v2's existing tensors).
BASE_TOKENS = 105_400_000
BUDGET = 100_000_000


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

    total = BASE_TOKENS + amp_tok
    print(f"facts (unique): {len(facts):,} recs, {fact_tok:,} tok")
    print(f"amplified x{rep}: {len(amp):,} recs, +{amp_tok:,} tok")
    print(f"total ≈ base {BASE_TOKENS:,} + {amp_tok:,} = {total:,} tok"
          f"  (budget {BUDGET:,}: {'OK' if total <= BUDGET else 'OVER!'})")
    print(f"hanzi-fact share ≈ {100*amp_tok/total:.1f}%")
    print(f"wrote amplified facts -> {OUT}")


if __name__ == "__main__":
    main()
