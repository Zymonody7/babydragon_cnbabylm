#!/usr/bin/env python3
"""Combine all injected facts = amplified hanzi facts (pinyin+structure) +
amplified grammar facts (ZhoBLiMP weak-phenomenon constructions, x30 for
construction exposure). Output: data/facts/all_facts.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
HANZI = DATA / "facts" / "hanzi_facts.jsonl"                 # amplified hanzi (pinyin+structure) facts
GRAMMAR = DATA / "supplementary" / "grammar_corpus" / "grammar_facts.jsonl"
OUT = DATA / "facts" / "all_facts.jsonl"
GRAMMAR_REP = 30


def wc(path):
    return sum(json.loads(l).get("num-tokens", 0) for l in open(path, encoding="utf-8"))


def main():
    gram = [l for l in open(GRAMMAR, encoding="utf-8")]
    gram_tok = wc(GRAMMAR)
    hanzi_tok = wc(HANZI)
    with open(OUT, "w", encoding="utf-8") as f:
        for l in open(HANZI, encoding="utf-8"):     # HANZI facts as-is
            f.write(l)
        for _ in range(GRAMMAR_REP):                 # grammar amplified
            for l in gram:
                f.write(l)
    gram_amp = gram_tok * GRAMMAR_REP
    print(f"HANZI facts: {hanzi_tok:,} tok | grammar: {len(gram):,} x{GRAMMAR_REP} = {gram_amp:,} tok")
    print(f"facts total: {hanzi_tok + gram_amp:,} tok -> {OUT}")
    print(f"est. base(86.15M) + facts(jieba ~) -- check budget after preprocess")


if __name__ == "__main__":
    main()
