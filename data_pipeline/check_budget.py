#!/usr/bin/env python3
"""Authoritative data-budget check in jieba words (competition unit, ≤102M).

Counts the base mix + all injected facts. Run after step 11 (combine_facts):
    python data_pipeline/check_budget.py
"""
from __future__ import annotations
import json
from pathlib import Path

import jieba  # 0.42.1

DATA = Path(__file__).resolve().parent.parent / "data"
BUDGET = 102_000_000
FILES = [DATA / "mixed" / "train.jsonl", DATA / "facts" / "all_facts.jsonl"]


def count(path: Path) -> int:
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n = r.get("num-tokens")
            total += n if isinstance(n, int) else len(list(jieba.cut(r["text"])))
    return total


def main() -> None:
    grand = 0
    for p in FILES:
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        n = count(p)
        grand += n
        print(f"{p.relative_to(DATA.parent)}: {n:,} jieba words")
    verdict = "OK" if grand <= BUDGET else "OVER BUDGET"
    print(f"TOTAL: {grand:,} jieba words / {BUDGET:,} budget -> {verdict}")


if __name__ == "__main__":
    main()
