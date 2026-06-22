#!/usr/bin/env python3
"""Save the set of in-distribution simplified chars = CJK chars appearing >=5 times
in the base corpus. Used to purify pinyin/structure fact injection (drop traditional
& ultra-rare chars that waste budget and aren't tested). Output: data/chars_indist.txt
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BASE = DATA / "mixed" / "train.jsonl"
OUT = DATA / "chars_indist.txt"
cjk = re.compile(r"[一-鿿]")
MIN_FREQ = 5


def main():
    freq = collections.Counter()
    for line in open(BASE, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line).get("text", "")
        except json.JSONDecodeError:
            continue
        for c in cjk.findall(t):
            freq[c] += 1
    keep = sorted(c for c, n in freq.items() if n >= MIN_FREQ)
    OUT.write_text("".join(keep), encoding="utf-8")
    print(f"in-distribution chars (freq>={MIN_FREQ}): {len(keep)} -> {OUT}")


if __name__ == "__main__":
    main()
