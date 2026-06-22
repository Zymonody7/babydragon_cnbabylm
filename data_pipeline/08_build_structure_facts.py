#!/usr/bin/env python3
"""FINAL structure facts: diversified frames; WHOLE char in-distribution (base
freq>=5, drops traditional), COMPONENTS from tokenizer_expanded vocab (keeps radicals
which don't appear standalone in base). Facts from CJKVI-IDS. NO LLM.
Output: data/supplementary/hanzi_corpus/structure_facts.jsonl
"""
from __future__ import annotations
import json, random
from pathlib import Path

random.seed(42)
DATA = Path(__file__).resolve().parent.parent / "data"
IDS = DATA / "supplementary" / "cjkvi-ids.txt"
TOK = DATA / "tokenizer_expanded"
OUT = DATA / "supplementary" / "hanzi_corpus" / "structure_facts.jsonl"
BINARY = {"⿰": ("左边", "右边"), "⿱": ("上边", "下边"), "⿴": ("外围", "内部"), "⿵": ("外围", "内部"),
          "⿶": ("外围", "内部"), "⿷": ("外围", "内部"), "⿸": ("外围", "内部"), "⿹": ("外围", "内部"), "⿺": ("外围", "内部")}
TERNARY = {"⿲": ("左边", "中间", "右边"), "⿳": ("上边", "中间", "下边")}
ARITY = {**{o: 2 for o in BINARY}, **{o: 3 for o in TERNARY}, "⿻": 2}
# EXACT eval frames (English quotes): whole2component + component2whole
W2C_EXACT = '"{c}"字{pos}的部分是"{x}"。'
W2C_VAR = ['"{c}"的{pos}是"{x}"。', '把"{c}"拆开，{pos}的部分是"{x}"。']
C2W_TAIL = '，这个字是"{c}"。'        # parts + tail = exact component2whole eval frame
COMPOSE = ['"{c}"由{cl}组成。', '"{c}"是由{cl}构成的。']


def is_cjk(c): return len(c) == 1 and "一" <= c <= "鿿"
def parse(s, i):
    if i >= len(s): return ("leaf", ""), i
    ch = s[i]
    if ch in ARITY:
        kids, j = [], i + 1
        for _ in range(ARITY[ch]):
            node, j = parse(s, j); kids.append(node)
        return (ch, kids), j
    return ("leaf", ch), i + 1


def main():
    vocab = {l.strip() for l in open(TOK / "vocab.txt", encoding="utf-8")}
    indist = set(open(DATA / "chars_indist.txt", encoding="utf-8").read())   # whole-char filter
    lines, n = [], 0
    for raw in open(IDS, encoding="utf-8"):
        if raw.startswith("#") or "\t" not in raw: continue
        cols = raw.rstrip("\n").split("\t")
        if len(cols) < 3: continue
        char = cols[1]
        if not is_cjk(char) or char not in vocab or char not in indist: continue   # whole char in-dist
        ids = cols[2].split("[")[0].strip()
        if not ids or (ids[0] not in BINARY and ids[0] not in TERNARY): continue
        tree, _ = parse(ids, 0); op, kids = tree
        if op not in BINARY and op not in TERNARY: continue
        positions = BINARY.get(op) or TERNARY.get(op)
        comps, ok = [], True
        for pos, kid in zip(positions, kids):
            if kid[0] == "leaf" and is_cjk(kid[1]) and kid[1] in vocab:   # component in vocab (radicals ok)
                comps.append((pos, kid[1]))
            else:
                ok = False
        if not comps: continue
        n += 1
        for pos, comp in comps:
            for _ in range(6):                                            # EXACT W2C frame x6 (heavy whole2component boost)
                lines.append(W2C_EXACT.format(c=char, pos=pos, x=comp))
            lines.append(random.choice(W2C_VAR).format(c=char, pos=pos, x=comp))  # +variant
        if ok and len(comps) == len(kids) >= 2:
            parts = "，".join(f'{pos}是"{comp}"' for pos, comp in comps)    # EXACT component2whole
            cl = "、".join(f'"{comp}"' for _, comp in comps)
            for _ in range(2):                                            # EXACT C2W frame x2
                lines.append(parts + C2W_TAIL.format(c=char))
            lines.append(random.choice(COMPOSE).format(c=char, cl=cl))
    random.shuffle(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for t in lines:
            f.write(json.dumps({"text": t, "category": "hanzi-knowledge",
                                "data-source": "self-built-structure", "num-tokens": len(t)}, ensure_ascii=False) + "\n")
    print(f"in-dist whole chars decomposed: {n} | wrote {len(lines):,} structure facts -> {OUT}")


if __name__ == "__main__":
    main()
