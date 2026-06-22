#!/usr/bin/env python3
"""FINAL pinyin facts: diversified frames + IN-DISTRIBUTION filter (only chars in
data/chars_indist.txt = base-corpus simplified chars, freq>=5). Drops traditional/
ultra-rare chars (25% of vocab, 0 eval loss) so the injection budget concentrates on
tested simplified chars. Facts from pypinyin. NO LLM.
Output: data/supplementary/hanzi_corpus/pinyin_facts.jsonl
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path
from pypinyin import Style, pinyin

random.seed(42)
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "supplementary" / "hanzi_corpus" / "pinyin_facts.jsonl"
LQ, RQ = '"', '"'   # eval uses ENGLISH quotes (official: pretrain data has no Chinese quotes)
MAX_HOMO, N_FRAMES, MAX_DIFF = 8, 2, 3
TONE_CN = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "轻"}
try:
    import jieba; _j = True
except ImportError:
    _j = False


def ntok(t): return len(list(jieba.cut(t))) if _j else len(t)
def _f(c, st, **kw):
    r = pinyin(c, style=st, heteronym=False, errors="ignore", **kw)
    return r[0][0] if r and r[0] else ""


def load_chars():
    indist = set(open(DATA / "chars_indist.txt", encoding="utf-8").read())
    vocab = {l.strip() for l in open(DATA / "tokenizer_expanded" / "vocab.txt", encoding="utf-8")}
    return [c for c in indist if c in vocab and "一" <= c <= "鿿"]


HOMO = ["{a}和{b}的声母、韵母和声调完全相同。", "{a}和{b}读音完全一样。", "{a}和{b}是同音字。",
        "{a}和{b}同音。", "{a}跟{b}的发音完全相同。", "{a}的读音和{b}相同。",
        "在普通话里，{a}和{b}读音相同。", "{a}、{b}两个字读音一致。", "{a}和{b}读起来一模一样。",
        "{a}与{b}的拼音相同。", "{a}和{b}都读{py}。", "{a}和{b}发音相同，都念{py}。"]
DIFF = ["{a}和{b}读音不同。", "{a}读{pa}，{b}读{pb}，读音不一样。", "{a}和{b}不是同音字。",
        "{a}念{pa}，{b}念{pb}。", "{a}与{b}的读音并不相同。"]
READ = ["{a}的拼音是{py}。", "{a}读作{py}。", "{a}字念{py}。", "{a}的读音是{py}。",
        "{a}念作{py}。", "“{a}”这个字读{py}。"]


def main():
    chars = load_chars()
    c2t, mark, groups = {}, {}, defaultdict(list)
    for c in chars:
        t = _f(c, Style.TONE3)
        if t:
            c2t[c] = t; mark[c] = _f(c, Style.TONE); groups[t].append(c)
    q = lambda x: LQ + x + RQ
    lines = []
    for c in chars:
        if c not in c2t:
            continue
        py = mark[c]
        homos = [x for x in groups[c2t[c]] if x != c]; random.shuffle(homos)
        for b in homos[:MAX_HOMO]:
            lines.append(HOMO[0].format(a=q(c), b=q(b), py=py))   # EXACT eval frame (always)
            lines.append(random.choice(HOMO[1:]).format(a=q(c), b=q(b), py=py))  # +1 variant for robustness
        diffs = [x for x in chars if x in c2t and c2t[x] != c2t[c]]
        for b in random.sample(diffs, min(MAX_DIFF, len(diffs))):
            lines.append(random.choice(DIFF).format(a=q(c), b=q(b), pa=py, pb=mark[b]))
        lines.append(random.choice(READ).format(a=c, py=py))
        sm, ym = _f(c, Style.INITIALS, strict=False), _f(c, Style.FINALS, strict=False)
        tn = c2t[c][-1] if c2t[c][-1].isdigit() else "5"
        lines.append(f"{q(c)}的声母是{sm if sm else '零声母'}，韵母是{ym}，声调是第{TONE_CN.get(tn, tn)}声。")
    random.shuffle(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for t in lines:
            f.write(json.dumps({"text": t, "category": "hanzi-knowledge",
                                "data-source": "self-built-pinyin", "num-tokens": ntok(t)}, ensure_ascii=False) + "\n")
    print(f"in-dist chars: {len(chars)} | wrote {len(lines):,} pinyin facts (diversified+filtered) -> {OUT}")


if __name__ == "__main__":
    main()
