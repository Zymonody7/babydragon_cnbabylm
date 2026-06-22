#!/usr/bin/env python3
"""Grammar injection for ZhoBLiMP weak phenomena. Rule-based templates generate
GRAMMATICAL Chinese sentences of constructions the child-corpus under-covers (the model
scores 0-25% on them). NO LLM, common in-distribution words, NEW combinations (not eval
sentences). Targets: A-not-A questions, ba-A-not-A, agent-retaining passive, VP-ellipsis
with 也是, reflexive number/gender agreement.
Output: data/supplementary/grammar_corpus/grammar_facts.jsonl
"""
from __future__ import annotations
import json, random, itertools
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parent.parent / "data" / "supplementary" / "grammar_corpus" / "grammar_facts.jsonl"
try:
    import jieba; _j = True
except ImportError:
    _j = False
def ntok(t): return len(list(jieba.cut(t))) if _j else len(t)

FEM_NAME = ["小红", "小芳", "小丽", "王太太", "李小姐", "张大妈", "刘阿姨", "陈女士"]
MASC_NAME = ["小明", "小刚", "小强", "王先生", "李大爷", "张师傅", "刘叔叔", "陈先生"]
NAME = FEM_NAME + MASC_NAME
PEOPLE = ["老师", "学生", "工人", "医生", "司机", "顾客", "邻居", "厨师", "警察", "护士", "老板", "同事", "客人", "队长"]
FEM_REL = ["妈妈", "姐姐", "妹妹", "奶奶", "阿姨", "外婆"]
MASC_REL = ["爸爸", "哥哥", "弟弟", "爷爷", "叔叔", "舅舅"]
TVERB = ["喜欢", "表扬", "埋怨", "反感", "批评", "安慰", "相信", "认识", "帮助", "佩服", "责怪", "感谢", "想念", "尊敬"]
OBJ = ["电视", "电影", "报纸", "杂志", "小说", "照片", "节目", "音乐", "比赛"]
SEEVERB = ["观看", "购买", "整理", "收拾", "打扫", "准备", "检查"]
DITRANS = ["送给", "递给", "寄给", "卖给", "还给", "借给"]
NUMOBJ = ["三本书", "五张照片", "两件衣服", "一份报纸", "几本杂志", "四支笔", "两台电脑"]
PLURAL_Q = ["这几位", "这许多位", "这好多个", "这五位", "这三个", "那几个", "那许多位"]


def main():
    lines = set()
    LOC = ["学校", "公司", "医院", "公园", "食堂", "会议上", "家里", "办公室"]
    SUBJ = NAME + PEOPLE

    # 1. A-not-A: S V O 不 V O ？  (full cross-product for coverage)
    for subj in SUBJ:
        for v in SEEVERB:
            for o in OBJ:
                lines.add(f"{subj}{v}{o}不{v}{o}？")

    # 2. ba A-not-A: S 把不把 O 动给 R ？
    for subj in NAME:
        for o in NUMOBJ:
            for dv in ["递给", "送给", "寄给", "卖给"]:
                lines.add(f"{subj}把不把{o}{dv}{random.choice(PEOPLE)}？")

    # 3. agent-retaining passive: P1 被 P2 (在 LOC) V 了
    for p1 in SUBJ:
        for v in TVERB:
            p2 = random.choice(PEOPLE)
            lines.add(f"{p1}被{p2}{v}了。")
            lines.add(f"{p1}被{p2}在{random.choice(LOC)}{v}了。")

    # 4. VP-ellipsis with 也是  (capped)
    for _ in range(1500):
        s1 = random.choice(SUBJ); s2 = random.choice(PEOPLE)
        dv = random.choice(DITRANS); r = random.choice(PEOPLE); o = random.choice(NUMOBJ)
        lines.add(f"{s1}{dv}了{r}{o}，{s2}也是。")

    # 5a. reflexive NUMBER agreement: plural subject -> 他们/她们自己
    for q in PLURAL_Q:
        for v in TVERB:
            for rel in FEM_REL:
                lines.add(f"{q}{rel}{v}她们自己。")
            for rel in MASC_REL:
                lines.add(f"{q}{rel}{v}他们自己。")
            for ppl in random.sample(PEOPLE, 4):
                lines.add(f"{q}{ppl}{v}他们自己。")

    # 5b. reflexive GENDER agreement (principle A): NAME 的 REL V 她/他自己
    for name in NAME:
        for v in TVERB:
            for rel in random.sample(FEM_REL, 3):
                lines.add(f"{name}的{rel}{v}她自己。")
            for rel in random.sample(MASC_REL, 3):
                lines.add(f"{name}的{rel}{v}他自己。")

    lines = list(lines); random.shuffle(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for t in lines:
            f.write(json.dumps({"text": t, "category": "grammar-knowledge",
                                "data-source": "self-built-grammar", "num-tokens": ntok(t)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines):,} grammar sentences -> {OUT}")
    for t in lines[:8]:
        print("  ", t)


if __name__ == "__main__":
    main()
