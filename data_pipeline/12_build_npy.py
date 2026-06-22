#!/usr/bin/env python3
"""Final training dataset = base npy (token ids 0..21127, character-level base
tokenizer) + facts npy (preprocessed with the EXPANDED tokenizer; radical targets
get ids 21128+; facts pinyin labels set to -100). Copies the expanded tokenizer +
base pinyin_vocab into dataset/.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data"
FACTS = ROOT / "data_facts"
TOK_EXP = ROOT / "data" / "tokenizer_expanded"
OUT = ROOT / "dataset"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bt = np.load(BASE / "token_ids.npy"); bw = np.load(BASE / "word_ids.npy"); bp = np.load(BASE / "pinyin_ids.npy")
    ft = np.load(FACTS / "token_ids.npy"); fw = np.load(FACTS / "word_ids.npy")
    assert bt.shape[1] == ft.shape[1] == 512
    fp = np.full(ft.shape, -100, dtype=np.int32)
    np.save(OUT / "token_ids.npy", np.concatenate([bt, ft], 0))
    np.save(OUT / "word_ids.npy", np.concatenate([bw, fw], 0))
    np.save(OUT / "pinyin_ids.npy", np.concatenate([bp, fp], 0))
    shutil.copy(BASE / "pinyin_vocab.json", OUT / "pinyin_vocab.json")
    if (OUT / "tokenizer").exists():
        shutil.rmtree(OUT / "tokenizer")
    shutil.copytree(TOK_EXP, OUT / "tokenizer")     # expanded tokenizer
    print(f"base {bt.shape[0]:,} + facts {ft.shape[0]:,} = {bt.shape[0]+ft.shape[0]:,} seqs -> {OUT}")
    print(f"facts max token id: {int(ft.max())} (expanded vocab in use if >21127)")


if __name__ == "__main__":
    main()
