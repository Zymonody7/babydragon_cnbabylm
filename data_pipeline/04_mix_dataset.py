"""
Mix official data with supplementary data.
Strategy:
  - Reduce subtitles from 56.8% to ~35% (cut ~20M tokens)
  - Keep all other official categories
  - Add hanzi-specific data (repeat to amplify)
  - Add children's reading data (repeat to amplify)
  - Stay within 100M word budget
"""
import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "mixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUDGET = 100_000_000  # 100M tokens


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def count_tokens_jieba(text):
    import jieba
    return len(list(jieba.cut(text)))


def main():
    print("加载官方数据...")
    official = load_jsonl(DATA_DIR / "official" / "official_data.jsonl")
    print(f"  官方数据: {len(official)} 条")

    # Split by category
    subtitles = [r for r in official if r["category"] == "subtitles"]
    non_subtitles = [r for r in official if r["category"] != "subtitles"]

    total_non_sub_tokens = sum(r["num-tokens"] for r in non_subtitles)
    total_sub_tokens = sum(r["num-tokens"] for r in subtitles)
    print(f"  字幕: {len(subtitles)} 条, {total_sub_tokens:,} tokens")
    print(f"  非字幕: {len(non_subtitles)} 条, {total_non_sub_tokens:,} tokens")

    # Keep about 60% of subtitles (reduce from 56.8% to ~38%)
    random.shuffle(subtitles)
    target_sub_tokens = int(total_sub_tokens * 0.6)
    kept_subtitles = []
    running = 0
    for r in subtitles:
        if running >= target_sub_tokens:
            break
        kept_subtitles.append(r)
        running += r["num-tokens"]

    kept_sub_tokens = sum(r["num-tokens"] for r in kept_subtitles)
    print(f"\n砍减字幕: {len(kept_subtitles)} 条, {kept_sub_tokens:,} tokens (原来 {total_sub_tokens:,})")

    # Load supplementary data
    print("\n加载补充数据...")
    hanzi = load_jsonl(DATA_DIR / "supplementary" / "hanzi_corpus" / "hanzi_corpus.jsonl")
    children = load_jsonl(DATA_DIR / "supplementary" / "children_corpus" / "children_corpus.jsonl")
    print(f"  汉字专项: {len(hanzi)} 条")
    print(f"  儿童读物: {len(children)} 条")

    # Estimate tokens for supplementary data (rough: chars * 0.6)
    for records in [hanzi, children]:
        for r in records:
            if "num-tokens" not in r:
                r["num-tokens"] = max(1, int(len(r["text"]) * 0.6))

    hanzi_tokens = sum(r["num-tokens"] for r in hanzi)
    children_tokens = sum(r["num-tokens"] for r in children)
    print(f"  汉字tokens: {hanzi_tokens:,}")
    print(f"  儿童tokens: {children_tokens:,}")

    # Amplify supplementary data by repeating
    # Target: hanzi ~2M tokens, children ~1M tokens
    hanzi_repeat = max(1, 2_000_000 // max(1, hanzi_tokens))
    children_repeat = max(1, 1_000_000 // max(1, children_tokens))
    print(f"\n数据重复倍数: 汉字 x{hanzi_repeat}, 儿童 x{children_repeat}")

    hanzi_amplified = hanzi * hanzi_repeat
    children_amplified = children * children_repeat

    hanzi_amp_tokens = sum(r["num-tokens"] for r in hanzi_amplified)
    children_amp_tokens = sum(r["num-tokens"] for r in children_amplified)

    # Combine all
    all_data = non_subtitles + kept_subtitles + hanzi_amplified + children_amplified
    total_tokens = sum(r["num-tokens"] for r in all_data)

    print(f"\n=== 混合后统计 ===")
    print(f"总条数: {len(all_data):,}")
    print(f"总tokens: {total_tokens:,}")
    print(f"预算: {BUDGET:,}")
    print(f"剩余: {BUDGET - total_tokens:,}")

    if total_tokens > BUDGET:
        print(f"\n超出预算! 需要进一步裁剪字幕...")
        excess = total_tokens - BUDGET
        # Remove more subtitles
        random.shuffle(kept_subtitles)
        remove_tokens = 0
        remove_count = 0
        for r in kept_subtitles:
            if remove_tokens >= excess:
                break
            remove_tokens += r["num-tokens"]
            remove_count += 1
        kept_subtitles = kept_subtitles[remove_count:]
        all_data = non_subtitles + kept_subtitles + hanzi_amplified + children_amplified
        total_tokens = sum(r["num-tokens"] for r in all_data)
        print(f"裁剪后总tokens: {total_tokens:,}")

    # Category distribution
    cat_counter = Counter()
    cat_tokens = Counter()
    for r in all_data:
        cat = r["category"]
        cat_counter[cat] += 1
        cat_tokens[cat] += r["num-tokens"]

    print(f"\n=== 各类别分布 ===")
    for cat, tokens in cat_tokens.most_common():
        pct = tokens / total_tokens * 100
        print(f"  {cat}: {cat_counter[cat]:,} 条, {tokens:,} tokens ({pct:.1f}%)")

    # Shuffle and save
    random.shuffle(all_data)
    output_path = OUT_DIR / "train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n已保存到 {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save stats
    stats = {
        "total_records": len(all_data),
        "total_tokens": total_tokens,
        "budget": BUDGET,
        "category_distribution": {
            cat: {"count": cat_counter[cat], "tokens": cat_tokens[cat]}
            for cat in cat_tokens
        },
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
