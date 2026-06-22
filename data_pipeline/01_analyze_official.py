"""分析官方数据并导出为本地文件，方便后续处理。"""
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OFFICIAL_DIR = DATA_DIR / "official"
OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)

print("加载官方数据集...")
ds = load_dataset("chinese-babylm-org/babylm-zho-100M", split="train")
df = pd.DataFrame(ds)

print(f"总条数: {len(df):,}")
print(f"总token数: {df['num-tokens'].sum():,}")

print("\n=== 按类别统计 ===")
cat_stats = df.groupby("category").agg(
    count=("num-tokens", "count"),
    total_tokens=("num-tokens", "sum"),
    avg_tokens=("num-tokens", "mean"),
).sort_values("total_tokens", ascending=False)
cat_stats["pct"] = cat_stats["total_tokens"] / cat_stats["total_tokens"].sum() * 100
print(cat_stats.to_string())

print("\n=== 按数据源统计（前20）===")
src_stats = df.groupby("data-source").agg(
    count=("num-tokens", "count"),
    total_tokens=("num-tokens", "sum"),
).sort_values("total_tokens", ascending=False).head(20)
src_stats["pct"] = src_stats["total_tokens"] / df["num-tokens"].sum() * 100
print(src_stats.to_string())

print("\n保存为本地jsonl...")
output_path = OFFICIAL_DIR / "official_data.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"已保存到 {output_path}")

stats = {
    "total_rows": len(df),
    "total_tokens": int(df["num-tokens"].sum()),
    "category_distribution": {
        cat: {"count": int(row["count"]), "tokens": int(row["total_tokens"]), "pct": round(row["pct"], 1)}
        for cat, row in cat_stats.iterrows()
    },
}
with open(OFFICIAL_DIR / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("统计信息已保存到 stats.json")
