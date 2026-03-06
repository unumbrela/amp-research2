"""合并所有数据源，去重，清洗，划分训练/验证/测试集。

加载 data/raw/ 下的所有 CSV 文件，统一格式后合并去重。
对于重复序列，属性取多源合并:
- mic_value: 取均值
- is_hemolytic / is_toxic: 多数投票
- source: 拼接所有来源
最终按 60% AMP / 40% non-AMP 平衡，分层划分。
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
TARGET_AMP_RATIO = 0.6  # 60% AMP


def load_all_raw() -> pd.DataFrame:
    """加载 raw/ 目录下所有 CSV 文件。"""
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {RAW_DIR} 中未找到 CSV 文件")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"  加载 {f.name}: {len(df)} 条")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"合并后总数: {len(combined)}")
    return combined


def clean_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """清洗序列: 大写、过滤非标准 AA、过滤长度。"""
    df = df.copy()
    df["sequence"] = df["sequence"].str.upper().str.strip()

    # 过滤
    valid = df["sequence"].apply(
        lambda s: isinstance(s, str) and len(s) >= 5 and len(s) <= 50
        and all(c in STANDARD_AA for c in s)
    )
    n_removed = (~valid).sum()
    df = df[valid].copy()
    print(f"序列清洗: 移除 {n_removed} 条无效序列，保留 {len(df)} 条")
    return df


def merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """按序列去重，合并属性。"""
    print(f"去重前: {len(df)} 条，唯一序列: {df['sequence'].nunique()}")

    def _merge_group(group):
        row = {
            "sequence": group["sequence"].iloc[0],
            # AMP 标签: 任一来源标记为 AMP 则为 AMP
            "is_amp": int(group["is_amp"].max()),
            # 来源: 拼接
            "source": "|".join(sorted(group["source"].unique())),
        }

        # MIC: 取均值 (已是 log10 scale)
        mic_vals = group["mic_value"].dropna()
        if len(mic_vals) > 0:
            row["mic_value"] = mic_vals.mean()
        else:
            row["mic_value"] = float("nan")

        # 溶血性: 多数投票
        hemo_vals = group["is_hemolytic"].dropna()
        if len(hemo_vals) > 0:
            row["is_hemolytic"] = float(round(hemo_vals.mean()))
        else:
            row["is_hemolytic"] = float("nan")

        # 毒性: 多数投票
        toxic_vals = group["is_toxic"].dropna()
        if len(toxic_vals) > 0:
            row["is_toxic"] = float(round(toxic_vals.mean()))
        else:
            row["is_toxic"] = float("nan")

        return pd.Series(row)

    merged = df.groupby("sequence", sort=False).apply(_merge_group).reset_index(drop=True)
    print(f"去重后: {len(merged)} 条唯一序列")
    return merged


def balance_dataset(df: pd.DataFrame, target_amp_ratio: float = TARGET_AMP_RATIO) -> pd.DataFrame:
    """平衡 AMP/非AMP 比例到目标比例。"""
    n_amp = (df["is_amp"] == 1).sum()
    n_nonamp = (df["is_amp"] == 0).sum()
    print(f"平衡前: AMP={n_amp}, 非AMP={n_nonamp}, AMP比例={n_amp/(n_amp+n_nonamp):.2%}")

    current_ratio = n_amp / (n_amp + n_nonamp)

    if abs(current_ratio - target_amp_ratio) < 0.02:
        print("比例已接近目标，无需调整")
        return df

    # 确定需要的数量
    if current_ratio > target_amp_ratio:
        # AMP 太多，下采样 AMP
        target_amp = int(n_nonamp * target_amp_ratio / (1 - target_amp_ratio))
        if target_amp < n_amp:
            amp_df = df[df["is_amp"] == 1].sample(n=target_amp, random_state=42)
            nonamp_df = df[df["is_amp"] == 0]
            df = pd.concat([amp_df, nonamp_df], ignore_index=True)
    else:
        # 非AMP 太多，下采样非AMP
        target_nonamp = int(n_amp * (1 - target_amp_ratio) / target_amp_ratio)
        if target_nonamp < n_nonamp:
            amp_df = df[df["is_amp"] == 1]
            nonamp_df = df[df["is_amp"] == 0].sample(n=target_nonamp, random_state=42)
            df = pd.concat([amp_df, nonamp_df], ignore_index=True)

    n_amp = (df["is_amp"] == 1).sum()
    n_nonamp = (df["is_amp"] == 0).sum()
    print(f"平衡后: AMP={n_amp}, 非AMP={n_nonamp}, AMP比例={n_amp/(n_amp+n_nonamp):.2%}")

    return df


def add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """添加 length 和 length_norm 列。"""
    df = df.copy()
    df["length"] = df["sequence"].str.len()
    df["length_norm"] = df["length"] / 50.0
    return df


def split_data(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, seed=42):
    """分层划分 train/val/test。"""
    train_df, temp_df = train_test_split(
        df, test_size=1 - train_ratio, stratify=df["is_amp"], random_state=seed
    )
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio_adjusted, stratify=temp_df["is_amp"], random_state=seed
    )
    return train_df, val_df, test_df


def compute_stats(df: pd.DataFrame, train_df, val_df, test_df) -> dict:
    """计算数据集统计信息。"""
    stats = {
        "total_sequences": len(df),
        "unique_sequences": df["sequence"].nunique(),
        "n_amp": int((df["is_amp"] == 1).sum()),
        "n_nonamp": int((df["is_amp"] == 0).sum()),
        "amp_ratio": float((df["is_amp"] == 1).mean()),
        "length_stats": {
            "min": int(df["length"].min()),
            "max": int(df["length"].max()),
            "mean": float(df["length"].mean()),
            "median": float(df["length"].median()),
        },
        "property_coverage": {
            "mic_value": int(df["mic_value"].notna().sum()),
            "is_toxic": int(df["is_toxic"].notna().sum()),
            "is_hemolytic": int(df["is_hemolytic"].notna().sum()),
        },
        "sources": df["source"].value_counts().to_dict(),
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    return stats


def main():
    print("=" * 60)
    print("AMP 数据集合并与清洗")
    print("=" * 60)

    # 1. 加载
    print("\n--- 步骤 1: 加载原始数据 ---")
    df = load_all_raw()

    # 2. 清洗
    print("\n--- 步骤 2: 序列清洗 ---")
    df = clean_sequences(df)

    # 3. 去重合并
    print("\n--- 步骤 3: 去重合并 ---")
    df = merge_duplicates(df)

    # 4. 平衡
    print("\n--- 步骤 4: 平衡数据集 (目标 60% AMP) ---")
    df = balance_dataset(df)

    # 5. 添加计算列
    print("\n--- 步骤 5: 添加计算列 ---")
    df = add_computed_columns(df)

    # 6. 划分
    print("\n--- 步骤 6: 分层划分 (80/10/10) ---")
    train_df, val_df, test_df = split_data(df)
    print(f"  训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    # 7. 保存
    print("\n--- 步骤 7: 保存 ---")
    columns = ["sequence", "is_amp", "source", "mic_value", "is_toxic", "is_hemolytic", "length", "length_norm"]

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df), ("all", df)]:
        out_path = PROCESSED_DIR / f"{name}.csv"
        split_df[columns].to_csv(out_path, index=False)
        print(f"  已保存: {out_path} ({len(split_df)} 条)")

    # 8. 统计
    stats = compute_stats(df, train_df, val_df, test_df)
    stats_path = PROCESSED_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计信息: {stats_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)
    print(f"总序列数: {stats['total_sequences']}")
    print(f"AMP: {stats['n_amp']} ({stats['amp_ratio']:.1%})")
    print(f"非AMP: {stats['n_nonamp']} ({1-stats['amp_ratio']:.1%})")
    print(f"序列长度: {stats['length_stats']['min']}-{stats['length_stats']['max']} "
          f"(平均 {stats['length_stats']['mean']:.1f})")
    print(f"属性覆盖:")
    for prop, count in stats["property_coverage"].items():
        pct = count / stats["total_sequences"] * 100
        print(f"  {prop}: {count} ({pct:.1f}%)")
    print(f"划分: 训练={stats['splits']['train']}, "
          f"验证={stats['splits']['val']}, 测试={stats['splits']['test']}")


if __name__ == "__main__":
    main()
