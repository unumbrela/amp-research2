"""解析本地数据源: Diff-AMP 和 AMPainter。

从 references/ 目录中读取已有数据集，转换为统一格式。
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _is_valid_sequence(seq: str) -> bool:
    return bool(seq) and all(c in STANDARD_AA for c in seq) and 5 <= len(seq) <= 50


def parse_diffamp() -> pd.DataFrame:
    """解析 Diff-AMP 数据集。"""
    print("[Diff-AMP] 正在解析本地数据...")
    data_dir = PROJECT_ROOT / "references" / "diff-amp" / "data"

    all_dfs = []

    # training_data.csv 和 val_data.csv: Seq, Label
    for fname in ["training_data.csv", "val_data.csv"]:
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            seq_col = "Seq" if "Seq" in df.columns else "seq"
            label_col = "Label" if "Label" in df.columns else "label"
            df = df.rename(columns={seq_col: "sequence", label_col: "is_amp"})
            df["source"] = f"diffamp_{fname}"
            all_dfs.append(df[["sequence", "is_amp", "source"]])
            print(f"  {fname}: {len(df)} 条")

    # AMPdb_data.csv: Id, Sequence, Length (全部为 AMP)
    ampdb_path = data_dir / "AMPdb_data.csv"
    if ampdb_path.exists():
        df = pd.read_csv(ampdb_path)
        seq_col = None
        for col in df.columns:
            if "sequence" in col.lower() or "seq" in col.lower():
                seq_col = col
                break
        if seq_col:
            df = df.rename(columns={seq_col: "sequence"})
            df["is_amp"] = 1
            df["source"] = "diffamp_AMPdb"
            all_dfs.append(df[["sequence", "is_amp", "source"]])
            print(f"  AMPdb_data.csv: {len(df)} 条")

    if not all_dfs:
        print("[Diff-AMP] 未找到数据文件")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result["sequence"] = result["sequence"].str.upper().str.strip()
    result = result[result["sequence"].apply(_is_valid_sequence)].copy()

    result["mic_value"] = float("nan")
    result["is_toxic"] = float("nan")
    result["is_hemolytic"] = float("nan")

    print(f"[Diff-AMP] 有效序列数: {len(result)}")
    return result


def parse_ampainter() -> pd.DataFrame:
    """解析 AMPainter 数据集。

    格式: label_id\tsequence\tscore
    - label_id 以 pos_ 或 neg_ 开头
    - score 为 MIC 相关数值 (正样本) 或 '/' (负样本)
    """
    print("[AMPainter] 正在解析本地数据...")
    data_dir = PROJECT_ROOT / "references" / "AMPainter" / "data"
    all_path = data_dir / "all.txt"

    if not all_path.exists():
        print(f"[AMPainter] 文件不存在: {all_path}")
        return pd.DataFrame()

    rows = []
    with open(all_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            label_id = parts[0]
            sequence = parts[1].upper().strip()
            score = parts[2] if len(parts) > 2 else "/"

            is_amp = 1 if label_id.startswith("pos_") else 0

            mic_value = float("nan")
            if score != "/" and score.strip():
                try:
                    mic_value = float(score)
                    # AMPainter 的 score 看起来是 log10(MIC in ug/ml)
                    # 直接保留，后续 merge 时统一处理
                except ValueError:
                    pass

            rows.append({
                "sequence": sequence,
                "is_amp": is_amp,
                "source": "ampainter",
                "mic_value": mic_value,
                "is_toxic": float("nan"),
                "is_hemolytic": float("nan"),
            })

    result = pd.DataFrame(rows)
    result = result[result["sequence"].apply(_is_valid_sequence)].copy()

    n_amp = (result["is_amp"] == 1).sum()
    n_nonamp = (result["is_amp"] == 0).sum()
    n_mic = result["mic_value"].notna().sum()
    print(f"[AMPainter] 有效序列数: {len(result)} (AMP={n_amp}, 非AMP={n_nonamp})")
    print(f"[AMPainter] 有 MIC 值: {n_mic}")

    return result


def main():
    # Diff-AMP
    try:
        diffamp_df = parse_diffamp()
        if len(diffamp_df) > 0:
            out_path = RAW_DIR / "diffamp.csv"
            diffamp_df.to_csv(out_path, index=False)
            print(f"[Diff-AMP] 已保存: {out_path} ({len(diffamp_df)} 条)")
    except Exception as e:
        print(f"[Diff-AMP] 错误: {e}")

    # AMPainter
    try:
        ampainter_df = parse_ampainter()
        if len(ampainter_df) > 0:
            out_path = RAW_DIR / "ampainter.csv"
            ampainter_df.to_csv(out_path, index=False)
            print(f"[AMPainter] 已保存: {out_path} ({len(ampainter_df)} 条)")
    except Exception as e:
        print(f"[AMPainter] 错误: {e}")


if __name__ == "__main__":
    main()
