"""爬取 DRAMP 数据库的 AMP 序列数据。

下载 general_amps.txt 和 synthetic_amps.txt，解析序列和属性注释。
从自由文本中提取 MIC 值、溶血性和细胞毒性标签。
"""

import math
import re
import sys
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DRAMP_URLS = {
    "general": "http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.txt",
    "synthetic": "http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/synthetic_amps.txt",
}

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# --- MIC 提取 ---
MIC_PATTERN = re.compile(
    r'(?:MIC|IC50|EC50)\s*[=:<>≤≥~]\s*([\d.]+)\s*'
    r'(μg/ml|µg/ml|ug/ml|μg/mL|µg/mL|ug/mL|µM|μM|uM|mM|mg/ml|mg/mL)',
    re.IGNORECASE,
)


def _parse_mic_values(text: str, seq_len: int) -> float | None:
    """从 Target_Organism 文本中提取 MIC 值，返回 ug/ml 的几何平均。"""
    if not isinstance(text, str) or not text.strip():
        return None

    values_ugml = []
    for match in MIC_PATTERN.finditer(text):
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        unit = match.group(2).lower().replace("μ", "u").replace("µ", "u")

        if val <= 0:
            continue

        if "ug/ml" in unit:
            values_ugml.append(val)
        elif "um" in unit and "mm" not in unit:
            # uM -> ug/ml: MW ≈ seq_len * 110 Da
            mw = seq_len * 110
            values_ugml.append(val * mw / 1e6)
        elif "mm" in unit:
            mw = seq_len * 110
            values_ugml.append(val * 1000 * mw / 1e6)
        elif "mg/ml" in unit:
            values_ugml.append(val * 1000)

    if not values_ugml:
        return None

    # 过滤极端值
    values_ugml = [v for v in values_ugml if 0.001 <= v <= 10000]
    if not values_ugml:
        return None

    # 几何平均
    log_mean = sum(math.log10(v) for v in values_ugml) / len(values_ugml)
    return log_mean  # 返回 log10(ug/ml)


# --- 溶血性分类 ---
HEMOLYTIC_POS = re.compile(
    r'hemoly|HC50\s*[<≤=]\s*\d{1,2}[.\d]*\s*(μ|µ|u)|'
    r'>\s*(10|15|20|25|30|40|50)\s*%\s*hemolysis|'
    r'hemolytic\s+at|strong\s+hemolytic',
    re.IGNORECASE,
)
HEMOLYTIC_NEG = re.compile(
    r'no\s+hemoly|non.hemoly|not\s+hemoly|negligible\s+hemoly|'
    r'<\s*[0-5]\s*%\s*hemolysis|'
    r'HC50\s*[>≥]\s*(100|200|300|500|1000)|'
    r'no\s+significant\s+hemoly|low\s+hemoly',
    re.IGNORECASE,
)


def _classify_hemolytic(text: str) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    neg = bool(HEMOLYTIC_NEG.search(text))
    pos = bool(HEMOLYTIC_POS.search(text)) and not neg
    if neg:
        return 0.0
    if pos:
        return 1.0
    return None


# --- 细胞毒性分类 ---
TOXIC_POS = re.compile(
    r'cytotoxic|toxic\s+to|IC50\s*[<≤=]\s*\d{1,2}[.\d]*\s*(μ|µ|u)|'
    r'significant\s+toxicity',
    re.IGNORECASE,
)
TOXIC_NEG = re.compile(
    r'no\s+cytotox|non.cytotox|not\s+cytotox|no\s+toxicity|'
    r'negligible\s+cytotox|low\s+cytotox|no\s+significant\s+cytotox|'
    r'IC50\s*[>≥]\s*(100|200|500)',
    re.IGNORECASE,
)


def _classify_toxic(text: str) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    neg = bool(TOXIC_NEG.search(text))
    pos = bool(TOXIC_POS.search(text)) and not neg
    if neg:
        return 0.0
    if pos:
        return 1.0
    return None


def _is_valid_sequence(seq: str) -> bool:
    return bool(seq) and all(c in STANDARD_AA for c in seq) and 5 <= len(seq) <= 50


def download_and_parse(name: str, url: str) -> pd.DataFrame:
    """下载并解析一个 DRAMP 文件。"""
    print(f"[DRAMP] 正在下载 {name}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # 保存原始文件
    raw_path = RAW_DIR / f"dramp_{name}_raw.txt"
    raw_path.write_bytes(resp.content)
    print(f"[DRAMP] 已保存原始文件: {raw_path} ({len(resp.content)} bytes)")

    # 解析 tab 分隔文件
    lines = resp.text.strip().split("\n")
    if not lines:
        print(f"[DRAMP] 警告: {name} 文件为空")
        return pd.DataFrame()

    header = lines[0].split("\t")
    print(f"[DRAMP] {name} 列名: {header}")

    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))
        rows.append(dict(zip(header, fields)))

    df = pd.DataFrame(rows)
    print(f"[DRAMP] {name} 原始行数: {len(df)}")

    # 找到序列列
    seq_col = None
    for col in df.columns:
        if "sequence" in col.lower() or "seq" in col.lower():
            seq_col = col
            break
    if seq_col is None:
        print(f"[DRAMP] 错误: 找不到序列列。列名: {list(df.columns)}")
        return pd.DataFrame()

    # 提取并清洗序列
    df["sequence"] = df[seq_col].str.upper().str.strip()
    df = df[df["sequence"].apply(_is_valid_sequence)].copy()
    print(f"[DRAMP] {name} 有效序列数 (标准AA, 长度5-50): {len(df)}")

    # 提取属性
    target_col = None
    for col in df.columns:
        if "target" in col.lower() and "organism" in col.lower():
            target_col = col
            break

    hemo_col = None
    for col in df.columns:
        if "hemoly" in col.lower():
            hemo_col = col
            break

    cyto_col = None
    for col in df.columns:
        if "cytotox" in col.lower() or "toxicity" in col.lower():
            cyto_col = col
            break

    # 解析 MIC
    if target_col:
        def _safe_parse_mic(r):
            try:
                return _parse_mic_values(r[target_col], len(r["sequence"]))
            except Exception:
                return None
        df["mic_value"] = df.apply(_safe_parse_mic, axis=1)
        mic_count = df["mic_value"].notna().sum()
        print(f"[DRAMP] {name} 成功提取 MIC 值: {mic_count}/{len(df)}")
    else:
        df["mic_value"] = float("nan")

    # 解析溶血性
    if hemo_col:
        def _safe_hemo(x):
            try:
                return _classify_hemolytic(x)
            except Exception:
                return None
        df["is_hemolytic"] = df[hemo_col].apply(_safe_hemo)
        hemo_count = df["is_hemolytic"].notna().sum()
        print(f"[DRAMP] {name} 成功提取溶血性标签: {hemo_count}/{len(df)}")
    else:
        df["is_hemolytic"] = float("nan")

    # 解析细胞毒性
    if cyto_col:
        def _safe_toxic(x):
            try:
                return _classify_toxic(x)
            except Exception:
                return None
        df["is_toxic"] = df[cyto_col].apply(_safe_toxic)
        toxic_count = df["is_toxic"].notna().sum()
        print(f"[DRAMP] {name} 成功提取毒性标签: {toxic_count}/{len(df)}")
    else:
        df["is_toxic"] = float("nan")

    # 构建输出
    result = pd.DataFrame({
        "sequence": df["sequence"].values,
        "is_amp": 1,
        "source": f"dramp_{name}",
        "mic_value": df["mic_value"].values,
        "is_toxic": df["is_toxic"].values,
        "is_hemolytic": df["is_hemolytic"].values,
    })

    return result


def main():
    all_dfs = []

    for name, url in DRAMP_URLS.items():
        try:
            df = download_and_parse(name, url)
            if len(df) > 0:
                out_path = RAW_DIR / f"dramp_{name}.csv"
                df.to_csv(out_path, index=False)
                print(f"[DRAMP] 已保存: {out_path} ({len(df)} 条)")
                all_dfs.append(df)
        except Exception as e:
            print(f"[DRAMP] 错误: 下载/解析 {name} 失败: {e}")

    if all_dfs:
        total = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[DRAMP] === 总结 ===")
        print(f"  总序列数: {len(total)}")
        print(f"  唯一序列数: {total['sequence'].nunique()}")
        print(f"  有 MIC 值: {total['mic_value'].notna().sum()}")
        print(f"  有溶血标签: {total['is_hemolytic'].notna().sum()}")
        print(f"  有毒性标签: {total['is_toxic'].notna().sum()}")
    else:
        print("[DRAMP] 未获取到任何数据")


if __name__ == "__main__":
    main()
