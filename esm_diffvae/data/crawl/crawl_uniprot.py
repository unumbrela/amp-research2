"""爬取 UniProt 数据库的 AMP 正样本和非 AMP 负样本。

使用 UniProt REST API 分页下载 TSV 格式数据。
- AMP 正样本: keyword:KW-0929 (Antimicrobial) AND length:[5 TO 50]
- 非 AMP 负样本: reviewed:true AND length:[5 TO 50] AND NOT keyword:KW-0929
"""

import io
import random
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500


def _is_valid_sequence(seq: str) -> bool:
    return bool(seq) and all(c in STANDARD_AA for c in seq) and 5 <= len(seq) <= 50


def fetch_uniprot(query: str, fields: str, max_results: int = 0) -> pd.DataFrame:
    """分页下载 UniProt 搜索结果。"""
    all_rows = []
    params = {
        "query": query,
        "format": "tsv",
        "fields": fields,
        "size": PAGE_SIZE,
    }

    url = BASE_URL
    page = 0

    while url:
        page += 1
        print(f"  下载第 {page} 页...")

        if page == 1:
            resp = requests.get(url, params=params, timeout=60)
        else:
            resp = requests.get(url, timeout=60)

        resp.raise_for_status()

        # 解析 TSV
        tsv_text = resp.text
        if tsv_text.strip():
            df_page = pd.read_csv(io.StringIO(tsv_text), sep="\t")
            all_rows.append(df_page)
            print(f"  第 {page} 页: {len(df_page)} 条")

        # 检查下一页
        link_header = resp.headers.get("Link", "")
        url = None
        if 'rel="next"' in link_header:
            # 格式: <url>; rel="next"
            url = link_header.split(">")[0].lstrip("<")

        if max_results > 0 and sum(len(d) for d in all_rows) >= max_results:
            break

        time.sleep(1)  # 速率限制

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def crawl_amp_positives() -> pd.DataFrame:
    """下载 AMP 正样本。"""
    print("[UniProt] 正在下载 AMP 正样本 (keyword:KW-0929)...")
    query = "(keyword:KW-0929) AND (length:[5 TO 50])"
    fields = "accession,sequence,length,keyword,protein_name,organism_name"

    df = fetch_uniprot(query, fields)
    if df.empty:
        print("[UniProt] 未获取到 AMP 数据")
        return pd.DataFrame()

    print(f"[UniProt] 原始 AMP 记录数: {len(df)}")

    # 找序列列
    seq_col = None
    for col in df.columns:
        if "sequence" in col.lower():
            seq_col = col
            break
    if seq_col is None:
        print(f"[UniProt] 错误: 找不到序列列。列名: {list(df.columns)}")
        return pd.DataFrame()

    df["sequence"] = df[seq_col].str.upper().str.strip()
    df = df[df["sequence"].apply(_is_valid_sequence)].copy()
    print(f"[UniProt] 有效 AMP 序列数: {len(df)}")

    result = pd.DataFrame({
        "sequence": df["sequence"].values,
        "is_amp": 1,
        "source": "uniprot_amp",
        "mic_value": float("nan"),
        "is_toxic": float("nan"),
        "is_hemolytic": float("nan"),
    })

    return result


def crawl_nonamp_negatives(target_count: int = 3750) -> pd.DataFrame:
    """下载非 AMP 负样本。"""
    print(f"[UniProt] 正在下载非 AMP 负样本 (目标: {target_count} 条)...")
    query = "(reviewed:true) AND (length:[5 TO 50]) AND NOT (keyword:KW-0929)"
    fields = "accession,sequence,length,protein_name"

    df = fetch_uniprot(query, fields, max_results=target_count + 1000)
    if df.empty:
        print("[UniProt] 未获取到非 AMP 数据")
        return pd.DataFrame()

    print(f"[UniProt] 原始非 AMP 记录数: {len(df)}")

    seq_col = None
    for col in df.columns:
        if "sequence" in col.lower():
            seq_col = col
            break
    if seq_col is None:
        print(f"[UniProt] 错误: 找不到序列列。列名: {list(df.columns)}")
        return pd.DataFrame()

    df["sequence"] = df[seq_col].str.upper().str.strip()
    df = df[df["sequence"].apply(_is_valid_sequence)].copy()
    print(f"[UniProt] 有效非 AMP 序列数: {len(df)}")

    # 随机采样到目标数量
    if len(df) > target_count:
        random.seed(42)
        df = df.sample(n=target_count, random_state=42)
        print(f"[UniProt] 随机采样到 {target_count} 条")

    result = pd.DataFrame({
        "sequence": df["sequence"].values,
        "is_amp": 0,
        "source": "uniprot_nonamp",
        "mic_value": float("nan"),
        "is_toxic": float("nan"),
        "is_hemolytic": float("nan"),
    })

    return result


def main():
    # AMP 正样本
    try:
        amp_df = crawl_amp_positives()
        if len(amp_df) > 0:
            out_path = RAW_DIR / "uniprot_amp.csv"
            amp_df.to_csv(out_path, index=False)
            print(f"[UniProt] 已保存 AMP: {out_path} ({len(amp_df)} 条)")
    except Exception as e:
        print(f"[UniProt] 错误: AMP 下载失败: {e}")
        amp_df = pd.DataFrame()

    # 非 AMP 负样本
    try:
        nonamp_df = crawl_nonamp_negatives()
        if len(nonamp_df) > 0:
            out_path = RAW_DIR / "uniprot_nonamp.csv"
            nonamp_df.to_csv(out_path, index=False)
            print(f"[UniProt] 已保存非 AMP: {out_path} ({len(nonamp_df)} 条)")
    except Exception as e:
        print(f"[UniProt] 错误: 非 AMP 下载失败: {e}")
        nonamp_df = pd.DataFrame()

    print(f"\n[UniProt] === 总结 ===")
    print(f"  AMP 正样本: {len(amp_df) if not amp_df.empty else 0}")
    print(f"  非 AMP 负样本: {len(nonamp_df) if not nonamp_df.empty else 0}")


if __name__ == "__main__":
    main()
