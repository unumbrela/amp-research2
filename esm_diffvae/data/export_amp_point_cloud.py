#!/usr/bin/env python3
"""Export AMP sequence point-cloud CSV for frontend visualization.

Input:  data/processed/all_amp.csv
Output: frontend/client/public/amp-all-3d.csv

The output schema matches PointCloudHero requirements:
  id,part_type,x,y,z,name
Extra columns are preserved for richer tooltips:
  source
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "all_amp.csv"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "frontend" / "client" / "public" / "amp-all-3d.csv"

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ORDER)

HYDROPHOBIC = set("AVILMFWYC")
AROMATIC = set("FYW")
POLAR = set("STNQ")
BASIC = set("KRH")
ACIDIC = set("DE")
ALIPHATIC = set("AILV")
SMALL = set("AGSTC")
KMER_BINS = 128


def clean_sequence(seq: str) -> str:
    seq = (seq or "").strip().upper()
    # Keep standard amino acids only.
    return "".join(ch for ch in seq if ch in AA_SET)


def source_to_part_type(source: str) -> str:
    source = (source or "").strip().lower()
    if not source:
        return "unknown"

    tokens = [t for t in re.split(r"[|,;/\s]+", source) if t]
    groups: set[str] = set()

    for token in tokens:
        if "apd" in token:
            groups.add("apd")
        elif "dramp" in token:
            groups.add("dramp")
        elif "ampainter" in token:
            groups.add("ampainter")
        elif "diffamp" in token:
            groups.add("diffamp")
        elif "uniprot" in token:
            groups.add("uniprot")

    if not groups:
        return "other"
    if len(groups) == 1:
        return next(iter(groups))
    return "mixed"


def ratio(seq: str, letters: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(1 for ch in seq if ch in letters) / len(seq)


def net_charge_density(seq: str) -> float:
    if not seq:
        return 0.0
    pos = sum(1 for ch in seq if ch in BASIC)
    neg = sum(1 for ch in seq if ch in ACIDIC)
    return (pos - neg) / len(seq)


def sequence_features(seq: str) -> np.ndarray:
    n = len(seq)
    counts = Counter(seq)
    aa_composition = [counts[aa] / n if n else 0.0 for aa in AA_ORDER]

    features = [
        float(n),
        float(n) / 50.0,
        net_charge_density(seq),
        ratio(seq, HYDROPHOBIC),
        ratio(seq, AROMATIC),
        ratio(seq, POLAR),
        ratio(seq, BASIC),
        ratio(seq, ACIDIC),
        ratio(seq, ALIPHATIC),
        ratio(seq, SMALL),
    ]
    features.extend(aa_composition)
    features.extend(kmer_hash_features(seq, bins=KMER_BINS))
    return np.asarray(features, dtype=np.float64)


def kmer_hash_features(seq: str, bins: int = KMER_BINS) -> list[float]:
    if len(seq) < 2:
        return [0.0] * bins

    vec = np.zeros(bins, dtype=np.float64)
    for i in range(len(seq) - 1):
        a = ord(seq[i]) - 64
        b = ord(seq[i + 1]) - 64
        # Stable hash for AA bigrams.
        idx = (a * 37 + b * 101 + i * 7) % bins
        vec[idx] += 1.0

    vec /= max(1.0, vec.sum())
    return vec.tolist()


def pca_3d(features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    normalized = (features - mean) / std
    centered = normalized - normalized.mean(axis=0, keepdims=True)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = min(3, vt.shape[0])
    coords = centered @ vt[:components].T

    if components < 3:
        pad = np.zeros((coords.shape[0], 3 - components), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)

    return coords


def robust_rescale(
    coords: np.ndarray,
    target_span: float = 8.0,
    padding: float = 0.82,
    quantile: float = 99.5,
) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords

    # Use soft compression instead of hard clipping to avoid points sticking on planes.
    center = np.median(coords, axis=0, keepdims=True)
    centered = coords - center

    scale = np.percentile(np.abs(centered), quantile, axis=0, keepdims=True)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale < 1e-9] = 1.0

    normalized = np.tanh(centered / (scale * 1.15))
    half_span = target_span * 0.5 * max(0.1, min(1.0, padding))
    return normalized * half_span


def standardize(features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def kmeans(features: np.ndarray, n_clusters: int = 8, max_iter: int = 40, seed: int = 42) -> np.ndarray:
    n, _ = features.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    n_clusters = max(2, min(n_clusters, n))

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centers = features[init_idx].copy()
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(max_iter):
        dists = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if not np.any(mask):
                centers[k] = features[rng.integers(0, n)]
            else:
                centers[k] = features[mask].mean(axis=0)

    return labels


def fibonacci_sphere_points(k: int, radius: float) -> np.ndarray:
    if k <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if k == 1:
        return np.array([[0.0, 0.0, radius]], dtype=np.float64)

    points = np.zeros((k, 3), dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(k):
        y = 1.0 - (2.0 * i) / (k - 1)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points[i] = np.array([x, y, z]) * radius
    return points


def clustered_layout(local_coords: np.ndarray, labels: np.ndarray, radius: float = 8.0, local_scale: float = 0.42) -> np.ndarray:
    if local_coords.shape[0] == 0:
        return local_coords

    unique = sorted(set(int(v) for v in labels.tolist()))
    sizes = {k: int(np.sum(labels == k)) for k in unique}
    ordered = sorted(unique, key=lambda k: (-sizes[k], k))
    anchors = fibonacci_sphere_points(len(ordered), radius=radius)
    anchor_map = {cluster_id: anchors[i] for i, cluster_id in enumerate(ordered)}

    out = np.zeros_like(local_coords, dtype=np.float64)
    for cluster_id in unique:
        mask = labels == cluster_id
        cluster_points = local_coords[mask]
        center = cluster_points.mean(axis=0, keepdims=True)
        centered = cluster_points - center
        out[mask] = anchor_map[cluster_id] + centered * local_scale
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AMP point-cloud CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to all_amp.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output point-cloud CSV path")
    parser.add_argument("--id-prefix", default="AMP", help="ID prefix in output CSV")
    parser.add_argument("--clusters", type=int, default=8, help="Number of sequence-similarity clusters")
    parser.add_argument("--cluster-radius", type=float, default=2.6, help="Distance between cluster anchors")
    parser.add_argument("--cluster-spread", type=float, default=0.2, help="Intra-cluster spread scale")
    parser.add_argument("--span", type=float, default=8.0, help="Final coordinate span after robust scaling")
    parser.add_argument("--padding", type=float, default=0.82, help="Keep margin from cloud boundary [0-1]")
    parser.add_argument("--quantile", type=float, default=99.5, help="Robust scaling quantile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    features: list[np.ndarray] = []

    with args.input.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = clean_sequence(row.get("sequence", ""))
            if not seq:
                continue

            rows.append(
                {
                    "source": source_to_part_type(row.get("source", "")),
                    "name": seq,
                }
            )
            features.append(sequence_features(seq))

    feature_matrix = np.vstack(features) if features else np.zeros((0, 30 + KMER_BINS), dtype=np.float64)
    standardized = standardize(feature_matrix)
    labels = kmeans(standardized, n_clusters=args.clusters, max_iter=50, seed=42)
    local_coords = pca_3d(standardized)
    clustered_coords = clustered_layout(
        local_coords,
        labels,
        radius=args.cluster_radius,
        local_scale=args.cluster_spread,
    )
    coords = robust_rescale(
        clustered_coords,
        target_span=args.span,
        padding=args.padding,
        quantile=args.quantile,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "part_type", "source", "x", "y", "z", "name"])
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            cluster = int(labels[i - 1]) + 1 if labels.size else 1
            writer.writerow(
                {
                    "id": f"{args.id_prefix}_{i:06d}",
                    "part_type": f"cluster_{cluster:02d}",
                    "source": row["source"],
                    "x": f"{coords[i - 1, 0]:.6f}",
                    "y": f"{coords[i - 1, 1]:.6f}",
                    "z": f"{coords[i - 1, 2]:.6f}",
                    "name": row["name"],
                }
            )

    cluster_counts = Counter(f"cluster_{int(v)+1:02d}" for v in labels.tolist())
    source_counts = Counter(row["source"] for row in rows)
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows  : {len(rows)}")
    print(
        "Layout: "
        f"clusters={args.clusters}, radius={args.cluster_radius}, spread={args.cluster_spread}, "
        f"span={args.span}, padding={args.padding}, q={args.quantile}"
    )
    print("Clusters:")
    for part, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {part:<10} {count}")
    print("Sources:")
    for part, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {part:<10} {count}")


if __name__ == "__main__":
    main()
