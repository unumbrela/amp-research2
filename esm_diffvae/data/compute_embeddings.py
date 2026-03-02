"""Pre-compute ESM-2 embeddings for all sequences and cache to disk.

Run this once after prepare_data.py. Embeddings are cached as .pt files
to avoid recomputing during training.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_extractor import ESMFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def compute_and_save(
    csv_path: Path,
    output_path: Path,
    extractor: ESMFeatureExtractor,
    max_len: int = 50,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Compute ESM-2 embeddings for all sequences in a CSV and save as .pt."""
    df = pd.read_csv(csv_path)
    sequences = df["sequence"].tolist()
    print(f"Computing embeddings for {len(sequences)} sequences from {csv_path.name}...")

    extractor = extractor.to(device)
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i : i + batch_size]
        with torch.cuda.amp.autocast():
            emb = extractor(batch_seqs, max_len=max_len)  # [B, max_len, D]
        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, max_len, D]
    torch.save(embeddings, output_path)
    print(f"Saved {embeddings.shape} to {output_path}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ESM-2 embeddings")
    parser.add_argument("--model", default="esm2_t6_8M_UR50D", help="ESM-2 model name")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"=== ESM-2 Embedding Computation ===")
    print(f"Model: {args.model}, Device: {args.device}\n")

    extractor = ESMFeatureExtractor(args.model)

    embeddings_dir = DATA_DIR / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = DATA_DIR / "processed"

    for split in ["train", "val", "test"]:
        csv_path = processed_dir / f"{split}.csv"
        if csv_path.exists():
            output_path = embeddings_dir / f"{split}_esm.pt"
            compute_and_save(
                csv_path, output_path, extractor,
                max_len=args.max_len, batch_size=args.batch_size, device=args.device,
            )
        else:
            print(f"Skipping {split}: {csv_path} not found")

    print("\nDone! Embeddings cached in", embeddings_dir)


if __name__ == "__main__":
    main()
