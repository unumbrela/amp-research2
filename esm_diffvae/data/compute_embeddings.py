"""Pre-compute PLM embeddings for all sequences and cache to disk.

Supports multiple PLM backends: ESM-2, Ankh, ProtT5.
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
from models.plm_extractor import PLMExtractor, BACKEND_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def compute_and_save(
    csv_path: Path,
    output_path: Path,
    extractor: PLMExtractor,
    max_len: int = 50,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Compute PLM embeddings for all sequences in a CSV and save as .pt."""
    df = pd.read_csv(csv_path)
    sequences = df["sequence"].tolist()
    print(f"Computing embeddings for {len(sequences)} sequences from {csv_path.name}...")

    extractor = extractor.to(device)
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i: i + batch_size]
        with torch.cuda.amp.autocast():
            emb = extractor(batch_seqs, max_len=max_len)  # [B, max_len, D]
        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, max_len, D]
    torch.save(embeddings, output_path)
    print(f"Saved {embeddings.shape} to {output_path}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Pre-compute PLM embeddings")
    parser.add_argument("--backend", default="esm2",
                        choices=list(BACKEND_REGISTRY.keys()),
                        help="PLM backend: esm2, ankh, prot_t5")
    parser.add_argument("--model", default=None,
                        help="Model name within backend (default: first available)")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--processed-dir", default="processed",
                        help="Subdirectory under data/ containing train.csv/val.csv/test.csv")
    parser.add_argument("--embeddings-dir", default="embeddings",
                        help="Subdirectory under data/ to write embedding .pt files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Default model name if not specified
    model_name = args.model
    if model_name is None:
        model_name = list(BACKEND_REGISTRY[args.backend].keys())[0]

    print(f"=== PLM Embedding Computation ===")
    print(f"Backend: {args.backend}, Model: {model_name}")
    print(f"Embedding dim: {BACKEND_REGISTRY[args.backend][model_name]}")
    print(f"Device: {args.device}\n")

    extractor = PLMExtractor(backend=args.backend, model_name=model_name)

    embeddings_dir = DATA_DIR / args.embeddings_dir
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = DATA_DIR / args.processed_dir
    print(f"CSV source: {processed_dir}")
    print(f"Output embeddings: {embeddings_dir}")

    # Output filename includes backend for multi-PLM support
    # Keep "esm" suffix for backward compatibility when using esm2
    suffix = "esm" if args.backend == "esm2" else args.backend

    for split in ["train", "val", "test"]:
        csv_path = processed_dir / f"{split}.csv"
        if csv_path.exists():
            output_path = embeddings_dir / f"{split}_{suffix}.pt"
            compute_and_save(
                csv_path, output_path, extractor,
                max_len=args.max_len, batch_size=args.batch_size, device=args.device,
            )
        else:
            print(f"Skipping {split}: {csv_path} not found")

    print(f"\nDone! Embeddings cached in {embeddings_dir}")


if __name__ == "__main__":
    main()
