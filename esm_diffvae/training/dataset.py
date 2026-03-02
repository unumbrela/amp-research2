"""PyTorch Dataset for AMP sequences with ESM-2 embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Amino acid vocabulary: 20 standard + padding token
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
PAD_IDX = len(AA_VOCAB)  # index 20 = padding
VOCAB_SIZE = len(AA_VOCAB) + 1  # 21


def sequence_to_indices(seq: str, max_len: int = 50) -> torch.Tensor:
    """Convert amino acid string to index tensor with padding."""
    indices = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[:max_len]]
    # Pad to max_len
    indices += [PAD_IDX] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def indices_to_sequence(indices: torch.Tensor) -> str:
    """Convert index tensor back to amino acid string (stops at first pad)."""
    seq = []
    for idx in indices:
        idx = idx.item()
        if idx == PAD_IDX:
            break
        if 0 <= idx < len(AA_VOCAB):
            seq.append(AA_VOCAB[idx])
    return "".join(seq)


def sequence_to_one_hot(seq: str, max_len: int = 50, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Convert amino acid string to one-hot tensor [max_len, vocab_size]."""
    indices = sequence_to_indices(seq, max_len)
    return torch.nn.functional.one_hot(indices, num_classes=vocab_size).float()


class AMPDataset(Dataset):
    """Dataset for AMP sequences with pre-computed ESM-2 embeddings.

    Loads:
    - Sequences and labels from CSV
    - Pre-computed ESM-2 embeddings from .pt file (if available)
    """

    def __init__(
        self,
        csv_path: str | Path,
        esm_embedding_path: str | Path | None = None,
        max_len: int = 50,
    ):
        self.max_len = max_len
        self.df = pd.read_csv(csv_path)

        # Load ESM embeddings if available
        if esm_embedding_path and Path(esm_embedding_path).exists():
            self.esm_embeddings = torch.load(esm_embedding_path, weights_only=True)
            assert len(self.esm_embeddings) == len(self.df), \
                f"ESM embeddings count ({len(self.esm_embeddings)}) != CSV rows ({len(self.df)})"
        else:
            self.esm_embeddings = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).upper()

        # One-hot encoding
        one_hot = sequence_to_one_hot(seq, self.max_len)  # [max_len, vocab_size]

        # Target indices
        target_indices = sequence_to_indices(seq, self.max_len)  # [max_len]

        # Padding mask (True = padded position)
        seq_len = min(len(seq), self.max_len)
        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        padding_mask[seq_len:] = True

        # Properties vector
        properties = torch.tensor([
            float(row.get("is_amp", 0)),
            float(row["mic_value"]) if pd.notna(row.get("mic_value")) else 0.0,
            float(row["is_toxic"]) if pd.notna(row.get("is_toxic")) else 0.0,
            float(row["is_hemolytic"]) if pd.notna(row.get("is_hemolytic")) else 0.0,
            float(row.get("length_norm", seq_len / 50.0)),
        ], dtype=torch.float32)

        # Property mask (True = property is known/available)
        prop_mask = torch.tensor([
            True,  # is_amp always available
            pd.notna(row.get("mic_value")),
            pd.notna(row.get("is_toxic")),
            pd.notna(row.get("is_hemolytic")),
            True,  # length always available
        ], dtype=torch.bool)

        # ESM-2 embeddings
        if self.esm_embeddings is not None:
            esm_emb = self.esm_embeddings[idx]  # [max_len, esm_dim]
        else:
            esm_emb = torch.zeros(self.max_len, 320)  # placeholder

        return {
            "sequence": seq,
            "one_hot": one_hot,
            "target_indices": target_indices,
            "padding_mask": padding_mask,
            "properties": properties,
            "prop_mask": prop_mask,
            "esm_emb": esm_emb,
            "seq_len": seq_len,
        }


def create_dataloader(
    csv_path: str | Path,
    esm_embedding_path: str | Path | None = None,
    max_len: int = 50,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for AMP data."""
    dataset = AMPDataset(csv_path, esm_embedding_path, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop last incomplete batch for training
    )
