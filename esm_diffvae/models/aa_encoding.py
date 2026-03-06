"""Amino acid encoding: BLOSUM62 + learnable embedding hybrid.

Replaces one-hot encoding (21-dim) with richer per-residue features:
- BLOSUM62 (20-dim): evolutionary substitution scores, captures AA similarity
- Learnable embedding (16-dim): task-specific features learned end-to-end

Total output: 36-dim per residue (configurable).
"""

import torch
import torch.nn as nn


# BLOSUM62 substitution matrix for 20 standard amino acids.
# Row/column order: ACDEFGHIKLMNPQRSTVWY (same as AA_VOCAB in dataset.py)
# Values from NCBI BLOSUM62, scaled to [-1, 1] range for neural network input.
_BLOSUM62_RAW = [
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2],  # A
    [ 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],  # C
    [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3],  # D
    [-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2, 0, 0,-1,-2,-3,-2],  # E
    [-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3],  # F
    [ 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3],  # G
    [-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2],  # H
    [-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1],  # I
    [-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2],  # K
    [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1],  # L
    [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1],  # M
    [-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2],  # N
    [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3],  # P
    [-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1],  # Q
    [-1,-3,-2, 0,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2],  # R
    [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2],  # S
    [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2],  # T
    [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1],  # V
    [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2],  # W
    [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7],  # Y
]


def _build_blosum62_table(scale: float = 1.0 / 4.0) -> torch.Tensor:
    """Build BLOSUM62 lookup table [vocab_size+1, 20].

    Rows 0-19 = standard amino acids (ACDEFGHIKLMNPQRSTVWY).
    Row 20 = padding token (all zeros).
    Scaled to reasonable range for neural network input.
    """
    mat = torch.tensor(_BLOSUM62_RAW, dtype=torch.float32) * scale
    # Add padding row (all zeros)
    pad_row = torch.zeros(1, 20, dtype=torch.float32)
    return torch.cat([mat, pad_row], dim=0)  # [21, 20]


class HybridAAEncoding(nn.Module):
    """Hybrid amino acid encoding: BLOSUM62 (fixed) + learnable embedding.

    Args:
        learned_dim: Dimension of the learnable embedding per AA. Default 16.
        vocab_size: Number of amino acid types (20 standard + 1 padding = 21).
        blosum_scale: Scale factor for BLOSUM62 values.
    """

    def __init__(self, learned_dim: int = 16, vocab_size: int = 21, blosum_scale: float = 0.25):
        super().__init__()
        self.blosum_dim = 20
        self.learned_dim = learned_dim
        self.output_dim = self.blosum_dim + learned_dim

        # Fixed BLOSUM62 features (not trainable)
        blosum_table = _build_blosum62_table(scale=blosum_scale)
        self.register_buffer("blosum_table", blosum_table)  # [vocab_size, 20]

        # Learnable embedding (trainable)
        # vocab_size + 1 to handle padding idx safely
        self.learned_embed = nn.Embedding(vocab_size + 1, learned_dim, padding_idx=vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode amino acid indices to hybrid features.

        Args:
            indices: [B, L] amino acid index tensor (0-19 = AAs, 20 = padding).

        Returns:
            [B, L, output_dim] hybrid encoding (BLOSUM62 + learned).
        """
        # Clamp indices to valid range for BLOSUM lookup
        safe_idx = indices.clamp(0, self.blosum_table.size(0) - 1)
        blosum_features = self.blosum_table[safe_idx]         # [B, L, 20]
        learned_features = self.learned_embed(safe_idx)       # [B, L, learned_dim]
        return torch.cat([blosum_features, learned_features], dim=-1)  # [B, L, output_dim]


class OneHotEncoding(nn.Module):
    """Classic one-hot encoding (backward compatibility).

    Wraps F.one_hot for consistent interface with HybridAAEncoding.
    """

    def __init__(self, vocab_size: int = 21):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = vocab_size

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode amino acid indices to one-hot vectors.

        Args:
            indices: [B, L] amino acid index tensor.

        Returns:
            [B, L, vocab_size] one-hot encoding.
        """
        return torch.nn.functional.one_hot(
            indices.clamp(0, self.vocab_size - 1), self.vocab_size
        ).float()
