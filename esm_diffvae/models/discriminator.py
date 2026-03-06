"""Sequence Discriminator for RL fine-tuning (Phase 1B).

BiGRU-based discriminator that distinguishes real sequences from generated ones.
Intentionally small (~30K params) to avoid mode collapse.
"""

import torch
import torch.nn as nn


class SequenceDiscriminator(nn.Module):
    """BiGRU discriminator: classifies sequences as real or generated.

    Architecture: Embedding → BiGRU → FC → sigmoid
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score sequences (higher = more likely real).

        Args:
            sequences: [B, L] amino acid index sequences.
            padding_mask: [B, L] True for padded positions.

        Returns:
            scores: [B] real/fake scores (pre-sigmoid logits).
        """
        x = self.embedding(sequences)  # [B, L, embed_dim]

        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).cpu().clamp(min=1)
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False,
            )

        _, h_n = self.gru(x)  # h_n: [2, B, hidden_dim]
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, hidden_dim*2]
        return self.classifier(h_cat).squeeze(-1)  # [B]
