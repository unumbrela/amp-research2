"""VAE Encoder: maps sequence + ESM-2 embeddings to latent space.

v4: Bidirectional GRU replaces Transformer + CLS token.
- Better suited for short peptides (~20 AA)
- Naturally aggregates full sequence info via bidirectional hidden states
- ~940K params (down from ~2.3M Transformer encoder)
"""

import torch
import torch.nn as nn


class AMPEncoder(nn.Module):
    """Bidirectional GRU VAE encoder.

    Concatenates one-hot AA encoding with ESM-2 embeddings, projects to
    hidden dim, runs through a bidirectional GRU, and maps the final
    hidden states to latent distribution parameters (mu, logvar).
    """

    def __init__(
        self,
        esm_dim: int = 320,
        aa_dim: int = 21,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        input_dim = esm_dim + aa_dim  # 341

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Bidirectional: forward + backward hidden concatenated → 2 * hidden_dim
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(
        self,
        one_hot: torch.Tensor,
        esm_emb: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters.

        Args:
            one_hot: [B, L, 21] one-hot encoded sequence.
            esm_emb: [B, L, esm_dim] ESM-2 per-residue embeddings.
            padding_mask: [B, L] True for padded positions.

        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        x = torch.cat([one_hot, esm_emb], dim=-1)  # [B, L, input_dim]
        x = self.input_proj(x)                       # [B, L, hidden_dim]

        # Pack padded sequences for efficient GRU processing
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).cpu().clamp(min=1)
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

        _, h_n = self.gru(x)  # h_n: [n_layers * 2, B, hidden_dim]

        # Take last layer's forward and backward hidden states
        h_forward = h_n[-2]   # [B, hidden_dim]
        h_backward = h_n[-1]  # [B, hidden_dim]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # [B, hidden_dim * 2]

        return self.fc_mu(h_cat), self.fc_logvar(h_cat)
