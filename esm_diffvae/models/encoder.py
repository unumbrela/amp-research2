"""VAE Encoder: maps sequence encoding + PLM embeddings to latent space.

v6: Bidirectional GRU encoder.
- Accepts any AA encoding (hybrid BLOSUM62+learned, or one-hot) + any PLM embeddings
- Better suited for short peptides (~20 AA)
- Naturally aggregates full sequence info via bidirectional hidden states
"""

import torch
import torch.nn as nn


class AMPEncoder(nn.Module):
    """Bidirectional GRU VAE encoder.

    Concatenates AA encoding (BLOSUM62+learned or one-hot) with PLM embeddings,
    projects to hidden dim, runs through a bidirectional GRU, and maps the final
    hidden states to latent distribution parameters (mu, logvar).
    """

    def __init__(
        self,
        esm_dim: int = 320,
        aa_dim: int = 36,  # 20 (BLOSUM62) + 16 (learned) for hybrid; 21 for one-hot
        hidden_dim: int = 256,
        latent_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        input_dim = esm_dim + aa_dim  # 341

        # Gradual compression: input_dim → 512 → hidden_dim (avoid 4× bottleneck)
        mid_dim = max(hidden_dim, min(512, input_dim))
        if mid_dim > hidden_dim and input_dim > hidden_dim * 2:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.LayerNorm(mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
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
        aa_features: torch.Tensor,
        plm_emb: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters.

        Args:
            aa_features: [B, L, aa_dim] AA encoding (BLOSUM62+learned or one-hot).
            plm_emb: [B, L, esm_dim] PLM per-residue embeddings.
            padding_mask: [B, L] True for padded positions.

        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        x = torch.cat([aa_features, plm_emb], dim=-1)  # [B, L, input_dim]
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
