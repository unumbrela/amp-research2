"""VAE Decoder: maps latent z + property conditions to amino acid sequence.

v8: Non-autoregressive Transformer decoder — predicts all positions in parallel.

Key advantages over Conv decoder (v7):
- Self-attention gives every position global context (not just ±2 neighbors)
- Smaller model (~350K params vs ~1.2M) better suited for 6.6K dataset
- Still non-autoregressive → no exposure bias
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with self-attention and FFN."""

    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class AMPDecoder(nn.Module):
    """Non-autoregressive Transformer decoder for short peptide generation.

    Predicts all amino acid positions simultaneously from the latent vector z
    and property conditions via self-attention, avoiding teacher forcing and
    error accumulation while capturing global inter-position dependencies.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        prop_dim: int = 5,
        hidden_dim: int = 128,
        max_len: int = 30,
        vocab_size: int = 21,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 256,
        embedding_dim: int = 64,  # kept for config compatibility
        dropout: float = 0.3,
        word_dropout: float = 0.0,  # kept for config compatibility (unused)
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.prop_dim = prop_dim
        self.bos_idx = vocab_size  # kept for compatibility

        cond_dim = latent_dim + prop_dim

        # Position embeddings: learnable per-position features
        self.position_embed = nn.Embedding(max_len, hidden_dim)

        # Condition projection: z + properties → per-position seed (2-layer MLP)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer self-attention blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm (needed for pre-norm architecture)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden → logits
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size),
        )

        # Length predictor (from z directly)
        self.length_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_len),
        )

    def forward(
        self,
        z: torch.Tensor,
        properties: torch.Tensor,
        target_indices: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
        target_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to sequence logits (non-autoregressive).

        Args:
            z: [B, latent_dim] latent vector.
            properties: [B, prop_dim] property condition vector.
            target_indices: Unused (kept for API compatibility).
            teacher_forcing_ratio: Unused (kept for API compatibility).
            target_len: Override output sequence length.

        Returns:
            logits: [B, L, vocab_size] amino acid logits.
            length_logits: [B, max_len] predicted length distribution.
        """
        B = z.size(0)
        device = z.device
        seq_len = target_len if target_len is not None else self.max_len

        # Length prediction
        length_logits = self.length_predictor(z)  # [B, max_len]

        # Condition vector → broadcast to all positions
        cond = torch.cat([z, properties], dim=-1)  # [B, cond_dim]
        h_cond = self.cond_proj(cond)  # [B, hidden_dim]
        h_cond = h_cond.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, hidden_dim]

        # Add position embeddings
        pos_ids = torch.arange(seq_len, device=device)
        pos_emb = self.position_embed(pos_ids)  # [seq_len, hidden_dim]
        h = h_cond + pos_emb.unsqueeze(0)  # [B, seq_len, hidden_dim]

        # Transformer self-attention: capture global AA dependencies
        for block in self.transformer_blocks:
            h = block(h)

        # Final norm + per-position logits
        h = self.final_norm(h)
        logits = self.output_proj(h)  # [B, seq_len, vocab_size]

        return logits, length_logits
