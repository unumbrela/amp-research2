"""VAE Decoder: maps latent z + property conditions to amino acid sequence.

v5: Autoregressive GRU decoder with z-injection at every timestep.
- z+properties concatenated to token embedding at EVERY step (not just h0)
- Word dropout: randomly replace input tokens with BOS to force z-reliance
- Teacher forcing with scheduled sampling
- ~2.0M params
"""

import torch
import torch.nn as nn


class AMPDecoder(nn.Module):
    """Autoregressive GRU decoder with per-step z-injection.

    Key improvement over v4: z+properties are concatenated to every timestep's
    input, preventing the GRU from "forgetting" the latent conditioning for
    longer sequences. Also conditions through initial hidden state for double
    coverage.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        prop_dim: int = 5,
        hidden_dim: int = 384,
        max_len: int = 50,
        vocab_size: int = 21,
        n_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        word_dropout: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.prop_dim = prop_dim
        self.bos_idx = vocab_size  # BOS token = index 21
        self.word_dropout = word_dropout

        # Token embedding: vocab_size + 1 for BOS token
        self.token_embedding = nn.Embedding(vocab_size + 1, embedding_dim)

        # Condition → GRU initial hidden state (still useful for h0)
        cond_dim = latent_dim + prop_dim
        self.cond_to_hidden = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Tanh(),
        )
        # Expand to n_layers hidden states
        self.hidden_expand = (
            nn.Linear(hidden_dim, hidden_dim * n_layers)
            if n_layers > 1
            else None
        )

        # GRU input = token_embedding + z + properties (per-step injection)
        gru_input_dim = embedding_dim + latent_dim + prop_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

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

    def _init_hidden(self, z: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
        """Compute GRU initial hidden state from z + properties.

        Returns:
            h_0: [n_layers, B, hidden_dim]
        """
        cond = torch.cat([z, properties], dim=-1)  # [B, cond_dim]
        h = self.cond_to_hidden(cond)               # [B, hidden_dim]

        if self.n_layers > 1 and self.hidden_expand is not None:
            h = self.hidden_expand(h)                           # [B, hidden_dim * n_layers]
            h = h.view(-1, self.n_layers, self.hidden_dim)      # [B, n_layers, hidden_dim]
            h = h.permute(1, 0, 2).contiguous()                 # [n_layers, B, hidden_dim]
        else:
            h = h.unsqueeze(0)  # [1, B, hidden_dim]

        return h

    def _apply_word_dropout(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Replace tokens with BOS index with probability word_dropout.

        This forces the decoder to rely on z rather than just copying
        teacher-forced inputs.
        """
        if not self.training or self.word_dropout <= 0:
            return token_indices
        mask = torch.rand_like(token_indices.float()) < self.word_dropout
        return token_indices.masked_fill(mask, self.bos_idx)

    def forward(
        self,
        z: torch.Tensor,
        properties: torch.Tensor,
        target_indices: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
        target_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to sequence logits (autoregressive).

        Args:
            z: [B, latent_dim] latent vector.
            properties: [B, prop_dim] property condition vector.
            target_indices: [B, L] ground truth token indices (for teacher forcing).
            teacher_forcing_ratio: probability of using ground truth at each step.
                1.0 = full teacher forcing, 0.0 = free running (generation).
            target_len: override output sequence length.

        Returns:
            logits: [B, L, vocab_size] amino acid logits.
            length_logits: [B, max_len] predicted length distribution.
        """
        B = z.size(0)
        device = z.device
        seq_len = target_len if target_len is not None else self.max_len

        # Initialize hidden state from condition
        h = self._init_hidden(z, properties)  # [n_layers, B, hidden_dim]

        # Length prediction
        length_logits = self.length_predictor(z)  # [B, max_len]

        # Condition vector to concatenate at every timestep
        cond = torch.cat([z, properties], dim=-1)  # [B, cond_dim]

        # Fast path: full teacher forcing — run entire sequence at once
        if teacher_forcing_ratio >= 1.0 and target_indices is not None:
            bos = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
            # Shift right: [BOS, t0, t1, ..., t_{L-2}]
            decoder_input = torch.cat([bos, target_indices[:, :seq_len - 1]], dim=1)
            # Apply word dropout
            decoder_input = self._apply_word_dropout(decoder_input)
            emb = self.token_embedding(decoder_input)  # [B, seq_len, embedding_dim]
            # Concatenate z+properties at every timestep
            cond_expanded = cond.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, cond_dim]
            gru_input = torch.cat([emb, cond_expanded], dim=-1)  # [B, seq_len, emb+cond]
            output, _ = self.gru(gru_input, h)              # [B, seq_len, hidden_dim]
            logits = self.output_proj(output)                # [B, seq_len, vocab_size]
            return logits, length_logits

        # Slow path: step-by-step with scheduled sampling
        input_token = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        all_logits = []
        cond_step = cond.unsqueeze(1)  # [B, 1, cond_dim]

        for t in range(seq_len):
            emb = self.token_embedding(input_token)  # [B, 1, embedding_dim]
            gru_input = torch.cat([emb, cond_step], dim=-1)  # [B, 1, emb+cond]
            output, h = self.gru(gru_input, h)             # [B, 1, hidden_dim]
            logit = self.output_proj(output)          # [B, 1, vocab_size]
            all_logits.append(logit)

            # Determine next input token
            if (
                target_indices is not None
                and t < target_indices.size(1)
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                # Teacher forcing: use ground truth
                input_token = target_indices[:, t : t + 1]
            else:
                # Free running: use own prediction
                input_token = logit.argmax(dim=-1)  # [B, 1]

        logits = torch.cat(all_logits, dim=1)  # [B, seq_len, vocab_size]
        return logits, length_logits
