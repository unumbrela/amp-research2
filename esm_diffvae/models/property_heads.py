"""Property prediction heads for latent space regularization."""

import torch
import torch.nn as nn


class PropertyHeads(nn.Module):
    """Predict multiple AMP properties from the latent vector z.

    Properties predicted:
    - is_amp: binary (antimicrobial or not)
    - mic_value: continuous (minimum inhibitory concentration, log-scale)
    - is_toxic: binary (cytotoxic or not)
    - is_hemolytic: binary (hemolytic or not)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        def make_head(out_dim: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, out_dim),
            )

        self.amp_head = make_head(1)
        self.mic_head = make_head(1)
        self.tox_head = make_head(1)
        self.hemo_head = make_head(1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict properties from latent vector.

        Args:
            z: [B, latent_dim]

        Returns:
            Dictionary with predicted property logits/values.
        """
        return {
            "is_amp": self.amp_head(z).squeeze(-1),        # [B]
            "mic_value": self.mic_head(z).squeeze(-1),      # [B]
            "is_toxic": self.tox_head(z).squeeze(-1),       # [B]
            "is_hemolytic": self.hemo_head(z).squeeze(-1),  # [B]
        }
