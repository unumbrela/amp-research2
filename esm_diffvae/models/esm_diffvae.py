"""ESM-DiffVAE v5: Full model combining all components.

v5: BiGRU encoder + autoregressive GRU decoder with per-step z-injection.
"""

import torch
import torch.nn as nn

from .esm_extractor import ESMFeatureExtractor
from .encoder import AMPEncoder
from .decoder import AMPDecoder
from .latent_diffusion import LatentDiffusion
from .property_heads import PropertyHeads


class ESMDiffVAE(nn.Module):
    """ESM-enhanced Diffusion Variational Autoencoder for AMP generation.

    v4 architecture:
    - Encoder: Bidirectional GRU (replaces Transformer + CLS token)
    - Decoder: Autoregressive GRU with teacher forcing (replaces non-autoregressive Transformer)
    - Reduced parameter count (~3.4M vs ~14.5M) for small dataset (6.6K samples)
    """

    def __init__(self, config: dict):
        super().__init__()
        esm_cfg = config["esm"]
        vae_cfg = config["vae"]
        prop_cfg = config["properties"]
        diff_cfg = config["diffusion"]

        self.latent_dim = vae_cfg["latent_dim"]
        self.max_len = vae_cfg["max_seq_len"]
        self.vocab_size = vae_cfg["aa_vocab_size"]

        # ESM-2 Feature Extractor (frozen)
        self.esm = ESMFeatureExtractor(esm_cfg["model_name"])

        # VAE Encoder (v4: Bidirectional GRU)
        self.encoder = AMPEncoder(
            esm_dim=esm_cfg["embedding_dim"],
            aa_dim=vae_cfg["aa_vocab_size"],
            hidden_dim=vae_cfg["hidden_dim"],
            latent_dim=vae_cfg["latent_dim"],
            n_layers=vae_cfg["n_encoder_layers"],
            dropout=vae_cfg["dropout"],
        )

        # VAE Decoder (v5: Autoregressive GRU with per-step z-injection)
        self.decoder = AMPDecoder(
            latent_dim=vae_cfg["latent_dim"],
            prop_dim=prop_cfg["dim"],
            hidden_dim=vae_cfg.get("decoder_hidden_dim", vae_cfg["hidden_dim"]),
            max_len=vae_cfg["max_seq_len"],
            vocab_size=vae_cfg["aa_vocab_size"],
            n_layers=vae_cfg["n_decoder_layers"],
            embedding_dim=vae_cfg.get("decoder_embedding_dim", 64),
            dropout=vae_cfg["dropout"],
            word_dropout=vae_cfg.get("word_dropout", 0.0),
        )

        # Property Prediction Heads
        self.prop_heads = PropertyHeads(latent_dim=vae_cfg["latent_dim"])

        # Latent Diffusion
        self.diffusion = LatentDiffusion(
            latent_dim=vae_cfg["latent_dim"],
            prop_dim=prop_cfg["dim"],
            T=diff_cfg["T"],
            schedule=diff_cfg["schedule"],
            beta_start=diff_cfg["beta_start"],
            beta_end=diff_cfg["beta_end"],
            hidden_dim=diff_cfg["denoiser_hidden_dim"],
            cfg_drop_prob=diff_cfg["cfg_drop_prob"],
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(
        self,
        sequences: list[str],
        one_hot: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode sequences to latent distribution."""
        esm_emb = self.esm(sequences, max_len=one_hot.size(1))
        esm_emb = esm_emb.to(one_hot.device)
        mu, logvar = self.encoder(one_hot, esm_emb, padding_mask)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(
        self,
        z: torch.Tensor,
        properties: torch.Tensor,
        target_indices: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        target_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to sequence logits.

        Args:
            z: [B, latent_dim]
            properties: [B, prop_dim]
            target_indices: [B, L] ground truth tokens (for teacher forcing).
            teacher_forcing_ratio: 0.0=free running, 1.0=full teacher forcing.
            target_len: Override output length.
        """
        return self.decoder(
            z, properties, target_indices, teacher_forcing_ratio, target_len
        )

    def forward(
        self,
        sequences: list[str],
        one_hot: torch.Tensor,
        properties: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for VAE training."""
        z, mu, logvar = self.encode(sequences, one_hot, padding_mask)
        logits, length_logits = self.decode(
            z, properties, target_indices, teacher_forcing_ratio
        )
        prop_preds = self.prop_heads(z)

        return {
            "logits": logits,
            "length_logits": length_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prop_preds": prop_preds,
        }
