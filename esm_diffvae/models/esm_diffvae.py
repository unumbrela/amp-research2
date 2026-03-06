"""ESM-DiffVAE v8: Full model combining all components.

v8: Multi-PLM backend + BLOSUM62 hybrid encoding + BiGRU encoder +
    non-autoregressive Transformer decoder (global context, no exposure bias).
"""

import torch
import torch.nn as nn

from .plm_extractor import PLMExtractor
from .aa_encoding import HybridAAEncoding, OneHotEncoding
from .encoder import AMPEncoder
from .decoder import AMPDecoder
from .latent_diffusion import LatentDiffusion
from .property_heads import PropertyHeads


class ESMDiffVAE(nn.Module):
    """PLM-enhanced Diffusion Variational Autoencoder for AMP generation.

    v8 architecture:
    - PLM: Supports ESM-2 / Ankh / ProtT5 (configurable backend)
    - Encoding: BLOSUM62 + learnable embedding (replaces one-hot)
    - Encoder: Bidirectional GRU (with gradual input projection)
    - Decoder: Non-autoregressive Transformer decoder (global context, no exposure bias)
    """

    def __init__(self, config: dict):
        super().__init__()
        # Support both old "esm" config and new "plm" config
        plm_cfg = config.get("plm", config.get("esm", {}))
        enc_cfg = config.get("encoding", {})
        vae_cfg = config["vae"]
        prop_cfg = config["properties"]
        diff_cfg = config["diffusion"]

        self.latent_dim = vae_cfg["latent_dim"]
        self.max_len = vae_cfg["max_seq_len"]
        self.vocab_size = vae_cfg["aa_vocab_size"]

        # PLM Feature Extractor (frozen)
        backend = plm_cfg.get("backend", "esm2")
        model_name = plm_cfg.get("model_name", plm_cfg.get("model_name", "esm2_t6_8M_UR50D"))
        self.plm = PLMExtractor(backend=backend, model_name=model_name)
        plm_dim = self.plm.embedding_dim

        # Amino Acid Encoding
        encoding_type = enc_cfg.get("type", "hybrid")
        if encoding_type == "hybrid":
            learned_dim = enc_cfg.get("learned_embed_dim", 16)
            self.aa_encoding = HybridAAEncoding(
                learned_dim=learned_dim,
                vocab_size=vae_cfg["aa_vocab_size"],
            )
        else:
            self.aa_encoding = OneHotEncoding(vocab_size=vae_cfg["aa_vocab_size"])
        aa_dim = self.aa_encoding.output_dim

        # VAE Encoder (v6: Bidirectional GRU)
        self.encoder = AMPEncoder(
            esm_dim=plm_dim,
            aa_dim=aa_dim,
            hidden_dim=vae_cfg["hidden_dim"],
            latent_dim=vae_cfg["latent_dim"],
            n_layers=vae_cfg["n_encoder_layers"],
            dropout=vae_cfg["dropout"],
        )

        # VAE Decoder (v8: Non-autoregressive Transformer decoder)
        self.decoder = AMPDecoder(
            latent_dim=vae_cfg["latent_dim"],
            prop_dim=prop_cfg["dim"],
            hidden_dim=vae_cfg.get("decoder_hidden_dim", vae_cfg["hidden_dim"]),
            max_len=vae_cfg["max_seq_len"],
            vocab_size=vae_cfg["aa_vocab_size"],
            n_layers=vae_cfg["n_decoder_layers"],
            n_heads=vae_cfg.get("decoder_n_heads", 4),
            ffn_dim=vae_cfg.get("decoder_ffn_dim", 256),
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
        aa_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode sequences to latent distribution.

        Args:
            sequences: List of amino acid strings (for PLM embedding).
            aa_features: [B, L, aa_dim] AA encoding (BLOSUM62+learned or one-hot).
            padding_mask: [B, L] True for padded positions.
        """
        plm_emb = self.plm(sequences, max_len=aa_features.size(1))
        plm_emb = plm_emb.to(aa_features.device)
        mu, logvar = self.encoder(aa_features, plm_emb, padding_mask)
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
        """Decode latent vector to sequence logits."""
        return self.decoder(
            z, properties, target_indices, teacher_forcing_ratio, target_len
        )

    def forward(
        self,
        sequences: list[str],
        aa_features: torch.Tensor,
        properties: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for VAE training."""
        z, mu, logvar = self.encode(sequences, aa_features, padding_mask)
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
