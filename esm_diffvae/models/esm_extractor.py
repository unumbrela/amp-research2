"""Frozen ESM-2 wrapper for per-residue embedding extraction."""

import torch
import torch.nn as nn
import esm


class ESMFeatureExtractor(nn.Module):
    """Extract per-residue embeddings from a frozen ESM-2 model.

    Uses esm2_t6_8M_UR50D by default (8M params, 320-dim embeddings),
    suitable for single consumer GPU (8-12GB).
    """

    MODEL_REGISTRY = {
        "esm2_t6_8M_UR50D": (esm.pretrained.esm2_t6_8M_UR50D, 320),
        "esm2_t12_35M_UR50D": (esm.pretrained.esm2_t12_35M_UR50D, 480),
    }

    def __init__(self, model_name: str = "esm2_t6_8M_UR50D"):
        super().__init__()
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODEL_REGISTRY)}")

        loader, self.embedding_dim = self.MODEL_REGISTRY[model_name]
        self.model, self.alphabet = loader()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.repr_layer = self.model.num_layers

    @torch.no_grad()
    def forward(self, sequences: list[str], max_len: int = 50) -> torch.Tensor:
        """Extract per-residue embeddings.

        Args:
            sequences: List of amino acid strings.
            max_len: Pad/truncate to this length.

        Returns:
            Tensor of shape [B, max_len, embedding_dim].
        """
        data = [(f"seq_{i}", seq[:max_len]) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)

        results = self.model(batch_tokens, repr_layers=[self.repr_layer])
        # results["representations"][layer] has shape [B, L+2, D] (with BOS/EOS tokens)
        embeddings = results["representations"][self.repr_layer]

        # Remove BOS token (index 0), keep up to max_len residues
        embeddings = embeddings[:, 1 : max_len + 1, :]

        # Pad if sequence is shorter than max_len
        B, L, D = embeddings.shape
        if L < max_len:
            pad = torch.zeros(B, max_len - L, D, device=embeddings.device)
            embeddings = torch.cat([embeddings, pad], dim=1)

        return embeddings

    @torch.no_grad()
    def extract_batch(
        self, sequences: list[str], max_len: int = 50, batch_size: int = 32
    ) -> torch.Tensor:
        """Extract embeddings in mini-batches for large datasets."""
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            emb = self.forward(batch_seqs, max_len)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)
