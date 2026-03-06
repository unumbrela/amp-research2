"""Unified protein language model feature extractor.

Supports multiple PLM backends via a common interface:
- ESM-2: Facebook's ESM family (320-480 dim, 8M-35M params)
- Ankh: T5-based protein model (768-1536 dim)
- ProtT5: ProtTrans T5 encoder (1024 dim)

All backends are frozen (no gradients) and output per-residue embeddings
with the same format: [B, max_len, embedding_dim].
"""

import re
import os
from pathlib import Path

import torch
import torch.nn as nn


# Backend → {model_name: embedding_dim}
BACKEND_REGISTRY = {
    "esm2": {
        "esm2_t6_8M_UR50D": 320,
        "esm2_t12_35M_UR50D": 480,
    },
    "ankh": {
        "ankh-base": 768,
        "ankh-large": 1536,
    },
    "prot_t5": {
        "prot_t5_xl_half": 1024,
    },
}


class PLMExtractor(nn.Module):
    """Unified protein language model feature extractor.

    All backends are frozen and extract per-residue embeddings.
    Output shape: [B, max_len, embedding_dim].
    """

    def __init__(self, backend: str = "esm2", model_name: str = "esm2_t6_8M_UR50D"):
        super().__init__()
        if backend not in BACKEND_REGISTRY:
            raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKEND_REGISTRY)}")
        if model_name not in BACKEND_REGISTRY[backend]:
            raise ValueError(
                f"Unknown model '{model_name}' for backend '{backend}'. "
                f"Choose from {list(BACKEND_REGISTRY[backend])}"
            )

        self.backend = backend
        self.model_name = model_name
        self.embedding_dim = BACKEND_REGISTRY[backend][model_name]

        if backend == "esm2":
            self._init_esm2()
        elif backend == "ankh":
            self._init_ankh()
        elif backend == "prot_t5":
            self._init_prot_t5()

    # ── ESM-2 ────────────────────────────────────────────────

    def _init_esm2(self):
        import esm
        loader_map = {
            "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
            "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        }
        loader = loader_map[self.model_name]
        self.model, self.alphabet = loader()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = self.model.num_layers
        self._freeze()

    def _forward_esm2(self, sequences: list[str], max_len: int) -> torch.Tensor:
        data = [(f"seq_{i}", seq[:max_len]) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self._device())

        results = self.model(batch_tokens, repr_layers=[self.repr_layer])
        embeddings = results["representations"][self.repr_layer]
        # Remove BOS token (index 0), keep up to max_len
        embeddings = embeddings[:, 1: max_len + 1, :]
        return self._pad_to_max_len(embeddings, max_len)

    # ── Ankh ─────────────────────────────────────────────────

    def _init_ankh(self):
        import ankh

        # ankh API changed names across versions:
        # old: load_ankh_base/load_ankh_large
        # new: load_base_model/load_large_model
        loader_candidates = {
            "ankh-base": ("load_base_model", "load_ankh_base"),
            "ankh-large": ("load_large_model", "load_ankh_large"),
        }
        primary_name, legacy_name = loader_candidates[self.model_name]
        loader = getattr(ankh, primary_name, None) or getattr(ankh, legacy_name, None)
        if loader is None:
            raise AttributeError(
                f"Unsupported ankh package version: missing both '{primary_name}' "
                f"and legacy '{legacy_name}' loader functions."
            )

        self.model, self.tokenizer = loader()
        self._freeze()

    def _forward_ankh(self, sequences: list[str], max_len: int) -> torch.Tensor:
        # Ankh tokenizer expects list of characters per sequence
        char_seqs = [list(seq[:max_len]) for seq in sequences]
        outputs = self.tokenizer(
            char_seqs,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        input_ids = outputs["input_ids"].to(self._device())
        attention_mask = outputs["attention_mask"].to(self._device())

        result = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = result.last_hidden_state

        # Remove special tokens: Ankh T5 adds EOS at end
        # Keep only the first max_len residue positions
        embeddings = embeddings[:, :max_len, :]
        return self._pad_to_max_len(embeddings, max_len)

    # ── ProtT5 ───────────────────────────────────────────────

    def _init_prot_t5(self):
        from transformers import T5Tokenizer, T5EncoderModel

        hf_model_id = "Rostlab/prot_t5_xl_half_uniref50-enc"
        local_model_dir = Path(
            os.getenv("PROT_T5_LOCAL_DIR", "/home/zihao/models/prot_t5_xl_half_uniref50_enc")
        )
        model_source = str(local_model_dir) if local_model_dir.exists() else hf_model_id
        local_only = local_model_dir.exists()

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_source,
            do_lower_case=False,
            local_files_only=local_only,
        )
        self.model = T5EncoderModel.from_pretrained(
            model_source,
            local_files_only=local_only,
        )
        self.model.half()  # float16, no performance loss per ProtTrans docs
        self._freeze()

    def _forward_prot_t5(self, sequences: list[str], max_len: int) -> torch.Tensor:
        # ProtT5 requires space-separated AAs, rare AAs replaced with X
        processed = [
            " ".join(list(re.sub(r"[UZOB]", "X", seq[:max_len])))
            for seq in sequences
        ]
        ids = self.tokenizer.batch_encode_plus(
            processed,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = ids["input_ids"].to(self._device())
        attention_mask = ids["attention_mask"].to(self._device())

        result = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = result.last_hidden_state

        # ProtT5 adds EOS token at end — keep only residue positions
        # Each space-separated AA becomes one token, so first L tokens = residues
        embeddings = embeddings[:, :max_len, :]

        # ProtT5 outputs float16, convert to float32 for downstream
        embeddings = embeddings.float()
        return self._pad_to_max_len(embeddings, max_len)

    # ── Common utilities ─────────────────────────────────────

    def _freeze(self):
        """Freeze all model parameters."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _pad_to_max_len(self, embeddings: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pad embeddings to [B, max_len, D] if shorter."""
        B, L, D = embeddings.shape
        if L < max_len:
            pad = torch.zeros(B, max_len - L, D, device=embeddings.device, dtype=embeddings.dtype)
            embeddings = torch.cat([embeddings, pad], dim=1)
        return embeddings

    @torch.no_grad()
    def forward(self, sequences: list[str], max_len: int = 50) -> torch.Tensor:
        """Extract per-residue embeddings.

        Args:
            sequences: List of amino acid strings.
            max_len: Pad/truncate to this length.

        Returns:
            [B, max_len, embedding_dim] tensor.
        """
        if self.backend == "esm2":
            return self._forward_esm2(sequences, max_len)
        elif self.backend == "ankh":
            return self._forward_ankh(sequences, max_len)
        elif self.backend == "prot_t5":
            return self._forward_prot_t5(sequences, max_len)

    @torch.no_grad()
    def extract_batch(
        self, sequences: list[str], max_len: int = 50, batch_size: int = 32
    ) -> torch.Tensor:
        """Extract embeddings in mini-batches for large datasets."""
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i: i + batch_size]
            emb = self.forward(batch_seqs, max_len)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)
