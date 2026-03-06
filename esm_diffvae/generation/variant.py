"""Conditional variant generation: generate biologically realistic AMP variants.

v7: Non-autoregressive variant generation.
Uses the non-autoregressive decoder — decode full sequence from z, then
splice the parent prefix with generated suffix for C-terminal variants.

Mutation modes:
  c_sub   : Keep first K AAs, substitute the last few (most common in nature)
  c_ext   : Keep entire parent, extend with new AAs at C-terminus
  c_trunc : Truncate last few AAs, regrow from the truncation point
  tag     : Append a common peptide tag (His, FLAG, etc.)
  latent  : Latent-space perturbation (whole-sequence, for diversity)
"""

import argparse
from datetime import datetime
import json
import math
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from training.dataset import (
    AA_TO_IDX, AA_VOCAB, PAD_IDX,
    indices_to_sequence, sequence_to_indices,
)
from training.utils import load_checkpoint, RunLogger


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common peptide tags for tag-addition mode
PEPTIDE_TAGS = {
    "his6": "HHHHHH",
    "his8": "HHHHHHHH",
    "flag": "DYKDDDDK",
    "strep": "WSHPQFEK",
    "myc": "EQKLISEEDL",
    "ha": "YPYDVPDYA",
    "gly_linker": "GGGGS",
    "ala_linker": "AAAA",
}


def _as_int_choices(value, default_choices):
    """Normalize config value to a sorted unique list of positive ints."""
    if isinstance(value, int):
        return [max(1, int(value))]
    if not isinstance(value, list) or not value:
        return list(default_choices)

    if len(value) == 2 and all(isinstance(v, int) for v in value) and value[1] >= value[0]:
        lo, hi = int(value[0]), int(value[1])
        lo = max(1, lo)
        hi = max(lo, hi)
        return list(range(lo, hi + 1))

    cleaned = sorted({max(1, int(v)) for v in value if isinstance(v, int)})
    return cleaned if cleaned else list(default_choices)


def _normalize_mode_weights(enabled_modes, mode_ratios):
    """Normalize mode ratios over enabled modes only."""
    raw = {m: float(mode_ratios.get(m, 0.0)) for m in enabled_modes}
    positive = {m: w for m, w in raw.items() if w > 0}
    if not positive:
        uniform = 1.0 / max(len(enabled_modes), 1)
        return {m: uniform for m in enabled_modes}

    total = sum(positive.values())
    return {m: positive.get(m, 0.0) / total for m in enabled_modes}


def _allocate_counts(total, weights):
    """Allocate integer counts from normalized weights (largest-remainder method)."""
    if total <= 0:
        return {k: 0 for k in weights}
    keys = list(weights.keys())
    raw = {k: total * float(weights[k]) for k in keys}
    counts = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(counts.values())
    if remainder > 0:
        order = sorted(keys, key=lambda k: raw[k] - counts[k], reverse=True)
        for k in order[:remainder]:
            counts[k] += 1
    return counts


def _dedupe_variants(variants, parent_seq):
    seen = set()
    unique = []
    parent_upper = parent_seq.upper()
    for v in variants:
        key = str(v["sequence"]).upper()
        if key == parent_upper or key in seen:
            continue
        seen.add(key)
        unique.append(v)
    return unique


def _get_variant_cfg(config):
    gen_cfg = config.get("generation", {})
    return gen_cfg.get("variant", {}) if isinstance(gen_cfg.get("variant", {}), dict) else {}


def encode_parent(model, input_seq, device):
    """Encode parent sequence to latent space."""
    seq_len = len(input_seq)
    with torch.no_grad():
        target_indices = sequence_to_indices(input_seq, model.max_len).unsqueeze(0).to(device)
        aa_features = model.aa_encoding(target_indices)  # [1, L, aa_dim]
        plm_emb = model.plm([input_seq], max_len=model.max_len).to(device)
        padding_mask = torch.zeros(1, model.max_len, dtype=torch.bool, device=device)
        padding_mask[:, seq_len:] = True
        mu, logvar = model.encoder(aa_features, plm_emb, padding_mask)
    return mu, logvar


def decode_full(model, z, properties, gen_len, temperature=1.0, top_k=0, top_p=0.9):
    """Decode z to full sequence using non-autoregressive decoder.

    Args:
        model: ESMDiffVAE model.
        z: [B, latent_dim] latent vectors.
        properties: [B, prop_dim] property conditions.
        gen_len: Target sequence length.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.

    Returns:
        [B, gen_len] sampled token indices.
    """
    with torch.no_grad():
        logits, _ = model.decode(z, properties, target_len=gen_len)

    # Suppress padding token
    logits[..., PAD_IDX] = float("-inf")
    return _sample_logits_batch(logits, temperature, top_k, top_p)


def splice_with_parent(tokens, parent_indices, preserve_len, device):
    """Replace the first `preserve_len` positions with parent tokens.

    For non-autoregressive decoder, we generate the full sequence and then
    splice the parent prefix back in.

    Args:
        tokens: [B, L] generated token indices.
        parent_indices: [max_len] parent sequence indices.
        preserve_len: Number of N-terminal positions to keep from parent.
        device: torch device.

    Returns:
        [B, L] token indices with parent prefix spliced in.
    """
    result = tokens.clone()
    parent_dev = parent_indices.to(device)
    result[:, :preserve_len] = parent_dev[:preserve_len].unsqueeze(0).expand(tokens.size(0), -1)
    return result


def _sample_step(logits, temperature=1.0, top_k=0, top_p=0.9):
    """Sample a single token from logits [B, vocab_size]."""
    # Suppress padding token
    logits[:, PAD_IDX] = float("-inf")

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.multinomial(probs, 1).squeeze(-1)  # [B]


def generate_c_terminal_substitution(
    model, input_seq, mu, logvar, n_variants, n_positions=3,
    properties=None, z_noise=0.1, temperature=0.8, top_k=10, top_p=0.9, device=None,
):
    """Mode: C-terminal substitution.

    Preserves the first (L - n_positions) AAs, freely regenerates the last n_positions.
    """
    seq_len = len(input_seq)
    preserve_len = max(1, seq_len - n_positions)
    parent_indices = sequence_to_indices(input_seq, model.max_len)

    if properties is None:
        properties = _default_properties(seq_len)
    prop = properties.unsqueeze(0).expand(n_variants, -1).to(device)

    # Sample z near the parent encoding
    z = mu.expand(n_variants, -1)
    if z_noise > 0:
        std = torch.exp(0.5 * logvar).expand(n_variants, -1)
        z = z + std * torch.randn_like(z) * z_noise

    tokens = decode_full(model, z, prop, seq_len, temperature=temperature, top_k=top_k, top_p=top_p)
    tokens = splice_with_parent(tokens, parent_indices, preserve_len, device)

    return _tokens_to_variants(tokens, input_seq, mode="c_sub")


def generate_c_terminal_extension(
    model, input_seq, mu, logvar, n_variants, extend_by=3,
    properties=None, z_noise=0.1, temperature=0.8, top_k=10, top_p=0.9, device=None,
):
    """Mode: C-terminal extension.

    Preserves the entire parent, extends by `extend_by` new AAs.
    """
    seq_len = len(input_seq)
    new_len = min(seq_len + extend_by, model.max_len)
    parent_indices = sequence_to_indices(input_seq, model.max_len)

    # Update length_norm in properties for the new length
    if properties is None:
        properties = _default_properties(new_len)
    prop = properties.unsqueeze(0).expand(n_variants, -1).to(device)

    z = mu.expand(n_variants, -1)
    if z_noise > 0:
        std = torch.exp(0.5 * logvar).expand(n_variants, -1)
        z = z + std * torch.randn_like(z) * z_noise

    tokens = decode_full(model, z, prop, new_len, temperature=temperature, top_k=top_k, top_p=top_p)
    tokens = splice_with_parent(tokens, parent_indices, seq_len, device)

    return _tokens_to_variants(tokens, input_seq, mode="c_ext")


def generate_c_terminal_truncation_regrow(
    model, input_seq, mu, logvar, n_variants, truncate_by=3,
    properties=None, z_noise=0.15, temperature=0.9, top_k=10, top_p=0.9, device=None,
):
    """Mode: truncate last few AAs and regrow from that point.

    Truncates the last `truncate_by` AAs, then freely regenerates from there
    to the original length.
    """
    seq_len = len(input_seq)
    preserve_len = max(1, seq_len - truncate_by)
    parent_indices = sequence_to_indices(input_seq, model.max_len)

    if properties is None:
        properties = _default_properties(seq_len)
    prop = properties.unsqueeze(0).expand(n_variants, -1).to(device)

    z = mu.expand(n_variants, -1)
    if z_noise > 0:
        std = torch.exp(0.5 * logvar).expand(n_variants, -1)
        z = z + std * torch.randn_like(z) * z_noise

    tokens = decode_full(model, z, prop, seq_len, temperature=temperature, top_k=top_k, top_p=top_p)
    tokens = splice_with_parent(tokens, parent_indices, preserve_len, device)

    return _tokens_to_variants(tokens, input_seq, mode="c_trunc")


def generate_tag_variants(input_seq, tags=None, linkers=None, max_len=50):
    """Mode: append peptide tags to the parent sequence.

    No model needed — direct sequence manipulation.
    """
    if tags is None:
        tags = list(PEPTIDE_TAGS.keys())
    if linkers is None:
        linkers = ["", "GG", "GGGGS"]

    variants = []
    for tag_name in tags:
        tag_seq = PEPTIDE_TAGS.get(tag_name, tag_name)
        # With and without configured linkers
        for linker in linkers:
            variant_seq = input_seq + linker + tag_seq
            if len(variant_seq) > max_len:
                continue
            variants.append({
                "sequence": variant_seq,
                "identity": len(input_seq) / len(variant_seq),
                "edit_distance": len(variant_seq) - len(input_seq),
                "length": len(variant_seq),
                "parent": input_seq,
                "mode": f"tag_{tag_name}" + (f"_linker_{linker}" if linker else ""),
            })
    return variants


def generate_latent_variants(
    model, input_seq, mu, logvar, n_variants,
    variation_strength=0.3, properties=None, guidance_scale=2.0,
    temperature=1.0, top_k=0, top_p=0.9, device=None,
):
    """Mode: traditional latent-space perturbation (whole-sequence diversity)."""
    seq_len = len(input_seq)
    if properties is None:
        properties = _default_properties(seq_len)
    prop = properties.unsqueeze(0).expand(n_variants, -1).to(device)

    z = mu.expand(n_variants, -1)
    std = torch.exp(0.5 * logvar).expand(n_variants, -1)
    z = z + std * torch.randn_like(z) * 0.1

    start_step = round(variation_strength * (model.diffusion.T - 1))
    start_step = max(1, min(start_step, model.diffusion.T - 1))

    with torch.no_grad():
        z_variants = model.diffusion.partial_denoise(z, start_step, prop, guidance_scale)
        logits, _ = model.decode(z_variants, prop, target_len=seq_len)

    logits[..., PAD_IDX] = float("-inf")
    tokens = _sample_logits_batch(logits, temperature, top_k, top_p)
    return _tokens_to_variants(tokens, input_seq, mode="latent")


def generate_variants(
    model,
    input_seq,
    n_variants=50,
    variation_strength=0.3,
    properties=None,
    guidance_scale=2.0,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    device=None,
):
    """Backward-compatible variant API used by evaluation scripts."""
    if device is None:
        device = next(model.parameters()).device
    input_seq = input_seq.upper().strip()
    mu, logvar = encode_parent(model, input_seq, device)
    return generate_latent_variants(
        model=model,
        input_seq=input_seq,
        mu=mu,
        logvar=logvar,
        n_variants=n_variants,
        variation_strength=variation_strength,
        properties=properties,
        guidance_scale=guidance_scale,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )


def _default_properties(seq_len):
    """Default properties: AMP, non-toxic, non-hemolytic."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0, seq_len / 50.0], dtype=torch.float32)


def _sample_logits_batch(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Sample from [B, L, V] logits."""
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)
        cutoff = values[..., -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = filtered
    probs = torch.softmax(logits, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])


def _tokens_to_variants(tokens, parent_seq, mode=""):
    """Convert token tensor to list of variant dicts."""
    variants = []
    seen = set()
    for i in range(tokens.size(0)):
        seq = indices_to_sequence(tokens[i])
        # Strip padding-like tokens at end
        seq = seq.rstrip()
        if not seq or seq.upper() == parent_seq.upper():
            continue
        if seq.upper() in seen:
            continue
        seen.add(seq.upper())
        variants.append({
            "sequence": seq,
            "identity": sequence_identity(parent_seq, seq),
            "edit_distance": edit_distance(parent_seq, seq),
            "length": len(seq),
            "parent": parent_seq,
            "mode": mode,
        })
    return variants


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    return matches / max_len


def edit_distance(seq1: str, seq2: str) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def highlight_mutations(parent: str, variant: str) -> str:
    """Show mutations: lowercase = changed, uppercase = preserved."""
    result = []
    for i, (p, v) in enumerate(zip(parent, variant)):
        if p == v:
            result.append(v)
        else:
            result.append(v.lower())
    # Extension beyond parent length
    if len(variant) > len(parent):
        result.append("|")
        result.extend(c.lower() for c in variant[len(parent):])
    return "".join(result)


def generate_all_variants(
    model, input_seq, mu, logvar, config, device,
    n_variants=50, properties=None,
):
    """Generate mixed variants according to config-driven mode ratios and strengths."""
    gen_cfg = config.get("generation", {})
    var_cfg = _get_variant_cfg(config)

    global_temp = float(var_cfg.get("temperature", gen_cfg.get("temperature", 0.8)))
    global_top_k = int(var_cfg.get("top_k", gen_cfg.get("top_k", 10)))
    global_top_p = float(var_cfg.get("top_p", gen_cfg.get("top_p", 0.9)))

    enabled_modes = var_cfg.get("enabled_modes", ["c_sub", "c_ext", "c_trunc", "tag", "latent"])
    enabled_modes = [m for m in enabled_modes if m in {"c_sub", "c_ext", "c_trunc", "tag", "latent"}]
    if not enabled_modes:
        enabled_modes = ["latent"]
    mode_ratios = var_cfg.get(
        "mode_ratios",
        {"c_sub": 0.4, "c_ext": 0.2, "c_trunc": 0.15, "tag": 0.1, "latent": 0.15},
    )
    mode_weights = _normalize_mode_weights(enabled_modes, mode_ratios)
    mode_counts = _allocate_counts(n_variants, mode_weights)

    all_variants = []
    for mode in enabled_modes:
        target_n = int(mode_counts.get(mode, 0))
        if target_n <= 0:
            continue

        if mode == "c_sub":
            cfg = var_cfg.get("c_sub", {})
            choices = _as_int_choices(cfg.get("n_positions", [1, 5]), [1, 2, 3, 4, 5])
            sub_counts = _allocate_counts(target_n, {str(v): 1.0 for v in choices})
            oversample = max(1, int(cfg.get("oversample_factor", 3)))
            mode_candidates = []
            for n_pos in choices:
                need = int(sub_counts[str(n_pos)])
                if need <= 0:
                    continue
                mode_candidates.extend(
                    generate_c_terminal_substitution(
                        model, input_seq, mu, logvar,
                        n_variants=max(need * oversample, need),
                        n_positions=n_pos,
                        properties=properties,
                        z_noise=float(cfg.get("z_noise", 0.10)),
                        temperature=float(cfg.get("temperature", global_temp)),
                        top_k=int(cfg.get("top_k", global_top_k)),
                        top_p=float(cfg.get("top_p", global_top_p)),
                        device=device,
                    )
                )
            mode_selected = _dedupe_variants(mode_candidates, input_seq)[:target_n]
            all_variants.extend(mode_selected)

        elif mode == "c_ext":
            cfg = var_cfg.get("c_ext", {})
            choices = _as_int_choices(cfg.get("extend_by", [1, 5]), [1, 2, 3, 4, 5])
            sub_counts = _allocate_counts(target_n, {str(v): 1.0 for v in choices})
            oversample = max(1, int(cfg.get("oversample_factor", 3)))
            mode_candidates = []
            for extend_by in choices:
                need = int(sub_counts[str(extend_by)])
                if need <= 0:
                    continue
                mode_candidates.extend(
                    generate_c_terminal_extension(
                        model, input_seq, mu, logvar,
                        n_variants=max(need * oversample, need),
                        extend_by=extend_by,
                        properties=properties,
                        z_noise=float(cfg.get("z_noise", 0.10)),
                        temperature=float(cfg.get("temperature", global_temp)),
                        top_k=int(cfg.get("top_k", global_top_k)),
                        top_p=float(cfg.get("top_p", global_top_p)),
                        device=device,
                    )
                )
            mode_selected = _dedupe_variants(mode_candidates, input_seq)[:target_n]
            all_variants.extend(mode_selected)

        elif mode == "c_trunc":
            cfg = var_cfg.get("c_trunc", {})
            choices = _as_int_choices(cfg.get("truncate_by", [2, 5]), [2, 3, 4, 5])
            sub_counts = _allocate_counts(target_n, {str(v): 1.0 for v in choices})
            oversample = max(1, int(cfg.get("oversample_factor", 3)))
            mode_candidates = []
            for truncate_by in choices:
                need = int(sub_counts[str(truncate_by)])
                if need <= 0:
                    continue
                mode_candidates.extend(
                    generate_c_terminal_truncation_regrow(
                        model, input_seq, mu, logvar,
                        n_variants=max(need * oversample, need),
                        truncate_by=truncate_by,
                        properties=properties,
                        z_noise=float(cfg.get("z_noise", 0.15)),
                        temperature=float(cfg.get("temperature", 0.9)),
                        top_k=int(cfg.get("top_k", global_top_k)),
                        top_p=float(cfg.get("top_p", global_top_p)),
                        device=device,
                    )
                )
            mode_selected = _dedupe_variants(mode_candidates, input_seq)[:target_n]
            all_variants.extend(mode_selected)

        elif mode == "tag":
            cfg = var_cfg.get("tag", {})
            tags = cfg.get("tags", list(PEPTIDE_TAGS.keys()))
            linkers = cfg.get("linkers", ["", "GG", "GGGGS"])
            mode_candidates = generate_tag_variants(
                input_seq,
                tags=tags,
                linkers=linkers,
                max_len=model.max_len,
            )
            random.shuffle(mode_candidates)
            mode_selected = _dedupe_variants(mode_candidates, input_seq)[:target_n]
            all_variants.extend(mode_selected)

        elif mode == "latent":
            cfg = var_cfg.get("latent", {})
            oversample = max(1, int(cfg.get("oversample_factor", 3)))
            guidance_scale = float(
                cfg.get("guidance_scale", config.get("diffusion", {}).get("guidance_scale", 1.2))
            )
            variation_strength = float(
                cfg.get(
                    "variation_strength",
                    var_cfg.get("default_variation_strength", gen_cfg.get("default_variation_strength", 0.2)),
                )
            )
            mode_candidates = generate_latent_variants(
                model, input_seq, mu, logvar,
                n_variants=max(target_n * oversample, target_n),
                variation_strength=variation_strength,
                properties=properties,
                guidance_scale=guidance_scale,
                temperature=float(cfg.get("temperature", 1.0)),
                top_k=int(cfg.get("top_k", 0)),
                top_p=float(cfg.get("top_p", global_top_p)),
                device=device,
            )
            mode_selected = _dedupe_variants(mode_candidates, input_seq)[:target_n]
            all_variants.extend(mode_selected)

    unique = _dedupe_variants(all_variants, input_seq)
    unique.sort(key=lambda v: (-v["identity"], v.get("mode", "")))
    return unique[:n_variants]


def main():
    parser = argparse.ArgumentParser(description="Generate biologically realistic AMP variants")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--input-sequence", required=True, help="Parent AMP sequence")
    parser.add_argument("--n-variants", type=int, default=None)
    parser.add_argument("--mode", default=None,
                        choices=["mixed", "c_sub", "c_ext", "c_trunc", "tag", "latent"],
                        help="Variant generation mode")
    parser.add_argument("--n-positions", type=int, default=None,
                        help="Number of C-terminal positions to modify (c_sub/c_trunc)")
    parser.add_argument("--extend-by", type=int, default=None,
                        help="Number of AAs to extend (c_ext)")
    parser.add_argument("--variation-strength", type=float, default=None,
                        help="Latent perturbation strength (latent mode)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--z-noise", type=float, default=None,
                        help="Noise scale for z perturbation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    run_name = f"generate_variant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_logger = RunLogger(ckpt_dir / "logs" / run_name, append=False)
    print(f"Run logs will be saved to: {run_logger.run_dir.resolve()}")

    gen_cfg = config.get("generation", {})
    var_cfg = _get_variant_cfg(config)
    mode = args.mode or var_cfg.get("mode", "mixed")
    selected_mode = mode
    n_variants = int(args.n_variants if args.n_variants is not None else var_cfg.get("n_variants", 50))
    global_temp = float(args.temperature if args.temperature is not None else var_cfg.get("temperature", gen_cfg.get("temperature", 0.8)))
    global_top_k = int(args.top_k if args.top_k is not None else var_cfg.get("top_k", gen_cfg.get("top_k", 10)))
    global_top_p = float(args.top_p if args.top_p is not None else var_cfg.get("top_p", gen_cfg.get("top_p", 0.9)))

    device = torch.device(args.device)
    model = ESMDiffVAE(config).to(device)
    load_checkpoint(model, args.checkpoint, device=args.device)
    model.eval()
    run_logger.info(
        f"run_start device={device} checkpoint={Path(args.checkpoint).resolve()} config={Path(args.config).resolve()} "
        f"mode={mode} n_variants={n_variants}"
    )

    input_seq = args.input_sequence.upper().strip()
    print(f"Input sequence: {input_seq} (len={len(input_seq)})")
    print(f"Mode: {mode}")

    # Encode parent
    mu, logvar = encode_parent(model, input_seq, device)

    if mode == "mixed":
        variants = generate_all_variants(
            model, input_seq, mu, logvar, config, device,
            n_variants=n_variants,
        )
    elif mode == "c_sub":
        csub_cfg = var_cfg.get("c_sub", {})
        n_positions = int(args.n_positions if args.n_positions is not None else csub_cfg.get("default_n_positions", 3))
        z_noise = float(args.z_noise if args.z_noise is not None else csub_cfg.get("z_noise", 0.1))
        variants = generate_c_terminal_substitution(
            model, input_seq, mu, logvar,
            n_variants=n_variants, n_positions=n_positions,
            z_noise=z_noise,
            temperature=float(csub_cfg.get("temperature", global_temp)),
            top_k=int(csub_cfg.get("top_k", global_top_k)),
            top_p=float(csub_cfg.get("top_p", global_top_p)),
            device=device,
        )
    elif mode == "c_ext":
        cext_cfg = var_cfg.get("c_ext", {})
        extend_by = int(args.extend_by if args.extend_by is not None else cext_cfg.get("default_extend_by", 3))
        z_noise = float(args.z_noise if args.z_noise is not None else cext_cfg.get("z_noise", 0.1))
        variants = generate_c_terminal_extension(
            model, input_seq, mu, logvar,
            n_variants=n_variants, extend_by=extend_by,
            z_noise=z_noise,
            temperature=float(cext_cfg.get("temperature", global_temp)),
            top_k=int(cext_cfg.get("top_k", global_top_k)),
            top_p=float(cext_cfg.get("top_p", global_top_p)),
            device=device,
        )
    elif mode == "c_trunc":
        ctrunc_cfg = var_cfg.get("c_trunc", {})
        truncate_by = int(args.n_positions if args.n_positions is not None else ctrunc_cfg.get("default_truncate_by", 3))
        z_noise = float(args.z_noise if args.z_noise is not None else ctrunc_cfg.get("z_noise", 0.15))
        variants = generate_c_terminal_truncation_regrow(
            model, input_seq, mu, logvar,
            n_variants=n_variants, truncate_by=truncate_by,
            z_noise=z_noise,
            temperature=float(ctrunc_cfg.get("temperature", 0.9)),
            top_k=int(ctrunc_cfg.get("top_k", global_top_k)),
            top_p=float(ctrunc_cfg.get("top_p", global_top_p)),
            device=device,
        )
    elif mode == "tag":
        tag_cfg = var_cfg.get("tag", {})
        variants = generate_tag_variants(
            input_seq,
            tags=tag_cfg.get("tags", list(PEPTIDE_TAGS.keys())),
            linkers=tag_cfg.get("linkers", ["", "GG", "GGGGS"]),
            max_len=model.max_len,
        )
    elif mode == "latent":
        latent_cfg = var_cfg.get("latent", {})
        guidance_scale = float(latent_cfg.get("guidance_scale", config.get("diffusion", {}).get("guidance_scale", 1.2)))
        variation_strength = float(
            args.variation_strength
            if args.variation_strength is not None
            else latent_cfg.get(
                "variation_strength",
                var_cfg.get("default_variation_strength", gen_cfg.get("default_variation_strength", 0.2)),
            )
        )
        variants = generate_latent_variants(
            model, input_seq, mu, logvar,
            n_variants=n_variants,
            variation_strength=variation_strength,
            guidance_scale=guidance_scale,
            temperature=float(latent_cfg.get("temperature", global_temp)),
            top_k=int(latent_cfg.get("top_k", global_top_k)),
            top_p=float(latent_cfg.get("top_p", global_top_p)),
            device=device,
        )
    else:
        variants = []

    # Display
    print(f"\nGenerated {len(variants)} variants:")
    print(f"{'#':>3} {'Mode':<8} {'Sequence':<45} {'Highlight':<45} {'Len':>4} {'Id':>6} {'Edit':>4}")
    print("-" * 120)
    for i, v in enumerate(variants[:30]):
        seq = v['sequence']
        hl = highlight_mutations(input_seq, seq)
        seq_disp = seq[:42] + "..." if len(seq) > 45 else seq
        hl_disp = hl[:42] + "..." if len(hl) > 45 else hl
        mode = v.get('mode', '')[:8]
        print(f"{i+1:3d} {mode:<8} {seq_disp:<45} {hl_disp:<45} {v['length']:4d} {v['identity']:5.1%} {v['edit_distance']:4d}")

    if len(variants) > 30:
        print(f"... and {len(variants) - 30} more variants")

    # Summary by mode
    if variants:
        modes = {}
        for v in variants:
            m = v.get("mode", "unknown")
            modes.setdefault(m, []).append(v)
        print(f"\nSummary by mode:")
        for mode_name, vlist in sorted(modes.items()):
            ids = [v["identity"] for v in vlist]
            eds = [v["edit_distance"] for v in vlist]
            print(f"  {mode_name:<12}: {len(vlist):3d} variants, "
                  f"identity={sum(ids)/len(ids):.1%} [{min(ids):.1%}-{max(ids):.1%}], "
                  f"edit_dist={sum(eds)/len(eds):.1f} [{min(eds)}-{max(eds)}]")

        identities = [v["identity"] for v in variants]
        edit_dists = [v["edit_distance"] for v in variants]
        print(f"\nOverall: {len(variants)} variants, "
              f"identity={sum(identities)/len(identities):.1%}, "
              f"edit_dist={sum(edit_dists)/len(edit_dists):.1f}")

    # Save
    if args.output is None:
        output_path = PROJECT_ROOT / config["paths"]["results_dir"] / "variants_generated.fasta"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".fasta")
    with open(output_path, "w") as f:
        f.write(f">parent {input_seq}\n{input_seq}\n")
        for i, v in enumerate(variants):
            mode = v.get("mode", "")
            f.write(f">variant_{i+1} mode={mode} identity={v['identity']:.3f} "
                    f"edit_dist={v['edit_distance']} len={v['length']}\n")
            f.write(f"{v['sequence']}\n")
    print(f"\nSaved to {output_path}")

    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "input_sequence": input_seq,
            "mode": selected_mode,
            "n_generated": len(variants),
            "variants": variants,
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    run_logger.write_result({
        "run_name": run_name,
        "status": "completed",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": selected_mode,
        "input_sequence": input_seq,
        "n_generated": int(len(variants)),
        "output_fasta": str(output_path.resolve()),
        "output_metrics_json": str(metrics_path.resolve()),
    })
    run_logger.info("run_complete")
    print(f"Run logs saved to: {run_logger.run_dir.resolve()}")
    print(f"Run summary: {(run_logger.run_dir / 'result_summary.json').resolve()}")


if __name__ == "__main__":
    main()
