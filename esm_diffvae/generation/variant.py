"""Conditional variant generation: generate biologically realistic AMP variants.

v5: C-terminal focused variant generation.
Leverages the autoregressive decoder's ability to do partial teacher forcing:
- Preserves N-terminal region (forced to match parent)
- Freely generates C-terminal modifications (substitutions, extensions, truncation+regrow)
- Optionally appends common peptide tags

Mutation modes:
  c_sub   : Keep first K AAs, substitute the last few (most common in nature)
  c_ext   : Keep entire parent, extend with 1-5 new AAs at C-terminus
  c_trunc : Truncate last few AAs, regrow from the truncation point
  tag     : Append a common peptide tag (His, FLAG, etc.)
  latent  : Traditional latent-space perturbation (whole-sequence, for diversity)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from training.dataset import (
    AA_TO_IDX, AA_VOCAB, PAD_IDX,
    indices_to_sequence, sequence_to_indices, sequence_to_one_hot,
)
from training.utils import load_checkpoint


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


def encode_parent(model, input_seq, device):
    """Encode parent sequence to latent space."""
    seq_len = len(input_seq)
    with torch.no_grad():
        one_hot = sequence_to_one_hot(input_seq, model.max_len).unsqueeze(0).to(device)
        esm_emb = model.esm([input_seq], max_len=model.max_len).to(device)
        padding_mask = torch.zeros(1, model.max_len, dtype=torch.bool, device=device)
        padding_mask[:, seq_len:] = True
        mu, logvar = model.encoder(one_hot, esm_emb, padding_mask)
    return mu, logvar


def decode_with_prefix(model, z, properties, parent_indices, preserve_len, gen_len, temperature=1.0, top_k=0, top_p=0.9):
    """Decode autoregressively with partial teacher forcing.

    Forces the first `preserve_len` positions to match `parent_indices`,
    then freely generates positions `preserve_len` to `gen_len-1`.

    Args:
        model: ESMDiffVAE model.
        z: [B, latent_dim] latent vectors.
        properties: [B, prop_dim] property conditions.
        parent_indices: [max_len] token indices of parent sequence.
        preserve_len: Number of N-terminal positions to preserve.
        gen_len: Total output sequence length.
        temperature: Sampling temperature for free-running positions.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.

    Returns:
        [B, gen_len] generated token indices.
    """
    B = z.size(0)
    device = z.device
    decoder = model.decoder

    h = decoder._init_hidden(z, properties)
    cond = torch.cat([z, properties], dim=-1)
    cond_step = cond.unsqueeze(1)  # [B, 1, cond_dim]

    bos_idx = decoder.bos_idx
    input_token = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    all_tokens = []

    parent_dev = parent_indices.to(device)

    for t in range(gen_len):
        emb = decoder.token_embedding(input_token)
        gru_input = torch.cat([emb, cond_step], dim=-1)
        output, h = decoder.gru(gru_input, h)
        logit = decoder.output_proj(output)  # [B, 1, vocab_size]

        if t < preserve_len:
            # Force to parent token
            token = parent_dev[t].expand(B, 1)
        else:
            # Free generation with sampling
            token = _sample_step(logit.squeeze(1), temperature, top_k, top_p)
            token = token.unsqueeze(1)  # [B, 1]

        all_tokens.append(token)
        input_token = token

    return torch.cat(all_tokens, dim=1)  # [B, gen_len]


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

    with torch.no_grad():
        tokens = decode_with_prefix(
            model, z, prop, parent_indices, preserve_len, seq_len,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

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

    with torch.no_grad():
        tokens = decode_with_prefix(
            model, z, prop, parent_indices, seq_len, new_len,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

    return _tokens_to_variants(tokens, input_seq, mode="c_ext")


def generate_c_terminal_truncation_regrow(
    model, input_seq, mu, logvar, n_variants, truncate_by=3,
    properties=None, z_noise=0.15, temperature=0.9, top_k=10, top_p=0.9, device=None,
):
    """Mode: truncate last few AAs and regrow from that point.

    Truncates the last `truncate_by` AAs, then freely regenerates from there
    to the original length (or slightly different).
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

    with torch.no_grad():
        tokens = decode_with_prefix(
            model, z, prop, parent_indices, preserve_len, seq_len,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

    return _tokens_to_variants(tokens, input_seq, mode="c_trunc")


def generate_tag_variants(input_seq, tags=None):
    """Mode: append peptide tags to the parent sequence.

    No model needed — direct sequence manipulation.
    """
    if tags is None:
        tags = list(PEPTIDE_TAGS.keys())

    variants = []
    for tag_name in tags:
        tag_seq = PEPTIDE_TAGS.get(tag_name, tag_name)
        # With and without glycine linker
        for linker in ["", "GG", "GGGGS"]:
            variant_seq = input_seq + linker + tag_seq
            if len(variant_seq) > 50:
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
    """Generate variants using a mix of biologically realistic modes.

    Distribution:
      40% C-terminal substitution (1-5 positions)
      20% C-terminal extension (1-5 AAs)
      15% C-terminal truncation + regrow
      10% Tag addition
      15% Latent perturbation (for diversity)
    """
    gen_cfg = config.get("generation", {})
    temperature = gen_cfg.get("temperature", 0.8)
    top_k = gen_cfg.get("top_k", 10)
    top_p = gen_cfg.get("top_p", 0.9)

    all_variants = []

    # C-terminal substitution (largest group)
    n_csub = max(1, int(n_variants * 0.4))
    for n_pos in [1, 2, 3, 4, 5]:
        batch_n = max(1, n_csub // 5)
        variants = generate_c_terminal_substitution(
            model, input_seq, mu, logvar, n_variants=batch_n * 3,
            n_positions=n_pos, properties=properties,
            z_noise=0.1, temperature=temperature, top_k=top_k, top_p=top_p, device=device,
        )
        all_variants.extend(variants[:batch_n])

    # C-terminal extension
    n_cext = max(1, int(n_variants * 0.2))
    for ext in [1, 2, 3, 4, 5]:
        batch_n = max(1, n_cext // 5)
        variants = generate_c_terminal_extension(
            model, input_seq, mu, logvar, n_variants=batch_n * 3,
            extend_by=ext, properties=properties,
            z_noise=0.1, temperature=temperature, top_k=top_k, top_p=top_p, device=device,
        )
        all_variants.extend(variants[:batch_n])

    # C-terminal truncation + regrow
    n_ctrunc = max(1, int(n_variants * 0.15))
    for trunc in [2, 3, 4, 5]:
        batch_n = max(1, n_ctrunc // 4)
        variants = generate_c_terminal_truncation_regrow(
            model, input_seq, mu, logvar, n_variants=batch_n * 3,
            truncate_by=trunc, properties=properties,
            z_noise=0.15, temperature=0.9, top_k=top_k, top_p=top_p, device=device,
        )
        all_variants.extend(variants[:batch_n])

    # Tag addition (no model needed)
    tag_variants = generate_tag_variants(input_seq)
    all_variants.extend(tag_variants)

    # Latent perturbation (for diversity)
    n_latent = max(1, int(n_variants * 0.15))
    variation_strength = gen_cfg.get("default_variation_strength", 0.2)
    guidance_scale = config.get("diffusion", {}).get("guidance_scale", 1.2)
    latent_variants = generate_latent_variants(
        model, input_seq, mu, logvar, n_variants=n_latent * 3,
        variation_strength=variation_strength, properties=properties,
        guidance_scale=guidance_scale,
        temperature=1.0, top_k=0, top_p=top_p, device=device,
    )
    all_variants.extend(latent_variants[:n_latent])

    # Deduplicate
    seen = set()
    unique = []
    for v in all_variants:
        key = v["sequence"].upper()
        if key not in seen and key != input_seq.upper():
            seen.add(key)
            unique.append(v)

    # Sort by identity (high first), then by mode for grouping
    unique.sort(key=lambda v: (-v["identity"], v.get("mode", "")))
    return unique[:n_variants]


def main():
    parser = argparse.ArgumentParser(description="Generate biologically realistic AMP variants")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--input-sequence", required=True, help="Parent AMP sequence")
    parser.add_argument("--n-variants", type=int, default=50)
    parser.add_argument("--mode", default="mixed",
                        choices=["mixed", "c_sub", "c_ext", "c_trunc", "tag", "latent"],
                        help="Variant generation mode")
    parser.add_argument("--n-positions", type=int, default=3,
                        help="Number of C-terminal positions to modify (c_sub/c_trunc)")
    parser.add_argument("--extend-by", type=int, default=3,
                        help="Number of AAs to extend (c_ext)")
    parser.add_argument("--variation-strength", type=float, default=0.2,
                        help="Latent perturbation strength (latent mode)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--z-noise", type=float, default=0.1,
                        help="Noise scale for z perturbation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = ESMDiffVAE(config).to(device)
    load_checkpoint(model, args.checkpoint, device=args.device)
    model.eval()

    input_seq = args.input_sequence.upper().strip()
    print(f"Input sequence: {input_seq} (len={len(input_seq)})")
    print(f"Mode: {args.mode}")

    # Encode parent
    mu, logvar = encode_parent(model, input_seq, device)

    if args.mode == "mixed":
        variants = generate_all_variants(
            model, input_seq, mu, logvar, config, device,
            n_variants=args.n_variants,
        )
    elif args.mode == "c_sub":
        variants = generate_c_terminal_substitution(
            model, input_seq, mu, logvar,
            n_variants=args.n_variants, n_positions=args.n_positions,
            z_noise=args.z_noise, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, device=device,
        )
    elif args.mode == "c_ext":
        variants = generate_c_terminal_extension(
            model, input_seq, mu, logvar,
            n_variants=args.n_variants, extend_by=args.extend_by,
            z_noise=args.z_noise, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, device=device,
        )
    elif args.mode == "c_trunc":
        variants = generate_c_terminal_truncation_regrow(
            model, input_seq, mu, logvar,
            n_variants=args.n_variants, truncate_by=args.n_positions,
            z_noise=args.z_noise, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, device=device,
        )
    elif args.mode == "tag":
        variants = generate_tag_variants(input_seq)
    elif args.mode == "latent":
        guidance_scale = config.get("diffusion", {}).get("guidance_scale", 1.2)
        variants = generate_latent_variants(
            model, input_seq, mu, logvar,
            n_variants=args.n_variants,
            variation_strength=args.variation_strength,
            guidance_scale=guidance_scale,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
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
        for mode, vlist in sorted(modes.items()):
            ids = [v["identity"] for v in vlist]
            eds = [v["edit_distance"] for v in vlist]
            print(f"  {mode:<12}: {len(vlist):3d} variants, "
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
            "mode": args.mode,
            "n_generated": len(variants),
            "variants": variants,
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
