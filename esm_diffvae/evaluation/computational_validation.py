"""Computational validation of generated AMP variants.

Evaluates generated sequences without wet-lab experiments using:
1. ESM-2 pseudo-perplexity (protein language model confidence)
2. Physicochemical property analysis
3. Amphipathic helix analysis (helical wheel)
4. Mutation position distribution analysis
5. Comprehensive scoring report
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.physicochemical import (
    compute_all_properties,
    compute_charge,
    compute_hydrophobicity,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard amino acid alphabet
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC_AAS = set("AILMFVWP")
POLAR_AAS = set("STNQYC")
CHARGED_POS_AAS = set("KRH")
CHARGED_NEG_AAS = set("DE")


# ---------------------------------------------------------------------------
# 1. ESM-2 Pseudo-Perplexity
# ---------------------------------------------------------------------------

def compute_esm2_pseudo_perplexity(
    sequences: list[str],
    model_name: str = "esm2_t6_8M_UR50D",
    device: str = "cuda",
    batch_size: int = 16,
) -> list[float]:
    """Compute ESM-2 pseudo-perplexity via masked language modeling.

    For each position, mask it and compute the log-probability of the true token.
    Lower pseudo-perplexity = ESM-2 considers the sequence more protein-like.

    Args:
        sequences: List of amino acid sequences.
        model_name: ESM-2 model name.
        device: Compute device.
        batch_size: Batch size for processing.

    Returns:
        List of pseudo-perplexity values (one per sequence).
    """
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    ppls = []
    for seq in sequences:
        seq_len = len(seq)
        # Create masked versions: one per position
        data = [(f"pos_{i}", seq[:i] + "<mask>" + seq[i + 1:]) for i in range(seq_len)]

        total_log_prob = 0.0
        for start in range(0, len(data), batch_size):
            batch_data = data[start:start + batch_size]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[], return_contacts=False)
                logits = results["logits"]  # [B, L+2, vocab]

            # For each masked position, get log-prob of true token
            for j, (label, _) in enumerate(batch_data):
                pos = int(label.split("_")[1])
                true_token_idx = alphabet.get_idx(seq[pos])
                # Position in tokenized sequence: +1 for BOS token
                log_probs = torch.log_softmax(logits[j, pos + 1], dim=-1)
                total_log_prob += log_probs[true_token_idx].item()

        avg_nll = -total_log_prob / seq_len
        ppls.append(math.exp(avg_nll))

    return ppls


# ---------------------------------------------------------------------------
# 2. Amphipathic Helix Analysis
# ---------------------------------------------------------------------------

def helical_wheel_amphipathicity(sequence: str, angle: float = 100.0) -> float:
    """Compute amphipathicity score using helical wheel projection.

    Projects residues onto a helical wheel (100 degrees per residue for alpha-helix)
    and computes the hydrophobic moment — a measure of amphipathicity.

    Higher values indicate stronger amphipathic character, which is a key
    feature of membrane-active antimicrobial peptides.

    Args:
        sequence: Amino acid sequence.
        angle: Rotation angle per residue (100 degrees for alpha-helix).

    Returns:
        Hydrophobic moment (amphipathicity score).
    """
    from evaluation.physicochemical import KD_SCALE

    angle_rad = math.radians(angle)
    sin_sum = 0.0
    cos_sum = 0.0

    for i, aa in enumerate(sequence.upper()):
        h = KD_SCALE.get(aa, 0.0)
        theta = i * angle_rad
        sin_sum += h * math.sin(theta)
        cos_sum += h * math.cos(theta)

    n = len(sequence) if sequence else 1
    return math.sqrt(sin_sum ** 2 + cos_sum ** 2) / n


def helical_wheel_positions(sequence: str, angle: float = 100.0) -> list[dict]:
    """Get helical wheel projection coordinates for visualization.

    Args:
        sequence: Amino acid sequence.
        angle: Rotation angle per residue.

    Returns:
        List of dicts with 'aa', 'position', 'x', 'y', 'hydrophobic'.
    """
    from evaluation.physicochemical import KD_SCALE

    angle_rad = math.radians(angle)
    positions = []
    for i, aa in enumerate(sequence.upper()):
        theta = i * angle_rad
        positions.append({
            "aa": aa,
            "position": i,
            "x": math.cos(theta),
            "y": math.sin(theta),
            "hydrophobic": KD_SCALE.get(aa, 0.0) > 0,
        })
    return positions


# ---------------------------------------------------------------------------
# 3. Mutation Position Distribution Analysis
# ---------------------------------------------------------------------------

def analyze_mutation_positions(
    parent: str, variants: list[str]
) -> dict:
    """Analyze where mutations occur relative to the parent sequence.

    Returns statistics on mutation distribution across N-terminal,
    middle, and C-terminal regions.
    """
    parent_len = len(parent)
    if parent_len == 0:
        return {}

    # Define regions: N-term (first 1/3), middle (middle 1/3), C-term (last 1/3)
    n_term_end = parent_len // 3
    c_term_start = parent_len - parent_len // 3

    position_counts = np.zeros(parent_len, dtype=int)
    region_counts = {"n_terminal": 0, "middle": 0, "c_terminal": 0}
    total_mutations = 0

    for variant in variants:
        min_len = min(len(variant), parent_len)
        for i in range(min_len):
            if variant[i] != parent[i]:
                position_counts[i] += 1
                total_mutations += 1
                if i < n_term_end:
                    region_counts["n_terminal"] += 1
                elif i >= c_term_start:
                    region_counts["c_terminal"] += 1
                else:
                    region_counts["middle"] += 1

    # Normalize by region size
    n_term_size = n_term_end
    middle_size = c_term_start - n_term_end
    c_term_size = parent_len - c_term_start

    if total_mutations > 0:
        region_density = {
            "n_terminal": region_counts["n_terminal"] / max(n_term_size, 1) / len(variants),
            "middle": region_counts["middle"] / max(middle_size, 1) / len(variants),
            "c_terminal": region_counts["c_terminal"] / max(c_term_size, 1) / len(variants),
        }
    else:
        region_density = {"n_terminal": 0.0, "middle": 0.0, "c_terminal": 0.0}

    return {
        "position_counts": position_counts.tolist(),
        "region_counts": region_counts,
        "region_density": region_density,
        "total_mutations": total_mutations,
        "mean_mutations_per_variant": total_mutations / max(len(variants), 1),
        "most_mutated_positions": [
            int(i) for i in np.argsort(position_counts)[::-1][:5]
        ],
    }


# ---------------------------------------------------------------------------
# 4. Amino Acid Composition Analysis
# ---------------------------------------------------------------------------

def aa_composition_analysis(sequences: list[str]) -> dict:
    """Analyze amino acid composition vs typical AMP distributions."""
    # Typical AMP amino acid frequencies (from APD3 database)
    typical_amp_freq = {
        "G": 0.093, "A": 0.073, "L": 0.089, "K": 0.102, "I": 0.053,
        "V": 0.041, "F": 0.046, "R": 0.044, "W": 0.026, "S": 0.039,
        "C": 0.069, "N": 0.026, "P": 0.030, "T": 0.031, "H": 0.020,
        "E": 0.018, "D": 0.017, "Q": 0.016, "M": 0.012, "Y": 0.016,
    }

    # Count amino acid frequencies in generated sequences
    total_count = 0
    aa_counts = {aa: 0 for aa in AA_VOCAB}
    for seq in sequences:
        for aa in seq.upper():
            if aa in aa_counts:
                aa_counts[aa] += 1
                total_count += 1

    if total_count == 0:
        return {"generated_freq": {}, "typical_amp_freq": typical_amp_freq, "kl_divergence": float("inf")}

    generated_freq = {aa: count / total_count for aa, count in aa_counts.items()}

    # KL divergence
    kl_div = 0.0
    for aa in AA_VOCAB:
        p = generated_freq.get(aa, 1e-10)
        q = typical_amp_freq.get(aa, 1e-10)
        if p > 0:
            kl_div += p * math.log(p / q)

    # Composition ratios
    hydrophobic_ratio = sum(generated_freq.get(aa, 0) for aa in HYDROPHOBIC_AAS)
    charged_ratio = sum(generated_freq.get(aa, 0) for aa in CHARGED_POS_AAS | CHARGED_NEG_AAS)
    pos_charge_ratio = sum(generated_freq.get(aa, 0) for aa in CHARGED_POS_AAS)

    return {
        "generated_freq": generated_freq,
        "typical_amp_freq": typical_amp_freq,
        "kl_divergence": kl_div,
        "hydrophobic_ratio": hydrophobic_ratio,
        "charged_ratio": charged_ratio,
        "positive_charge_ratio": pos_charge_ratio,
    }


# ---------------------------------------------------------------------------
# 5. Comprehensive Scoring
# ---------------------------------------------------------------------------

def score_sequence(
    sequence: str,
    parent: str | None = None,
    ppl: float | None = None,
) -> dict:
    """Compute a comprehensive quality score for a generated sequence.

    Scores are in [0, 1] where 1 is best. The overall score is the mean.
    """
    scores = {}

    # Physicochemical properties
    props = compute_all_properties(sequence)

    # Length score: AMPs typically 10-50 AA
    length = props["length"]
    if 10 <= length <= 50:
        scores["length"] = 1.0
    elif 5 <= length < 10 or 50 < length <= 60:
        scores["length"] = 0.5
    else:
        scores["length"] = 0.0

    # Charge score: AMPs typically cationic (+2 to +9)
    charge = props["charge_ph7"]
    if 2.0 <= charge <= 9.0:
        scores["charge"] = 1.0
    elif 0.0 <= charge < 2.0 or 9.0 < charge <= 12.0:
        scores["charge"] = 0.5
    else:
        scores["charge"] = 0.0

    # Hydrophobicity score: AMPs moderately hydrophobic (-1.0 to 1.0 mean KD)
    hydro = props["hydrophobicity"]
    if -1.0 <= hydro <= 1.0:
        scores["hydrophobicity"] = 1.0
    elif -2.0 <= hydro < -1.0 or 1.0 < hydro <= 2.0:
        scores["hydrophobicity"] = 0.5
    else:
        scores["hydrophobicity"] = 0.0

    # Amphipathicity score
    amphipathicity = helical_wheel_amphipathicity(sequence)
    # Typical AMP amphipathicity: 0.3-1.5
    if 0.3 <= amphipathicity <= 1.5:
        scores["amphipathicity"] = 1.0
    elif 0.1 <= amphipathicity < 0.3 or 1.5 < amphipathicity <= 2.0:
        scores["amphipathicity"] = 0.5
    else:
        scores["amphipathicity"] = 0.0

    # ESM-2 pseudo-perplexity score (if provided)
    if ppl is not None:
        if ppl < 5.0:
            scores["esm2_ppl"] = 1.0
        elif ppl < 10.0:
            scores["esm2_ppl"] = 0.7
        elif ppl < 20.0:
            scores["esm2_ppl"] = 0.3
        else:
            scores["esm2_ppl"] = 0.0

    # Overall score
    scores["overall"] = sum(scores.values()) / len(scores) if scores else 0.0

    return {
        "scores": scores,
        "properties": props,
        "amphipathicity": amphipathicity,
        "esm2_ppl": ppl,
    }


# ---------------------------------------------------------------------------
# 6. Full Validation Pipeline
# ---------------------------------------------------------------------------

def validate_variants(
    parent_sequence: str,
    variant_sequences: list[str],
    compute_ppl: bool = True,
    esm_model: str = "esm2_t6_8M_UR50D",
    device: str = "cuda",
) -> dict:
    """Run full computational validation on generated variants.

    Args:
        parent_sequence: Original AMP sequence.
        variant_sequences: List of generated variant sequences.
        compute_ppl: Whether to compute ESM-2 pseudo-perplexity (slow).
        esm_model: ESM-2 model name.
        device: Compute device.

    Returns:
        Comprehensive validation report.
    """
    print(f"Validating {len(variant_sequences)} variants of {parent_sequence}")

    # 1. ESM-2 pseudo-perplexity
    ppls = None
    parent_ppl = None
    if compute_ppl:
        print("  Computing ESM-2 pseudo-perplexity...")
        all_seqs = [parent_sequence] + variant_sequences
        all_ppls = compute_esm2_pseudo_perplexity(all_seqs, esm_model, device)
        parent_ppl = all_ppls[0]
        ppls = all_ppls[1:]
        print(f"  Parent PPL: {parent_ppl:.2f}")
        print(f"  Variant PPL: mean={np.mean(ppls):.2f}, "
              f"min={np.min(ppls):.2f}, max={np.max(ppls):.2f}")

    # 2. Physicochemical properties
    print("  Computing physicochemical properties...")
    parent_props = compute_all_properties(parent_sequence)
    parent_amphipathicity = helical_wheel_amphipathicity(parent_sequence)

    # 3. Score each variant
    print("  Scoring variants...")
    variant_reports = []
    for i, seq in enumerate(variant_sequences):
        ppl_val = ppls[i] if ppls is not None else None
        report = score_sequence(seq, parent_sequence, ppl_val)
        report["sequence"] = seq
        variant_reports.append(report)

    # 4. Mutation position analysis
    print("  Analyzing mutation positions...")
    mutation_analysis = analyze_mutation_positions(parent_sequence, variant_sequences)

    # 5. AA composition
    composition = aa_composition_analysis(variant_sequences)

    # 6. Aggregate statistics
    overall_scores = [r["scores"]["overall"] for r in variant_reports]
    charge_scores = [r["scores"]["charge"] for r in variant_reports]
    hydro_scores = [r["scores"]["hydrophobicity"] for r in variant_reports]
    amphi_scores = [r["scores"]["amphipathicity"] for r in variant_reports]

    summary = {
        "parent": {
            "sequence": parent_sequence,
            "properties": parent_props,
            "amphipathicity": parent_amphipathicity,
            "esm2_ppl": parent_ppl,
        },
        "n_variants": len(variant_sequences),
        "scores": {
            "overall": {"mean": np.mean(overall_scores), "std": np.std(overall_scores)},
            "charge": {"mean": np.mean(charge_scores), "std": np.std(charge_scores)},
            "hydrophobicity": {"mean": np.mean(hydro_scores), "std": np.std(hydro_scores)},
            "amphipathicity": {"mean": np.mean(amphi_scores), "std": np.std(amphi_scores)},
        },
        "mutation_analysis": mutation_analysis,
        "aa_composition": {
            "kl_divergence": composition["kl_divergence"],
            "hydrophobic_ratio": composition["hydrophobic_ratio"],
            "positive_charge_ratio": composition["positive_charge_ratio"],
        },
        "variant_details": variant_reports,
    }

    if ppls is not None:
        summary["esm2_ppl"] = {
            "parent": parent_ppl,
            "variants_mean": float(np.mean(ppls)),
            "variants_std": float(np.std(ppls)),
            "variants_min": float(np.min(ppls)),
            "variants_max": float(np.max(ppls)),
            "pct_better_than_parent": float(np.mean(np.array(ppls) < parent_ppl) * 100),
        }

    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Overall quality score: {np.mean(overall_scores):.3f} +/- {np.std(overall_scores):.3f}")
    print(f"Charge score: {np.mean(charge_scores):.3f}")
    print(f"Hydrophobicity score: {np.mean(hydro_scores):.3f}")
    print(f"Amphipathicity score: {np.mean(amphi_scores):.3f}")
    if ppls is not None:
        print(f"ESM-2 PPL: {np.mean(ppls):.2f} (parent: {parent_ppl:.2f})")
        print(f"  Variants with lower PPL than parent: "
              f"{np.mean(np.array(ppls) < parent_ppl) * 100:.1f}%")
    print(f"\nMutation distribution:")
    rd = mutation_analysis.get("region_density", {})
    print(f"  N-terminal density: {rd.get('n_terminal', 0):.3f}")
    print(f"  Middle density: {rd.get('middle', 0):.3f}")
    print(f"  C-terminal density: {rd.get('c_terminal', 0):.3f}")
    print(f"AA composition KL-divergence vs typical AMPs: {composition['kl_divergence']:.4f}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Computational validation of AMP variants")
    parser.add_argument("--fasta", required=True, help="FASTA file with parent + variants")
    parser.add_argument("--no-ppl", action="store_true", help="Skip ESM-2 pseudo-perplexity (faster)")
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Parse FASTA
    sequences = {}
    current_name = None
    with open(args.fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_name = line[1:].split()[0]
                sequences[current_name] = ""
            elif current_name:
                sequences[current_name] += line

    names = list(sequences.keys())
    if not names:
        print("Error: No sequences found in FASTA file")
        return

    # First sequence is parent
    parent_name = names[0]
    parent_seq = sequences[parent_name]
    variant_seqs = [sequences[n] for n in names[1:]]

    print(f"Parent: {parent_name} ({parent_seq}, len={len(parent_seq)})")
    print(f"Variants: {len(variant_seqs)}")

    report = validate_variants(
        parent_seq,
        variant_seqs,
        compute_ppl=not args.no_ppl,
        esm_model=args.esm_model,
        device=args.device,
    )

    # Save report
    if args.output is None:
        output_path = PROJECT_ROOT / "results" / "validation_report.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=convert)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
