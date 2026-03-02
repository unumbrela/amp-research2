"""Evaluation metrics for generated AMP sequences.

Includes: diversity, novelty, composition analysis, and variant-specific metrics.
"""

import itertools
from collections import Counter

import numpy as np


# Standard amino acid frequencies in natural AMPs (approximate from APD3)
NATURAL_AMP_AA_FREQ = {
    "G": 0.089, "A": 0.087, "L": 0.084, "K": 0.081, "I": 0.060,
    "V": 0.055, "S": 0.051, "R": 0.050, "F": 0.045, "P": 0.042,
    "T": 0.040, "C": 0.038, "N": 0.036, "W": 0.033, "D": 0.032,
    "E": 0.030, "Q": 0.028, "H": 0.027, "Y": 0.026, "M": 0.017,
}


def compute_aa_composition(sequences: list[str]) -> dict[str, float]:
    """Compute amino acid frequency distribution."""
    total = 0
    counts = Counter()
    for seq in sequences:
        for aa in seq.upper():
            counts[aa] += 1
            total += 1
    return {aa: counts.get(aa, 0) / max(total, 1) for aa in "ACDEFGHIKLMNPQRSTVWY"}


def composition_kl_divergence(sequences: list[str], reference_freq: dict | None = None) -> float:
    """KL divergence between generated AA composition and natural AMP composition."""
    if reference_freq is None:
        reference_freq = NATURAL_AMP_AA_FREQ
    gen_freq = compute_aa_composition(sequences)
    kl = 0.0
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        p = gen_freq.get(aa, 1e-8)
        q = reference_freq.get(aa, 1e-8)
        if p > 0:
            kl += p * np.log(p / q)
    return kl


def pairwise_diversity(sequences: list[str], max_pairs: int = 5000) -> dict[str, float]:
    """Compute pairwise sequence diversity metrics.

    Returns mean, std of pairwise edit distances (normalized).
    """
    from generation.variant import edit_distance

    n = len(sequences)
    if n < 2:
        return {"mean_diversity": 0.0, "std_diversity": 0.0}

    pairs = list(itertools.combinations(range(n), 2))
    if len(pairs) > max_pairs:
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    distances = []
    for i, j in pairs:
        d = edit_distance(sequences[i], sequences[j])
        max_len = max(len(sequences[i]), len(sequences[j]))
        distances.append(d / max_len if max_len > 0 else 0.0)

    return {
        "mean_diversity": float(np.mean(distances)),
        "std_diversity": float(np.std(distances)),
    }


def compute_novelty(generated: list[str], training_set: list[str]) -> dict[str, float]:
    """Compute what fraction of generated sequences are novel (not in training set)."""
    training_set_upper = set(s.upper() for s in training_set)
    n_novel = sum(1 for s in generated if s.upper() not in training_set_upper)
    return {
        "novelty_rate": n_novel / len(generated) if generated else 0.0,
        "n_novel": n_novel,
        "n_total": len(generated),
    }


def compute_uniqueness(sequences: list[str]) -> dict[str, float]:
    """Fraction of generated sequences that are unique."""
    unique = set(s.upper() for s in sequences)
    return {
        "uniqueness_rate": len(unique) / len(sequences) if sequences else 0.0,
        "n_unique": len(unique),
        "n_total": len(sequences),
    }


def length_statistics(sequences: list[str]) -> dict[str, float]:
    """Basic length statistics."""
    lengths = [len(s) for s in sequences]
    return {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "median_length": float(np.median(lengths)),
    }


def charge_distribution(sequences: list[str]) -> dict[str, float]:
    """Estimate net charge at pH 7 (simplified)."""
    pos_aa = set("RK")   # positively charged
    neg_aa = set("DE")   # negatively charged

    charges = []
    for seq in sequences:
        charge = sum(1 for aa in seq if aa in pos_aa) - sum(1 for aa in seq if aa in neg_aa)
        charges.append(charge)

    return {
        "mean_charge": float(np.mean(charges)),
        "std_charge": float(np.std(charges)),
        "frac_positive": float(np.mean([c > 0 for c in charges])),
    }


def hydrophobicity_ratio(sequences: list[str]) -> dict[str, float]:
    """Fraction of hydrophobic residues."""
    hydrophobic = set("AILMFWV")
    ratios = []
    for seq in sequences:
        ratio = sum(1 for aa in seq if aa in hydrophobic) / len(seq) if seq else 0
        ratios.append(ratio)
    return {
        "mean_hydrophobic_ratio": float(np.mean(ratios)),
        "std_hydrophobic_ratio": float(np.std(ratios)),
    }


# --- Variant-specific metrics ---

def variant_identity_distribution(variants: list[dict]) -> dict[str, float]:
    """Statistics of sequence identity between variants and parent."""
    identities = [v["identity"] for v in variants]
    return {
        "mean_identity": float(np.mean(identities)),
        "std_identity": float(np.std(identities)),
        "min_identity": float(np.min(identities)),
        "max_identity": float(np.max(identities)),
    }


def variant_edit_distribution(variants: list[dict]) -> dict[str, float]:
    """Statistics of edit distance between variants and parent."""
    dists = [v["edit_distance"] for v in variants]
    return {
        "mean_edit_distance": float(np.mean(dists)),
        "std_edit_distance": float(np.std(dists)),
        "min_edit_distance": int(np.min(dists)),
        "max_edit_distance": int(np.max(dists)),
    }


def full_evaluation(
    generated_sequences: list[str],
    training_sequences: list[str] | None = None,
    variants: list[dict] | None = None,
) -> dict:
    """Run all evaluation metrics.

    Args:
        generated_sequences: List of generated AA strings.
        training_sequences: Optional training set for novelty computation.
        variants: Optional list of variant dicts (from variant generation).

    Returns:
        Dictionary with all metrics.
    """
    results = {}

    # Basic stats
    results["count"] = len(generated_sequences)
    results.update(length_statistics(generated_sequences))
    results.update(compute_uniqueness(generated_sequences))

    # Composition
    results["composition_kl"] = composition_kl_divergence(generated_sequences)
    results.update(charge_distribution(generated_sequences))
    results.update(hydrophobicity_ratio(generated_sequences))

    # Diversity
    results.update(pairwise_diversity(generated_sequences))

    # Novelty
    if training_sequences:
        results.update(compute_novelty(generated_sequences, training_sequences))

    # Variant-specific
    if variants:
        results.update(variant_identity_distribution(variants))
        results.update(variant_edit_distribution(variants))

    return results
