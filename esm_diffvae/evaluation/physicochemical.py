"""Physicochemical property analysis using modlAMP (if available) or built-in calculations."""

import numpy as np


# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino acid molecular weights (Da)
AA_MW = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1, "Q": 146.2, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

# pKa values for charged groups
PKA_VALUES = {"K": 10.5, "R": 12.5, "H": 6.0, "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}


def compute_hydrophobicity(sequence: str) -> float:
    """Mean Kyte-Doolittle hydrophobicity."""
    values = [KD_SCALE.get(aa, 0.0) for aa in sequence.upper()]
    return np.mean(values) if values else 0.0


def compute_molecular_weight(sequence: str) -> float:
    """Approximate molecular weight in Daltons."""
    seq = sequence.upper()
    mw = sum(AA_MW.get(aa, 0.0) for aa in seq)
    # Subtract water for each peptide bond
    mw -= (len(seq) - 1) * 18.02
    return mw


def compute_charge(sequence: str, ph: float = 7.0) -> float:
    """Net charge at given pH using Henderson-Hasselbalch."""
    charge = 0.0
    seq = sequence.upper()

    # N-terminus (pKa ~9.69)
    charge += 1.0 / (1.0 + 10 ** (ph - 9.69))
    # C-terminus (pKa ~2.34)
    charge -= 1.0 / (1.0 + 10 ** (2.34 - ph))

    for aa in seq:
        if aa in ("K", "R"):
            pka = PKA_VALUES[aa]
            charge += 1.0 / (1.0 + 10 ** (ph - pka))
        elif aa == "H":
            charge += 1.0 / (1.0 + 10 ** (ph - 6.0))
        elif aa in ("D", "E"):
            pka = PKA_VALUES[aa]
            charge -= 1.0 / (1.0 + 10 ** (pka - ph))
        elif aa == "C":
            charge -= 1.0 / (1.0 + 10 ** (8.3 - ph))
        elif aa == "Y":
            charge -= 1.0 / (1.0 + 10 ** (10.1 - ph))

    return charge


def compute_isoelectric_point(sequence: str) -> float:
    """Estimate isoelectric point by bisection method."""
    low, high = 0.0, 14.0
    for _ in range(100):
        mid = (low + high) / 2
        charge = compute_charge(sequence, mid)
        if charge > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def compute_aromaticity(sequence: str) -> float:
    """Fraction of aromatic residues (F, W, Y)."""
    aromatic = set("FWY")
    count = sum(1 for aa in sequence.upper() if aa in aromatic)
    return count / len(sequence) if sequence else 0.0


def compute_all_properties(sequence: str) -> dict[str, float]:
    """Compute all physicochemical properties for a single sequence."""
    return {
        "length": len(sequence),
        "molecular_weight": compute_molecular_weight(sequence),
        "hydrophobicity": compute_hydrophobicity(sequence),
        "charge_ph7": compute_charge(sequence, 7.0),
        "isoelectric_point": compute_isoelectric_point(sequence),
        "aromaticity": compute_aromaticity(sequence),
    }


def batch_compute_properties(sequences: list[str]) -> dict[str, list[float]]:
    """Compute properties for a list of sequences."""
    results = {}
    for seq in sequences:
        props = compute_all_properties(seq)
        for key, val in props.items():
            results.setdefault(key, []).append(val)
    return results


def property_summary(sequences: list[str]) -> dict[str, dict[str, float]]:
    """Summary statistics of physicochemical properties."""
    props = batch_compute_properties(sequences)
    summary = {}
    for key, values in props.items():
        arr = np.array(values)
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }
    return summary


def try_modlamp_analysis(sequences: list[str]) -> dict | None:
    """Try to use modlAMP for more comprehensive analysis."""
    try:
        from modlamp.descriptors import GlobalDescriptor
        desc = GlobalDescriptor(sequences)
        desc.calculate_all()
        return {
            "modlamp_descriptors": desc.descriptor.tolist(),
            "modlamp_names": desc.featurenames,
        }
    except ImportError:
        return None
    except Exception as e:
        print(f"modlAMP analysis failed: {e}")
        return None
