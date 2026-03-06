"""Full evaluation pipeline for ESM-DiffVAE.

Evaluates both unconditional generation and variant generation,
computes all metrics, generates plots, and saves results.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from training.utils import load_checkpoint
from generation.unconditional import generate_unconditional, resolve_unconditional_params
from generation.variant import generate_variants
from evaluation.metrics import (
    full_evaluation, compute_aa_composition, NATURAL_AMP_AA_FREQ,
)
from evaluation.physicochemical import property_summary, batch_compute_properties
from evaluation.visualization import (
    plot_aa_composition, plot_property_distributions,
    plot_variant_identity_histogram,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Well-known AMPs for variant generation testing
BENCHMARK_AMPS = {
    "magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "defensin-1": "ACYCRIPACIAGERRYGTCIYQGRLWAFCC",
    "indolicidin": "ILPWKWPWWPWRR",
}


def evaluate_unconditional(model, config, device, results_dir, training_seqs=None):
    """Evaluate unconditional generation."""
    print("\n=== Unconditional Generation Evaluation ===")

    eval_cfg = config.get("generation", {}).get("evaluation", {})
    uncond_params = resolve_unconditional_params(config)
    n_samples = int(eval_cfg.get("uncond_n_samples", 500))
    print(f"Generating {n_samples} sequences...")
    sequences = generate_unconditional(
        model, n_samples=n_samples,
        guidance_scale=uncond_params["guidance_scale"],
        temperature=uncond_params["temperature"],
        top_p=uncond_params["top_p"],
        device=device,
    )
    print(f"  Got {len(sequences)} valid sequences")

    # Metrics
    metrics = full_evaluation(sequences, training_seqs)
    print(f"\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Physicochemical properties
    phys_summary = property_summary(sequences)
    print(f"\nPhysicochemical properties:")
    for prop, stats in phys_summary.items():
        print(f"  {prop}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    # Plots
    gen_freq = compute_aa_composition(sequences)
    plot_aa_composition(gen_freq, NATURAL_AMP_AA_FREQ, results_dir / "aa_composition.png")
    print(f"  Saved AA composition plot")

    props = batch_compute_properties(sequences)
    plot_property_distributions(props, results_dir / "property_distributions.png")
    print(f"  Saved property distribution plot")

    # Save sequences
    with open(results_dir / "unconditional_sequences.fasta", "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">generated_{i+1}\n{seq}\n")

    return {"metrics": metrics, "physicochemical": phys_summary}


def evaluate_variants(model, config, device, results_dir):
    """Evaluate variant generation on benchmark AMPs."""
    print("\n=== Variant Generation Evaluation ===")

    all_variant_results = {}

    eval_cfg = config.get("generation", {}).get("evaluation", {})
    strengths = eval_cfg.get("variant_strengths", [0.1, 0.3, 0.5])
    n_variant_samples = int(eval_cfg.get("variant_n_samples", 50))
    hist_strength = float(eval_cfg.get("variant_hist_strength", 0.3))
    hist_n_samples = int(eval_cfg.get("variant_hist_n_samples", 100))
    guidance = config.get("generation", {}).get("variant", {}).get(
        "latent", {}
    ).get("guidance_scale", config.get("diffusion", {}).get("guidance_scale", 1.2))

    for name, parent_seq in BENCHMARK_AMPS.items():
        if len(parent_seq) > config["vae"]["max_seq_len"]:
            print(f"\n  Skipping {name}: too long ({len(parent_seq)} > {config['vae']['max_seq_len']})")
            continue

        print(f"\n  {name}: {parent_seq} (len={len(parent_seq)})")

        for strength in strengths:
            variants = generate_variants(
                model, parent_seq,
                n_variants=n_variant_samples,
                variation_strength=strength,
                guidance_scale=guidance,
                device=device,
            )

            variant_seqs = [v["sequence"] for v in variants]
            metrics = full_evaluation(variant_seqs, variants=variants)

            print(f"    strength={strength}: {len(variants)} variants, "
                  f"identity={metrics.get('mean_identity', 0):.1%}, "
                  f"diversity={metrics.get('mean_diversity', 0):.3f}, "
                  f"uniqueness={metrics.get('uniqueness_rate', 0):.1%}")

            key = f"{name}_s{strength}"
            all_variant_results[key] = {
                "parent": parent_seq,
                "strength": strength,
                "n_variants": len(variants),
                "metrics": metrics,
            }

            # Save variants
            with open(results_dir / f"variants_{key}.fasta", "w") as f:
                f.write(f">parent_{name}\n{parent_seq}\n")
                for i, v in enumerate(variants):
                    f.write(f">variant_{i+1} identity={v['identity']:.3f}\n{v['sequence']}\n")

        # Identity histogram for moderate strength
        moderate_variants = generate_variants(
            model, parent_seq, n_variants=hist_n_samples, variation_strength=hist_strength, device=device,
        )
        identities = [v["identity"] for v in moderate_variants]
        plot_variant_identity_histogram(identities, results_dir / f"identity_hist_{name}.png")

    return all_variant_results


def main():
    parser = argparse.ArgumentParser(description="Full ESM-DiffVAE evaluation")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"=== ESM-DiffVAE Full Evaluation ===")
    print(f"Device: {device}")

    # Load model
    model = ESMDiffVAE(config).to(device)
    load_checkpoint(model, args.checkpoint, device=args.device)
    print(f"Loaded model from {args.checkpoint}")

    # Results directory
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = PROJECT_ROOT / config["paths"]["results_dir"] / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load training sequences for novelty check
    paths_cfg = config["paths"]
    data_dir = PROJECT_ROOT / paths_cfg["data_dir"]
    processed_dir = data_dir / paths_cfg.get("processed_dir", "processed")
    train_csv = processed_dir / "train.csv"
    training_seqs = None
    print(f"Novelty reference CSV: {train_csv.resolve()}")
    if train_csv.exists():
        training_seqs = pd.read_csv(train_csv)["sequence"].tolist()
        print(f"Loaded {len(training_seqs)} training sequences for novelty check")

    # Run evaluations
    uncond_results = evaluate_unconditional(model, config, device, results_dir, training_seqs)
    variant_results = evaluate_variants(model, config, device, results_dir)

    # Save full results
    all_results = {
        "unconditional": uncond_results,
        "variants": variant_results,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import numpy as np
    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
