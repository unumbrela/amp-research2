"""Latent space interpolation between two AMP sequences."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from training.dataset import indices_to_sequence, sequence_to_one_hot
from training.utils import load_checkpoint
from generation.variant import sequence_identity


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def interpolate(
    model: ESMDiffVAE,
    seq_a: str,
    seq_b: str,
    n_steps: int = 10,
    properties: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """Generate intermediate sequences between two AMPs via latent interpolation.

    Args:
        model: Trained ESM-DiffVAE model.
        seq_a: First AMP sequence.
        seq_b: Second AMP sequence.
        n_steps: Number of interpolation steps.
        properties: Property conditioning vector.
        device: Target device.

    Returns:
        List of dicts with 'sequence', 'alpha', 'identity_a', 'identity_b'.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    seq_a = seq_a.upper().strip()
    seq_b = seq_b.upper().strip()

    # Encode both sequences
    with torch.no_grad():
        z_a = _encode_sequence(model, seq_a, device)
        z_b = _encode_sequence(model, seq_b, device)

    # Default properties
    if properties is None:
        properties = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.5]], device=device)

    # Interpolate
    results = []
    alphas = torch.linspace(0, 1, n_steps)

    for alpha in alphas:
        z_interp = (1 - alpha) * z_a + alpha * z_b  # [1, latent_dim]

        with torch.no_grad():
            logits, length_logits = model.decode(z_interp, properties)
            pred_len = length_logits.argmax(dim=-1).item() + 1
            seq = indices_to_sequence(logits.argmax(dim=-1)[0])
            seq = seq[:pred_len]

        results.append({
            "sequence": seq,
            "alpha": alpha.item(),
            "identity_a": sequence_identity(seq_a, seq),
            "identity_b": sequence_identity(seq_b, seq),
            "length": len(seq),
        })

    return results


def _encode_sequence(model: ESMDiffVAE, seq: str, device: torch.device) -> torch.Tensor:
    """Encode a single sequence to latent vector."""
    one_hot = sequence_to_one_hot(seq, model.max_len).unsqueeze(0).to(device)
    esm_emb = model.esm([seq], max_len=model.max_len).to(device)
    padding_mask = torch.zeros(1, model.max_len, dtype=torch.bool, device=device)
    padding_mask[:, len(seq):] = True
    mu, _ = model.encoder(one_hot, esm_emb, padding_mask)
    return mu


def main():
    parser = argparse.ArgumentParser(description="Interpolate between two AMPs")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-a", required=True, help="First AMP sequence")
    parser.add_argument("--seq-b", required=True, help="Second AMP sequence")
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = ESMDiffVAE(config).to(device)
    load_checkpoint(model, args.checkpoint, device=args.device)

    print(f"Seq A: {args.seq_a}")
    print(f"Seq B: {args.seq_b}")
    print(f"Steps: {args.n_steps}\n")

    results = interpolate(model, args.seq_a, args.seq_b, args.n_steps, device=device)

    print(f"{'Alpha':>6} {'Sequence':<40} {'Len':>4} {'Id_A':>6} {'Id_B':>6}")
    print("-" * 70)
    for r in results:
        seq_display = r['sequence'][:37] + "..." if len(r['sequence']) > 40 else r['sequence']
        print(f"{r['alpha']:6.2f} {seq_display:<40} {r['length']:4d} {r['identity_a']:5.1%} {r['identity_b']:5.1%}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(f">interp_alpha={r['alpha']:.2f} id_a={r['identity_a']:.3f} id_b={r['identity_b']:.3f}\n")
                f.write(f"{r['sequence']}\n")
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
