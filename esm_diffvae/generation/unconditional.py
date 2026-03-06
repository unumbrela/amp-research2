"""Unconditional AMP generation: generate novel sequences from scratch."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from training.dataset import indices_to_sequence
from training.utils import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_unconditional_params(
    config: dict,
    n_samples: int | None = None,
    guidance_scale: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict:
    """Resolve unconditional generation params from config with optional overrides."""
    gen_cfg = config.get("generation", {})
    uncond_cfg = gen_cfg.get("unconditional", {})
    diff_cfg = config.get("diffusion", {})

    return {
        "n_samples": int(n_samples if n_samples is not None else uncond_cfg.get("n_samples", gen_cfg.get("n_samples", 100))),
        "guidance_scale": float(
            guidance_scale
            if guidance_scale is not None
            else uncond_cfg.get("guidance_scale", diff_cfg.get("guidance_scale", 1.2))
        ),
        "temperature": float(temperature if temperature is not None else uncond_cfg.get("temperature", gen_cfg.get("temperature", 1.0))),
        "top_p": float(top_p if top_p is not None else uncond_cfg.get("top_p", gen_cfg.get("top_p", 0.9))),
    }


def generate_unconditional(
    model: ESMDiffVAE,
    n_samples: int = 100,
    properties: torch.Tensor | None = None,
    guidance_scale: float = 2.0,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> list[str]:
    """Generate novel AMP sequences from scratch.

    Args:
        model: Trained ESM-DiffVAE model.
        n_samples: Number of sequences to generate.
        properties: [n_samples, prop_dim] desired properties.
            Default: is_amp=1, mic=0, non-toxic, non-hemolytic.
        guidance_scale: Classifier-free guidance strength.
        temperature: Sampling temperature for decoder.
        top_p: Nucleus sampling threshold.
        device: Target device.

    Returns:
        List of generated amino acid sequences.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Default: generate AMPs with desired properties
    if properties is None:
        properties = torch.zeros(n_samples, 5, device=device)
        properties[:, 0] = 1.0   # is_amp = True
        properties[:, 1] = 0.0   # mic_value = low (good)
        properties[:, 2] = 0.0   # is_toxic = False
        properties[:, 3] = 0.0   # is_hemolytic = False
        properties[:, 4] = 0.5   # length_norm = medium length (~25 AA)

    # Step 1: Sample from diffusion prior
    z = model.diffusion.sample(
        shape=(n_samples, model.latent_dim),
        properties=properties,
        guidance_scale=guidance_scale,
        device=device,
    )

    # Step 2: Predict length
    with torch.no_grad():
        _, length_logits = model.decode(z, properties)
        pred_lengths = length_logits.argmax(dim=-1) + 1  # 1-indexed

    # Step 3: Decode to sequences
    sequences = []
    with torch.no_grad():
        logits, _ = model.decode(z, properties)  # [n_samples, max_len, vocab]

        if temperature != 1.0:
            logits = logits / temperature

        if top_p < 1.0:
            probs = nucleus_sampling(logits, top_p)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
        else:
            sampled = logits.argmax(dim=-1)

        for i in range(n_samples):
            seq = indices_to_sequence(sampled[i])
            # Trim to predicted length
            pred_len = pred_lengths[i].item()
            seq = seq[:pred_len]
            if len(seq) >= 5:  # minimum viable length
                sequences.append(seq)

    return sequences


def nucleus_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Apply nucleus (top-p) sampling to logits.

    Args:
        logits: [B, L, V] raw logits.
        top_p: Cumulative probability threshold.

    Returns:
        Filtered probability distributions [B, L, V].
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_mask] = 0.0

    # Scatter back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_indices, sorted_probs)

    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return filtered_probs


def main():
    parser = argparse.ArgumentParser(description="Generate AMPs unconditionally")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = ESMDiffVAE(config).to(device)
    load_checkpoint(model, args.checkpoint, device=args.device)
    print(f"Loaded model from {args.checkpoint}")

    gen_params = resolve_unconditional_params(
        config,
        n_samples=args.n_samples,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(
        f"\nGenerating {gen_params['n_samples']} AMP sequences "
        f"(guidance={gen_params['guidance_scale']}, temp={gen_params['temperature']}, top_p={gen_params['top_p']})..."
    )
    sequences = generate_unconditional(
        model, gen_params["n_samples"],
        guidance_scale=gen_params["guidance_scale"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        device=device,
    )

    print(f"Generated {len(sequences)} valid sequences")
    for i, seq in enumerate(sequences[:10]):
        print(f"  {i+1:3d}. {seq} (len={len(seq)})")

    if len(sequences) > 10:
        print(f"  ... and {len(sequences) - 10} more")

    # Save
    if args.output is None:
        output_path = PROJECT_ROOT / config["paths"]["results_dir"] / "unconditional_generated.fasta"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">generated_{i+1} len={len(seq)}\n{seq}\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
