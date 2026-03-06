"""Phase 1B Training: RL fine-tuning with discriminator (REINFORCE).

Loads a pre-trained VAE checkpoint (Phase 1A), freezes the encoder,
and fine-tunes the decoder with adversarial training using a BiGRU
discriminator and REINFORCE policy gradient.
"""

import argparse
import csv
from datetime import datetime
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from models.discriminator import SequenceDiscriminator
from models.plm_extractor import BACKEND_REGISTRY
from training.dataset import create_dataloader
from training.losses import reconstruction_loss
from training.utils import save_checkpoint, load_checkpoint, TrainingLogger, RunLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        n = sum(1 for _ in reader)
    return max(0, n - 1)


def _resolve_plm_setup(config: dict):
    plm_cfg = config.get("plm", config.get("esm", {}))
    backend = plm_cfg.get("backend", "esm2")
    model_name = plm_cfg.get("model_name", list(BACKEND_REGISTRY[backend])[0])
    expected_dim = BACKEND_REGISTRY[backend][model_name]
    emb_suffix = "esm" if backend == "esm2" else backend
    return backend, model_name, expected_dim, emb_suffix


def _encode_batch(model, batch, device):
    """Encode a batch to get z, properties, targets, masks."""
    properties = batch["properties"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    plm_emb = batch["esm_emb"].to(device)
    target_indices = batch["target_indices"].to(device)
    seq_lengths = batch["seq_len"].to(device)

    aa_features = model.aa_encoding(target_indices)

    has_plm = bool((plm_emb.abs().sum() > 0).item())
    if not has_plm:
        sequences = batch["sequence"]
        plm_emb = model.plm(sequences, max_len=target_indices.size(1)).to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(aa_features, plm_emb, padding_mask)
        z = model.reparameterize(mu, logvar)

    return z, properties, target_indices, padding_mask, seq_lengths


def train_discriminator_step(
    model, discriminator, disc_optimizer, z, properties, targets, padding_mask, scaler, fp16,
):
    """Train discriminator to distinguish real from generated sequences."""
    discriminator.train()
    disc_optimizer.zero_grad()

    B = z.size(0)
    device = z.device

    with torch.amp.autocast("cuda", enabled=fp16):
        # Generate sequences from decoder
        with torch.no_grad():
            logits, _ = model.decode(z, properties)
            generated = logits.argmax(dim=-1)  # [B, L]

        # Score real and generated
        real_scores = discriminator(targets, padding_mask)
        fake_scores = discriminator(generated, padding_mask)

        # BCE loss
        real_labels = torch.ones(B, device=device)
        fake_labels = torch.zeros(B, device=device)
        loss = (
            F.binary_cross_entropy_with_logits(real_scores, real_labels)
            + F.binary_cross_entropy_with_logits(fake_scores, fake_labels)
        ) / 2

    scaler.scale(loss).backward()
    scaler.unscale_(disc_optimizer)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    scaler.step(disc_optimizer)
    scaler.update()

    # Compute discriminator accuracy
    with torch.no_grad():
        real_acc = (real_scores > 0).float().mean().item()
        fake_acc = (fake_scores <= 0).float().mean().item()

    return {
        "disc_loss": loss.item(),
        "disc_real_acc": real_acc,
        "disc_fake_acc": fake_acc,
        "disc_acc": (real_acc + fake_acc) / 2,
    }


def train_generator_step(
    model, discriminator, gen_optimizer, z, properties, targets, padding_mask,
    scaler, fp16, temperature, rl_weight, running_reward,
):
    """Train decoder with REINFORCE + MLE hybrid loss."""
    model.decoder.train()
    gen_optimizer.zero_grad()

    with torch.amp.autocast("cuda", enabled=fp16):
        logits, _ = model.decode(z, properties)  # [B, L, V]
        B, L, V = logits.shape

        # MLE reconstruction loss
        mle_loss = reconstruction_loss(logits, targets, padding_mask, label_smoothing=0.02)

        # REINFORCE: sample from decoder, get discriminator reward
        probs = F.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, L)

        with torch.no_grad():
            reward = torch.sigmoid(discriminator(sampled, padding_mask))  # [B]
            baseline = running_reward
            advantage = reward - baseline

        log_probs = F.log_softmax(logits, dim=-1)
        sampled_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [B, L]

        # Mask padding
        if padding_mask is not None:
            sampled_log_probs = sampled_log_probs.masked_fill(padding_mask, 0.0)

        rl_loss = -(advantage.unsqueeze(1) * sampled_log_probs).sum(dim=1).mean()

        total_loss = (1 - rl_weight) * mle_loss + rl_weight * rl_loss

    scaler.scale(total_loss).backward()
    scaler.unscale_(gen_optimizer)
    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
    scaler.step(gen_optimizer)
    scaler.update()

    # Update running reward
    new_running_reward = 0.9 * running_reward + 0.1 * reward.mean().item()

    return {
        "gen_total_loss": total_loss.item(),
        "mle_loss": mle_loss.item(),
        "rl_loss": rl_loss.item(),
        "mean_reward": reward.mean().item(),
        "running_reward": new_running_reward,
    }, new_running_reward


def main():
    parser = argparse.ArgumentParser(description="Train ESM-DiffVAE (RL fine-tuning phase)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--vae-checkpoint", default=None,
                        help="Path to Phase 1A VAE checkpoint (default: checkpoints/vae_best.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    rl_cfg = config["train_vae_rl"]
    vae_cfg = config["vae"]

    backend, model_name, expected_plm_dim, emb_suffix = _resolve_plm_setup(config)
    print(f"=== ESM-DiffVAE v8 RL Fine-tuning (Phase 1B) ===")
    print(f"Device: {device}")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"PLM backend: {backend} ({model_name})")

    # Data
    paths_cfg = config["paths"]
    data_dir = PROJECT_ROOT / paths_cfg["data_dir"]
    processed_dir = data_dir / paths_cfg.get("processed_dir", "processed")
    embeddings_dir = data_dir / paths_cfg.get("embeddings_dir", "embeddings")
    train_csv = processed_dir / "train.csv"
    train_emb_path = embeddings_dir / f"train_{emb_suffix}.pt"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing training CSV: {train_csv}")
    if not train_emb_path.exists():
        raise FileNotFoundError(
            f"Missing training embedding for backend '{backend}': {train_emb_path}\n"
            f"Run: python data/compute_embeddings.py --backend {backend} --model {model_name}"
        )
    print(f"Using CSV data directory: {processed_dir.resolve()}")
    print(f"Dataset size: train={_count_csv_rows(train_csv)}")
    print(f"Using training embedding (dim={expected_plm_dim}): {train_emb_path.resolve()}")
    train_loader = create_dataloader(
        train_csv,
        train_emb_path,
        max_len=vae_cfg["max_seq_len"],
        batch_size=config["train_vae"]["batch_size"],
        shuffle=True,
        plm_embedding_dim=expected_plm_dim,
    )

    # Load pre-trained VAE
    model = ESMDiffVAE(config).to(device)
    ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    vae_ckpt = args.vae_checkpoint or str(ckpt_dir / "vae_best.pt")
    print(f"Loading VAE checkpoint: {vae_ckpt}")
    load_checkpoint(model, vae_ckpt, device=args.device)

    # Freeze encoder + PLM + aa_encoding + property heads
    for param in model.plm.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.aa_encoding.parameters():
        param.requires_grad = False
    for param in model.prop_heads.parameters():
        param.requires_grad = False
    # Only decoder is trainable
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Trainable decoder parameters: {decoder_params:,}")

    # Discriminator
    discriminator = SequenceDiscriminator(
        vocab_size=vae_cfg["aa_vocab_size"],
        embed_dim=rl_cfg["disc_embed_dim"],
        hidden_dim=rl_cfg["disc_hidden_dim"],
    ).to(device)
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {disc_params:,}")

    # Optimizers
    gen_optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=rl_cfg["lr"],
    )
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=rl_cfg["disc_lr"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=rl_cfg.get("fp16", True))

    # Logger
    logger = TrainingLogger(ckpt_dir / "vae_rl_training_log.jsonl", append=False)
    run_name = f"vae_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_logger = RunLogger(ckpt_dir / "logs" / run_name, append=False)
    print(f"Run logs will be saved to: {run_logger.run_dir.resolve()}")
    run_logger.info(
        f"run_start device={device} backend={backend} model={model_name} "
        f"epochs={rl_cfg['epochs']} batch_size={config['train_vae']['batch_size']} "
        f"gen_lr={rl_cfg['lr']} disc_lr={rl_cfg['disc_lr']}"
    )

    running_reward = 0.5  # initial baseline
    disc_steps = rl_cfg["disc_steps_per_gen_step"]
    temperature = rl_cfg["temperature"]
    rl_weight = rl_cfg["rl_weight"]

    for epoch in range(rl_cfg["epochs"]):
        epoch_metrics = {}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"RL Epoch {epoch}", leave=False):
            z, properties, targets, padding_mask, seq_lengths = _encode_batch(
                model, batch, device
            )

            # Discriminator steps
            for _ in range(disc_steps):
                disc_metrics = train_discriminator_step(
                    model, discriminator, disc_optimizer,
                    z, properties, targets, padding_mask,
                    scaler, rl_cfg.get("fp16", True),
                )

            # Generator step
            gen_metrics, running_reward = train_generator_step(
                model, discriminator, gen_optimizer,
                z, properties, targets, padding_mask,
                scaler, rl_cfg.get("fp16", True),
                temperature, rl_weight, running_reward,
            )

            metrics = {**disc_metrics, **gen_metrics}
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            n_batches += 1

        if n_batches == 0:
            continue

        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        avg_metrics["running_reward"] = running_reward
        logger.log(epoch, avg_metrics, "train_rl")
        run_logger.log_metrics(epoch, avg_metrics, "train")

        print(
            f"RL Epoch {epoch:3d} | "
            f"gen_loss={avg_metrics['gen_total_loss']:.4f} "
            f"mle={avg_metrics['mle_loss']:.4f} rl={avg_metrics['rl_loss']:.4f} | "
            f"disc_loss={avg_metrics['disc_loss']:.4f} "
            f"disc_acc={avg_metrics['disc_acc']:.3f} | "
            f"reward={avg_metrics['mean_reward']:.3f} "
            f"baseline={running_reward:.3f}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, gen_optimizer, epoch, avg_metrics["gen_total_loss"],
                ckpt_dir / f"vae_rl_epoch_{epoch}.pt",
            )
            run_logger.info(
                f"periodic_checkpoint epoch={epoch} gen_total_loss={avg_metrics['gen_total_loss']:.6f} "
                f"checkpoint=vae_rl_epoch_{epoch}.pt"
            )

    # Save final model
    final_gen_loss = float(avg_metrics.get("gen_total_loss", 0)) if 'avg_metrics' in locals() else 0.0
    save_checkpoint(
        model, gen_optimizer, rl_cfg["epochs"] - 1,
        final_gen_loss,
        ckpt_dir / "vae_rl_final.pt",
    )
    run_logger.write_result({
        "run_name": run_name,
        "status": "completed",
        "epochs_configured": int(rl_cfg["epochs"]),
        "final_gen_total_loss": float(final_gen_loss),
        "final_checkpoint": str((ckpt_dir / "vae_rl_final.pt").resolve()),
    })
    run_logger.info("run_complete")
    print(f"\nRL fine-tuning complete. Model saved to {ckpt_dir}/vae_rl_final.pt")
    print(f"Run logs saved to: {run_logger.run_dir.resolve()}")
    print(f"Run summary: {(run_logger.run_dir / 'result_summary.json').resolve()}")


if __name__ == "__main__":
    main()
