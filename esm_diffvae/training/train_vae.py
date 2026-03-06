"""Phase 2 Training: VAE with PLM embeddings, contrastive loss, and property prediction.

v8: Multi-PLM backend + BLOSUM62 hybrid encoding + non-autoregressive Conv decoder.
"""

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from models.plm_extractor import BACKEND_REGISTRY
from training.dataset import create_dataloader
from training.losses import ESMDiffVAELoss
from training.utils import (
    save_checkpoint, load_checkpoint, compute_accuracy,
    TrainingLogger, RunLogger, EarlyStopping, ModelEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        n = sum(1 for _ in reader)
    return max(0, n - 1)


def _resolve_plm_setup(config: dict) -> tuple[str, str, int, str]:
    """Resolve backend/model/dim and embedding filename suffix from config."""
    plm_cfg = config.get("plm", config.get("esm", {}))
    backend = plm_cfg.get("backend", "esm2")
    if backend not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown PLM backend '{backend}'. Choose from {list(BACKEND_REGISTRY)}")

    model_name = plm_cfg.get("model_name")
    if model_name is None:
        model_name = list(BACKEND_REGISTRY[backend])[0]
    if model_name not in BACKEND_REGISTRY[backend]:
        raise ValueError(
            f"Unknown model '{model_name}' for backend '{backend}'. "
            f"Choose from {list(BACKEND_REGISTRY[backend])}"
        )

    expected_dim = BACKEND_REGISTRY[backend][model_name]
    # Keep historical esm suffix for backward compatibility.
    emb_suffix = "esm" if backend == "esm2" else backend
    return backend, model_name, expected_dim, emb_suffix


def _check_embedding_dim(loader, expected_dim: int, emb_path: Path):
    dataset = loader.dataset
    emb = getattr(dataset, "esm_embeddings", None)
    if emb is None:
        return
    loaded_dim = int(emb.shape[-1])
    if loaded_dim != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch for {emb_path}: loaded {loaded_dim}, "
            f"expected {expected_dim}. Check config.plm.* and embedding file suffix."
        )


def _fmt_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.4f}"


def compute_teacher_forcing_ratio(epoch: int, config: dict) -> float:
    """Compute scheduled sampling teacher forcing ratio for current epoch."""
    tf_cfg = config["train_vae"]
    start = tf_cfg.get("teacher_forcing_start", 1.0)
    end = tf_cfg.get("teacher_forcing_end", 0.5)
    warmup = tf_cfg.get("teacher_forcing_warmup", 100)
    if warmup <= 0 or epoch >= warmup:
        return end
    return start - (start - end) * epoch / warmup


def _forward_batch(model, batch, device, teacher_forcing_ratio=1.0):
    """Run forward pass with pre-computed PLM embeddings + hybrid AA encoding."""
    properties = batch["properties"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    plm_emb = batch["esm_emb"].to(device)  # key kept as "esm_emb" for dataset compat
    target_indices = batch["target_indices"].to(device)
    esm_noise_std = float(getattr(_forward_batch, "esm_noise_std", 0.0))

    # AA encoding: BLOSUM62 + learned embedding (replaces one-hot)
    aa_features = model.aa_encoding(target_indices)  # [B, L, aa_dim]

    # PLM embeddings: use pre-computed or compute on-the-fly
    has_plm = bool((plm_emb.abs().sum() > 0).item())
    if has_plm:
        if model.training and esm_noise_std > 0:
            plm_emb = plm_emb + torch.randn_like(plm_emb) * esm_noise_std
    else:
        sequences = batch["sequence"]
        plm_emb = model.plm(sequences, max_len=target_indices.size(1)).to(device)
        if model.training and esm_noise_std > 0:
            plm_emb = plm_emb + torch.randn_like(plm_emb) * esm_noise_std

    mu, logvar = model.encoder(aa_features, plm_emb, padding_mask)
    z = model.reparameterize(mu, logvar)
    # Non-autoregressive decoder: teacher_forcing_ratio is ignored
    logits, length_logits = model.decode(z, properties)
    prop_preds = model.prop_heads(z)
    return {
        "logits": logits,
        "length_logits": length_logits,
        "mu": mu,
        "logvar": logvar,
        "z": z,
        "prop_preds": prop_preds,
    }


def _compute_property_head_metrics(output, properties, prop_mask):
    metrics = {}

    def safe_value(loss):
        return loss.item() if torch.isfinite(loss) else float("nan")

    if prop_mask[:, 0].any():
        m = prop_mask[:, 0]
        metrics["prop_is_amp_loss"] = safe_value(
            F.binary_cross_entropy_with_logits(output["prop_preds"]["is_amp"][m], properties[m, 0])
        )
    if prop_mask[:, 1].any():
        m = prop_mask[:, 1]
        metrics["prop_mic_loss"] = safe_value(F.mse_loss(output["prop_preds"]["mic_value"][m], properties[m, 1]))
    if prop_mask[:, 2].any():
        m = prop_mask[:, 2]
        metrics["prop_is_toxic_loss"] = safe_value(
            F.binary_cross_entropy_with_logits(output["prop_preds"]["is_toxic"][m], properties[m, 2])
        )
    if prop_mask[:, 3].any():
        m = prop_mask[:, 3]
        metrics["prop_is_hemolytic_loss"] = safe_value(
            F.binary_cross_entropy_with_logits(output["prop_preds"]["is_hemolytic"][m], properties[m, 3])
        )
    return metrics


def _compute_extra_metrics(output, targets, padding_mask, seq_lengths, properties, prop_mask):
    recon = F.cross_entropy(
        output["logits"].view(-1, output["logits"].size(-1)),
        targets.view(-1),
        reduction="none",
    ).view_as(targets)
    recon = recon.masked_fill(padding_mask, 0.0)
    n_tokens = (~padding_mask).sum().clamp(min=1)
    token_ce = (recon.sum() / n_tokens).item()
    token_ppl = math.exp(min(token_ce, 20.0))

    pred_lengths = output["length_logits"].argmax(dim=-1) + 1
    length_mae = (pred_lengths - seq_lengths).abs().float().mean().item()

    metrics = {
        "token_ce": token_ce,
        "token_ppl": token_ppl,
        "nonpad_top1": ((output["logits"].argmax(dim=-1) == targets) & ~padding_mask).sum().item()
        / max((~padding_mask).sum().item(), 1),
        "length_mae": length_mae,
    }
    metrics.update(_compute_property_head_metrics(output, properties, prop_mask))
    return metrics


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, device, config, ema=None):
    model.train()
    total_loss_dict = {}
    total_acc = 0.0
    n_batches = 0
    tf_ratio = compute_teacher_forcing_ratio(epoch, config)

    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False):
        targets = batch["target_indices"].to(device)
        properties = batch["properties"].to(device)
        prop_mask = batch["prop_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        seq_lengths = batch["seq_len"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=config["train_vae"]["fp16"]):
            output = _forward_batch(model, batch, device, teacher_forcing_ratio=tf_ratio)
            loss, loss_dict = criterion(
                output, targets, properties, prop_mask, padding_mask, seq_lengths, epoch
            )

        if not math.isfinite(loss.item()):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["train_vae"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        acc = compute_accuracy(output["logits"].detach(), targets, padding_mask)
        total_acc += acc
        for k, v in loss_dict.items():
            total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
        for k, v in _compute_extra_metrics(
            output, targets, padding_mask, seq_lengths, properties, prop_mask
        ).items():
            total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
        n_batches += 1

    if n_batches == 0:
        return {"total": float("nan"), "accuracy": 0.0, "tf_ratio": tf_ratio}
    avg_metrics = {k: v / n_batches for k, v in total_loss_dict.items()}
    avg_metrics["accuracy"] = total_acc / n_batches
    avg_metrics["tf_ratio"] = tf_ratio
    return avg_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, epoch, device, config):
    model.eval()
    total_loss_dict = {}
    total_acc = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=False):
        targets = batch["target_indices"].to(device)
        properties = batch["properties"].to(device)
        prop_mask = batch["prop_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        seq_lengths = batch["seq_len"].to(device)

        with torch.amp.autocast("cuda", enabled=config["train_vae"]["fp16"]):
            # Validate with free running (no teacher forcing) to measure true generalization
            output = _forward_batch(model, batch, device, teacher_forcing_ratio=0.0)
            loss, loss_dict = criterion(
                output, targets, properties, prop_mask, padding_mask, seq_lengths, epoch
            )

        if math.isfinite(loss.item()):
            acc = compute_accuracy(output["logits"], targets, padding_mask)
            total_acc += acc
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
            for k, v in _compute_extra_metrics(
                output, targets, padding_mask, seq_lengths, properties, prop_mask
            ).items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
            n_batches += 1

    if n_batches == 0:
        return {"total": float("inf"), "accuracy": 0.0}
    avg_metrics = {k: v / n_batches for k, v in total_loss_dict.items()}
    avg_metrics["accuracy"] = total_acc / n_batches
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ESM-DiffVAE (VAE phase)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--append-log", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    plm_cfg = config.get("plm", config.get("esm", {}))
    backend, model_name, expected_plm_dim, emb_suffix = _resolve_plm_setup(config)
    enc_cfg = config.get("encoding", {})
    vae_cfg = config["vae"]
    print(f"=== ESM-DiffVAE v8 VAE Training ===")
    print(f"Device: {device}")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"PLM backend: {backend} ({model_name})")
    print(f"Encoding: {enc_cfg.get('type', 'hybrid')}")
    print(f"Architecture: BiGRU encoder (h={vae_cfg['hidden_dim']}) + "
          f"Transformer decoder (h={vae_cfg.get('decoder_hidden_dim', vae_cfg['hidden_dim'])}, "
          f"layers={vae_cfg.get('n_decoder_layers', 3)}, "
          f"heads={vae_cfg.get('decoder_n_heads', 4)})")
    print(
        f"latent_dim={vae_cfg['latent_dim']} max_seq_len={vae_cfg['max_seq_len']} "
        f"batch_size={config['train_vae']['batch_size']}"
    )

    # Data
    paths_cfg = config["paths"]
    data_dir = PROJECT_ROOT / paths_cfg["data_dir"]
    processed_dir = data_dir / paths_cfg.get("processed_dir", "processed")
    embeddings_dir = data_dir / paths_cfg.get("embeddings_dir", "embeddings")
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing CSV files. Expected:\n"
            f"  - {train_csv}\n"
            f"  - {val_csv}"
        )
    train_rows = _count_csv_rows(train_csv)
    val_rows = _count_csv_rows(val_csv)
    train_emb_path = embeddings_dir / f"train_{emb_suffix}.pt"
    val_emb_path = embeddings_dir / f"val_{emb_suffix}.pt"
    if not train_emb_path.exists() or not val_emb_path.exists():
        raise FileNotFoundError(
            f"Missing PLM embedding files for backend '{backend}'. Expected:\n"
            f"  - {train_emb_path}\n"
            f"  - {val_emb_path}\n"
            f"Run: python data/compute_embeddings.py --backend {backend} --model {model_name}"
        )
    print(f"Using CSV data directory: {processed_dir.resolve()}")
    print(f"Dataset sizes: train={train_rows}, val={val_rows}")
    print(
        f"Using embeddings (dim={expected_plm_dim}):\n"
        f"  - {train_emb_path.resolve()}\n"
        f"  - {val_emb_path.resolve()}"
    )

    train_loader = create_dataloader(
        train_csv,
        train_emb_path,
        max_len=config["vae"]["max_seq_len"],
        batch_size=config["train_vae"]["batch_size"],
        shuffle=True,
        plm_embedding_dim=expected_plm_dim,
    )
    val_loader = create_dataloader(
        val_csv,
        val_emb_path,
        max_len=config["vae"]["max_seq_len"],
        batch_size=config["train_vae"]["batch_size"],
        shuffle=False,
        plm_embedding_dim=expected_plm_dim,
    )
    _check_embedding_dim(train_loader, expected_plm_dim, train_emb_path)
    _check_embedding_dim(val_loader, expected_plm_dim, val_emb_path)

    # Model
    model = ESMDiffVAE(config).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer with cosine LR schedule
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        train_params,
        lr=config["train_vae"]["lr"],
        weight_decay=config["train_vae"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train_vae"]["epochs"], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["train_vae"]["fp16"])
    _forward_batch.esm_noise_std = config["train_vae"].get("esm_noise_std", 0.0)
    ema = ModelEMA(model, decay=config["train_vae"].get("ema_decay", 0.999))

    # Loss
    criterion = ESMDiffVAELoss(
        beta_max=config["train_vae"]["beta_max"],
        beta_warmup_epochs=config["train_vae"]["beta_warmup_epochs"],
        lambda_contrastive=config["train_vae"]["lambda_contrastive"],
        lambda_property=config["train_vae"]["lambda_property"],
        lambda_length=config["train_vae"].get("lambda_length", 0.1),
        label_smoothing=config["train_vae"].get("label_smoothing", 0.1),
        free_bits=config["train_vae"].get("free_bits", 0.0),
        kl_n_cycles=config["train_vae"].get("kl_n_cycles", 1),
        kl_ratio_ramp=config["train_vae"].get("kl_ratio_ramp", 0.5),
    )

    # Logger and early stopping
    ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    logger = TrainingLogger(ckpt_dir / "vae_training_log.jsonl", append=args.append_log or bool(args.resume))
    run_name = f"vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_logger = RunLogger(ckpt_dir / "logs" / run_name, append=False)
    print(f"Run logs will be saved to: {run_logger.run_dir.resolve()}")
    run_logger.info(
        f"run_start device={device} backend={backend} model={model_name} "
        f"max_seq_len={config['vae']['max_seq_len']} batch_size={config['train_vae']['batch_size']} "
        f"lr={config['train_vae']['lr']} resume={bool(args.resume)}"
    )
    early_stopping = EarlyStopping(patience=config["train_vae"]["early_stopping_patience"])

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, optimizer, device=args.device)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
        run_logger.info(f"resume_from checkpoint={args.resume} start_epoch={start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    best_val_recon = float("inf")
    best_val_acc = -1.0
    for epoch in range(start_epoch, config["train_vae"]["epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, device, config, ema=ema
        )
        ema.store(model)
        ema.copy_to(model)
        val_metrics = validate(model, val_loader, criterion, epoch, device, config)
        ema.restore(model)
        scheduler.step()

        logger.log(epoch, train_metrics, "train")
        logger.log(epoch, val_metrics, "val")
        run_logger.log_metrics(epoch, train_metrics, "train")
        run_logger.log_metrics(epoch, val_metrics, "val")

        lr = optimizer.param_groups[0]["lr"]
        tf_ratio = train_metrics.get("tf_ratio", 1.0)
        print(f"Epoch {epoch:3d} | "
              f"Train loss={train_metrics['total']:.4f} recon={train_metrics.get('recon', 0):.4f} "
              f"kl={train_metrics.get('kl', 0):.4f} acc={train_metrics['accuracy']:.3f} "
              f"ppl={train_metrics.get('token_ppl', float('nan')):.2f} "
              f"len_mae={train_metrics.get('length_mae', float('nan')):.2f} | "
              f"Val loss={val_metrics['total']:.4f} recon={val_metrics.get('recon', 0):.4f} "
              f"acc={val_metrics['accuracy']:.3f} ppl={val_metrics.get('token_ppl', float('nan')):.2f} "
              f"len_mae={val_metrics.get('length_mae', float('nan')):.2f} | "
              f"beta={train_metrics.get('beta', 0):.4f} tf={tf_ratio:.2f} lr={lr:.2e}")
        print(
            "  Prop losses (val): "
            f"is_amp={_fmt_metric(val_metrics.get('prop_is_amp_loss'))}, "
            f"mic={_fmt_metric(val_metrics.get('prop_mic_loss'))}, "
            f"is_toxic={_fmt_metric(val_metrics.get('prop_is_toxic_loss'))}, "
            f"is_hemolytic={_fmt_metric(val_metrics.get('prop_is_hemolytic_loss'))}"
        )

        # Save best model
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            save_checkpoint(model, optimizer, epoch, val_metrics["total"],
                          ckpt_dir / "vae_best.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
            run_logger.info(f"best_val_loss epoch={epoch} value={best_val_loss:.6f} checkpoint=vae_best.pt")
        if val_metrics["recon"] < best_val_recon:
            best_val_recon = val_metrics["recon"]
            save_checkpoint(model, optimizer, epoch, val_metrics["recon"], ckpt_dir / "vae_best_recon.pt")
            print(f"  -> Saved best reconstruction model (val_recon={best_val_recon:.4f})")
            run_logger.info(f"best_val_recon epoch={epoch} value={best_val_recon:.6f} checkpoint=vae_best_recon.pt")
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, optimizer, epoch, val_metrics["accuracy"], ckpt_dir / "vae_best_acc.pt")
            print(f"  -> Saved best accuracy model (val_acc={best_val_acc:.3f})")
            run_logger.info(f"best_val_acc epoch={epoch} value={best_val_acc:.6f} checkpoint=vae_best_acc.pt")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics["total"],
                          ckpt_dir / f"vae_epoch_{epoch}.pt")

        run_logger.write_result({
            "status": "running",
            "run_name": run_name,
            "last_epoch": int(epoch),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": float(best_val_loss),
            "best_val_recon": float(best_val_recon),
            "best_val_acc": float(best_val_acc),
        })

        # Early stopping
        if early_stopping.step(val_metrics["total"]):
            print(f"Early stopping at epoch {epoch}")
            run_logger.info(f"early_stopping epoch={epoch} patience={config['train_vae']['early_stopping_patience']}")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")
    print(f"Run logs saved to: {run_logger.run_dir.resolve()}")
    print(f"Run summary: {(run_logger.run_dir / "result_summary.json").resolve()}")

    result_summary = {
        "run_name": run_name,
        "epochs_configured": int(config["train_vae"]["epochs"]),
        "best_val_loss": float(best_val_loss),
        "best_val_recon": float(best_val_recon),
        "best_val_acc": float(best_val_acc),
        "status": "completed",
        "checkpoint_dir": str(ckpt_dir),
        "best_checkpoints": {
            "val_loss": str(ckpt_dir / "vae_best.pt"),
            "val_recon": str(ckpt_dir / "vae_best_recon.pt"),
            "val_acc": str(ckpt_dir / "vae_best_acc.pt"),
        },
    }
    run_logger.write_result(result_summary)
    run_logger.info("run_complete")


if __name__ == "__main__":
    main()
