"""Phase 3 Training: Latent diffusion module (after VAE is trained).

Freezes the VAE encoder, encodes all training data to latent vectors,
then trains the diffusion denoising network on these latent representations.
"""

import argparse
import csv
from datetime import datetime
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from models.plm_extractor import BACKEND_REGISTRY
from training.dataset import create_dataloader
from training.utils import save_checkpoint, load_checkpoint, TrainingLogger, RunLogger, EarlyStopping


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


def encode_dataset(model, dataloader, device):
    """Encode entire dataset to latent vectors using the frozen VAE encoder."""
    model.eval()
    all_z = []
    all_props = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding dataset"):
            target_indices = batch["target_indices"].to(device)
            plm_emb = batch["esm_emb"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            properties = batch["properties"].to(device)
            sequences = batch["sequence"]

            # AA encoding via model's hybrid encoder
            aa_features = model.aa_encoding(target_indices)

            if plm_emb.sum() != 0:
                mu, logvar = model.encoder(aa_features, plm_emb, padding_mask)
            else:
                plm_emb_live = model.plm(sequences, max_len=target_indices.size(1))
                plm_emb_live = plm_emb_live.to(device)
                mu, logvar = model.encoder(aa_features, plm_emb_live, padding_mask)

            # Use mu (not sampled z) for diffusion training
            all_z.append(mu.cpu())
            all_props.append(properties.cpu())

    return torch.cat(all_z, dim=0), torch.cat(all_props, dim=0)


class LatentDataset(torch.utils.data.Dataset):
    """Simple dataset of pre-encoded latent vectors."""

    def __init__(self, z: torch.Tensor, properties: torch.Tensor):
        self.z = z
        self.properties = properties

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.properties[idx]


def run_epoch(model, dataloader, optimizer, device, grad_clip: float, train: bool = True):
    if train:
        model.diffusion.train()
    else:
        model.diffusion.eval()

    total_loss = 0.0
    n_batches = 0
    for z_batch, prop_batch in tqdm(dataloader, desc="Train" if train else "Val", leave=False):
        z_batch = z_batch.to(device)
        prop_batch = prop_batch.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            loss = model.diffusion.training_loss(z_batch, prop_batch)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.diffusion.parameters(), grad_clip
            )
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train latent diffusion module")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--vae-checkpoint", default=None, help="Path to trained VAE checkpoint")
    parser.add_argument("--use-val-latent", action="store_true", default=False,
                        help="Encode validation split and track val denoising loss")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Override diffusion early stopping patience")
    parser.add_argument("--append-log", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    backend, model_name, expected_plm_dim, emb_suffix = _resolve_plm_setup(config)
    print(f"=== Latent Diffusion Training ===")
    print(f"Device: {device}")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"PLM backend: {backend} ({model_name})")
    vae_ckpt = args.vae_checkpoint
    if vae_ckpt is None:
        preferred = [
            PROJECT_ROOT / "checkpoints" / "vae_best_recon.pt",
            PROJECT_ROOT / "checkpoints" / "vae_best.pt",
        ]
        for candidate in preferred:
            if candidate.exists():
                vae_ckpt = str(candidate)
                break
    if vae_ckpt is None:
        raise FileNotFoundError(
            "No VAE checkpoint found. Provide --vae-checkpoint or train VAE first."
        )

    # Load trained VAE
    model = ESMDiffVAE(config).to(device)
    load_checkpoint(model, vae_ckpt, device=args.device)
    print(f"Loaded VAE from {vae_ckpt}")

    # Freeze everything except diffusion
    for name, param in model.named_parameters():
        if "diffusion" not in name:
            param.requires_grad = False

    diff_params = sum(p.numel() for p in model.diffusion.parameters())
    print(f"Diffusion parameters: {diff_params:,}")

    # Encode training data
    paths_cfg = config["paths"]
    data_dir = PROJECT_ROOT / paths_cfg["data_dir"]
    processed_dir = data_dir / paths_cfg.get("processed_dir", "processed")
    embeddings_dir = data_dir / paths_cfg.get("embeddings_dir", "embeddings")
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    train_emb_path = embeddings_dir / f"train_{emb_suffix}.pt"
    val_emb_path = embeddings_dir / f"val_{emb_suffix}.pt"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing training CSV: {train_csv}")
    if not train_emb_path.exists():
        raise FileNotFoundError(
            f"Missing training embeddings for backend '{backend}': {train_emb_path}\n"
            f"Run: python data/compute_embeddings.py --backend {backend} --model {model_name}"
        )
    print(f"Using CSV data directory: {processed_dir.resolve()}")
    print(f"Dataset sizes: train={_count_csv_rows(train_csv)}"
          + (f", val={_count_csv_rows(val_csv)}" if val_csv.exists() else ", val=missing"))
    print(f"Using training embedding (dim={expected_plm_dim}): {train_emb_path.resolve()}")
    train_loader = create_dataloader(
        train_csv,
        train_emb_path,
        max_len=config["vae"]["max_seq_len"],
        batch_size=config["train_diffusion"]["batch_size"],
        shuffle=False,
        plm_embedding_dim=expected_plm_dim,
    )
    _check_embedding_dim(train_loader, expected_plm_dim, train_emb_path)

    print("\nEncoding training data to latent space...")
    train_z, train_props = encode_dataset(model, train_loader, device)
    print(f"Encoded {len(train_z)} samples, latent shape: {train_z.shape}")

    # Create latent dataloader
    latent_dataset = LatentDataset(train_z, train_props)
    latent_loader = torch.utils.data.DataLoader(
        latent_dataset,
        batch_size=config["train_diffusion"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = None
    val_enabled = (
        args.use_val_latent
        or config["train_diffusion"].get("validation", {}).get("enabled", False)
    )
    if val_enabled:
        val_emb = val_emb_path
        if val_csv.exists():
            if not val_emb.exists():
                raise FileNotFoundError(
                    f"Validation CSV exists but embedding file is missing: {val_emb}"
                )
            print("Encoding validation data to latent space...")
            val_data_loader = create_dataloader(
                val_csv,
                val_emb,
                max_len=config["vae"]["max_seq_len"],
                batch_size=config["train_diffusion"]["batch_size"],
                shuffle=False,
                plm_embedding_dim=expected_plm_dim,
            )
            _check_embedding_dim(val_data_loader, expected_plm_dim, val_emb)
            val_z, val_props = encode_dataset(model, val_data_loader, device)
            val_dataset = LatentDataset(val_z, val_props)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config["train_diffusion"]["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            print(f"Encoded {len(val_z)} validation samples")
        else:
            print("Validation latent encoding skipped: val.csv not found")
            val_enabled = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.diffusion.parameters(),
        lr=config["train_diffusion"]["lr"],
        weight_decay=config["train_diffusion"]["weight_decay"],
    )

    # Training
    ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    logger = TrainingLogger(ckpt_dir / "diffusion_training_log.jsonl", append=args.append_log)
    run_name = f"diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_logger = RunLogger(ckpt_dir / "logs" / run_name, append=False)
    print(f"Run logs will be saved to: {run_logger.run_dir.resolve()}")
    run_logger.info(
        f"run_start device={device} backend={backend} model={model_name} "
        f"epochs={config['train_diffusion']['epochs']} batch_size={config['train_diffusion']['batch_size']} "
        f"lr={config['train_diffusion']['lr']} val_enabled={val_enabled}"
    )
    best_loss = float("inf")
    patience = (
        args.early_stop_patience
        if args.early_stop_patience is not None
        else config["train_diffusion"].get("validation", {}).get("early_stop_patience", 30)
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5) if val_enabled else None
    grad_clip = config["train_diffusion"]["grad_clip"]

    for epoch in range(config["train_diffusion"]["epochs"]):
        train_loss = run_epoch(model, latent_loader, optimizer, device, grad_clip, train=True)
        logger.log(epoch, {"loss": train_loss}, "train")
        run_logger.log_metrics(epoch, {"loss": train_loss}, "train")
        val_loss = None
        if val_enabled and val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, device, grad_clip, train=False)
            logger.log(epoch, {"loss": val_loss}, "val")
            run_logger.log_metrics(epoch, {"loss": val_loss}, "val")

        if epoch % 10 == 0:
            if val_loss is None:
                print(f"Epoch {epoch:4d} | Train loss: {train_loss:.6f}")
            else:
                print(f"Epoch {epoch:4d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        score = val_loss if val_loss is not None else train_loss
        if score < best_loss:
            best_loss = score
            save_checkpoint(model, optimizer, epoch, score,
                          ckpt_dir / "diffusion_best.pt")
            run_logger.info(f"best_score epoch={epoch} value={best_loss:.6f} checkpoint=diffusion_best.pt")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch, score,
                          ckpt_dir / f"diffusion_epoch_{epoch}.pt")
            run_logger.info(f"periodic_checkpoint epoch={epoch} score={score:.6f}")

        if early_stopping is not None and val_loss is not None and early_stopping.step(val_loss):
            print(f"Early stopping diffusion at epoch {epoch} (best val loss={best_loss:.6f})")
            run_logger.info(f"early_stopping epoch={epoch} best={best_loss:.6f}")
            break

    # Save final full model
    final_score = val_loss if val_loss is not None else train_loss
    save_checkpoint(model, optimizer, epoch, final_score,
                  ckpt_dir / "esm_diffvae_full.pt")
    result_summary = {
        "run_name": run_name,
        "status": "completed",
        "best_loss": float(best_loss),
        "final_score": float(final_score),
        "best_checkpoint": str((ckpt_dir / "diffusion_best.pt").resolve()),
        "full_model_checkpoint": str((ckpt_dir / "esm_diffvae_full.pt").resolve()),
    }
    run_logger.write_result(result_summary)
    run_logger.info("run_complete")
    print(f"\nDiffusion training complete. Best loss: {best_loss:.6f}")
    print(f"Full model saved to {ckpt_dir}/esm_diffvae_full.pt")
    print(f"Run logs saved to: {run_logger.run_dir.resolve()}")
    print(f"Run summary: {(run_logger.run_dir / 'result_summary.json').resolve()}")


if __name__ == "__main__":
    main()
