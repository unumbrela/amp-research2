"""Phase 3 Training: Latent diffusion module (after VAE is trained).

Freezes the VAE encoder, encodes all training data to latent vectors,
then trains the diffusion denoising network on these latent representations.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.esm_diffvae import ESMDiffVAE
from training.dataset import create_dataloader
from training.utils import save_checkpoint, load_checkpoint, TrainingLogger, EarlyStopping


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def encode_dataset(model, dataloader, device):
    """Encode entire dataset to latent vectors using the frozen VAE encoder."""
    model.eval()
    all_z = []
    all_props = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding dataset"):
            one_hot = batch["one_hot"].to(device)
            esm_emb = batch["esm_emb"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            properties = batch["properties"].to(device)
            sequences = batch["sequence"]

            if esm_emb.sum() != 0:
                mu, logvar = model.encoder(one_hot, esm_emb, padding_mask)
            else:
                esm_emb_live = model.esm(sequences, max_len=one_hot.size(1))
                esm_emb_live = esm_emb_live.to(device)
                mu, logvar = model.encoder(one_hot, esm_emb_live, padding_mask)

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
    print(f"=== Latent Diffusion Training ===")
    print(f"Device: {device}")
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
    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    train_loader = create_dataloader(
        data_dir / "processed" / "train.csv",
        data_dir / "embeddings" / "train_esm.pt",
        max_len=config["vae"]["max_seq_len"],
        batch_size=config["train_diffusion"]["batch_size"],
        shuffle=False,
    )

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
        val_csv = data_dir / "processed" / "val.csv"
        val_emb = data_dir / "embeddings" / "val_esm.pt"
        if val_csv.exists():
            print("Encoding validation data to latent space...")
            val_data_loader = create_dataloader(
                val_csv,
                val_emb,
                max_len=config["vae"]["max_seq_len"],
                batch_size=config["train_diffusion"]["batch_size"],
                shuffle=False,
            )
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
        val_loss = None
        if val_enabled and val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, device, grad_clip, train=False)
            logger.log(epoch, {"loss": val_loss}, "val")

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

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch, score,
                          ckpt_dir / f"diffusion_epoch_{epoch}.pt")

        if early_stopping is not None and val_loss is not None and early_stopping.step(val_loss):
            print(f"Early stopping diffusion at epoch {epoch} (best val loss={best_loss:.6f})")
            break

    # Save final full model
    final_score = val_loss if val_loss is not None else train_loss
    save_checkpoint(model, optimizer, epoch, final_score,
                  ckpt_dir / "esm_diffvae_full.pt")
    print(f"\nDiffusion training complete. Best loss: {best_loss:.6f}")
    print(f"Full model saved to {ckpt_dir}/esm_diffvae_full.pt")


if __name__ == "__main__":
    main()
