"""Training utilities: checkpointing, logging, metrics."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
    extra: dict | None = None,
):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra:
        state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor) -> float:
    """Compute per-residue accuracy (excluding padding)."""
    preds = logits.argmax(dim=-1)  # [B, L]
    correct = (preds == targets) & ~padding_mask
    total = (~padding_mask).sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


class TrainingLogger:
    """Simple JSON-lines logger for training metrics."""

    def __init__(self, log_path: str | Path, append: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not append and self.log_path.exists():
            self.log_path.unlink()
        self.entries = []

    def log(self, epoch: int, metrics: dict, phase: str = "train"):
        entry = {"epoch": epoch, "phase": phase, **metrics}
        self.entries.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load(self) -> list[dict]:
        entries = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    entries.append(json.loads(line.strip()))
        return entries


class RunLogger:
    """Structured run logger for train/val/events/final results."""

    def __init__(self, run_dir: str | Path, append: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self.run_dir / "train_metrics.jsonl"
        self.val_path = self.run_dir / "val_metrics.jsonl"
        self.event_path = self.run_dir / "events.log"
        self.result_path = self.run_dir / "result_summary.json"

        if not append:
            for path in (self.train_path, self.val_path, self.event_path, self.result_path):
                if path.exists():
                    path.unlink()

        self._logger = logging.getLogger(f"esm_diffvae.run.{self.run_dir.as_posix()}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers = []
        handler = logging.FileHandler(self.event_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._logger.addHandler(handler)

    def _write_jsonl(self, path: Path, payload: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_metrics(self, epoch: int, metrics: dict, phase: str):
        payload = {"epoch": int(epoch), "phase": phase, **metrics}
        if phase == "train":
            self._write_jsonl(self.train_path, payload)
        elif phase == "val":
            self._write_jsonl(self.val_path, payload)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def info(self, message: str):
        self._logger.info(message)

    def write_result(self, result: dict):
        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}
