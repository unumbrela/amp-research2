"""Visualization utilities for evaluation results."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def plot_aa_composition(
    generated_freq: dict[str, float],
    reference_freq: dict[str, float] | None = None,
    save_path: str | Path | None = None,
):
    """Bar chart of amino acid composition."""
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    gen_vals = [generated_freq.get(aa, 0) for aa in aas]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(aas))
    width = 0.35

    ax.bar(x - width / 2, gen_vals, width, label="Generated", color="#2196F3")
    if reference_freq:
        ref_vals = [reference_freq.get(aa, 0) for aa in aas]
        ax.bar(x + width / 2, ref_vals, width, label="Natural AMPs", color="#FF9800")

    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Frequency")
    ax.set_title("Amino Acid Composition")
    ax.set_xticks(x)
    ax.set_xticklabels(aas)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_property_distributions(
    properties: dict[str, list[float]],
    save_path: str | Path | None = None,
):
    """Histograms of physicochemical properties."""
    n_props = len(properties)
    cols = min(3, n_props)
    rows = (n_props + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_props == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, values) in enumerate(properties.items()):
        ax = axes[i]
        ax.hist(values, bins=30, color="#4CAF50", edgecolor="white", alpha=0.8)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel(name)
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_variant_identity_histogram(
    identities: list[float],
    save_path: str | Path | None = None,
):
    """Histogram of variant-to-parent sequence identity."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(identities, bins=20, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(identities), color="red", linestyle="--", label=f"Mean: {np.mean(identities):.1%}")
    ax.set_xlabel("Sequence Identity to Parent")
    ax.set_ylabel("Count")
    ax.set_title("Variant Sequence Identity Distribution")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_latent_space_tsne(
    z_train: np.ndarray,
    z_generated: np.ndarray,
    labels_train: np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """t-SNE visualization of latent space."""
    from sklearn.manifold import TSNE

    z_all = np.concatenate([z_train, z_generated], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_all)

    n_train = len(z_train)
    z_train_2d = z_2d[:n_train]
    z_gen_2d = z_2d[n_train:]

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels_train is not None:
        scatter = ax.scatter(
            z_train_2d[:, 0], z_train_2d[:, 1],
            c=labels_train, cmap="coolwarm", alpha=0.3, s=10, label="Training"
        )
        plt.colorbar(scatter, ax=ax, label="AMP label")
    else:
        ax.scatter(z_train_2d[:, 0], z_train_2d[:, 1], alpha=0.3, s=10, c="gray", label="Training")

    ax.scatter(z_gen_2d[:, 0], z_gen_2d[:, 1], alpha=0.6, s=20, c="green", marker="*", label="Generated")

    ax.set_title("Latent Space (t-SNE)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_training_curves(
    log_entries: list[dict],
    save_path: str | Path | None = None,
):
    """Plot training and validation loss curves from log entries."""
    train_entries = [e for e in log_entries if e.get("phase") == "train"]
    val_entries = [e for e in log_entries if e.get("phase") == "val"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot([e["epoch"] for e in train_entries], [e["total"] for e in train_entries], label="Train")
    axes[0].plot([e["epoch"] for e in val_entries], [e["total"] for e in val_entries], label="Val")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    # Reconstruction loss
    axes[1].plot([e["epoch"] for e in train_entries], [e["recon"] for e in train_entries], label="Train")
    axes[1].plot([e["epoch"] for e in val_entries], [e["recon"] for e in val_entries], label="Val")
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()

    # Accuracy
    if "accuracy" in train_entries[0]:
        axes[2].plot([e["epoch"] for e in train_entries], [e["accuracy"] for e in train_entries], label="Train")
        axes[2].plot([e["epoch"] for e in val_entries], [e["accuracy"] for e in val_entries], label="Val")
        axes[2].set_title("Reconstruction Accuracy")
        axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig
