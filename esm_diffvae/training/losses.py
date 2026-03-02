"""Loss functions for ESM-DiffVAE training.

Combines:
- Reconstruction loss (cross-entropy)
- KL divergence with cyclical annealing
- Supervised contrastive loss for latent structure
- Multi-property prediction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-residue cross-entropy reconstruction loss.

    Args:
        logits: [B, L, vocab_size] predicted logits.
        targets: [B, L] target amino acid indices.
        padding_mask: [B, L] True for padded positions (excluded from loss).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
    """
    B, L, V = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, V), targets.view(-1),
        reduction="none", label_smoothing=label_smoothing,
    )
    loss = loss.view(B, L)

    if padding_mask is not None:
        loss = loss.masked_fill(padding_mask, 0.0)
        # Average over non-padded positions
        n_tokens = (~padding_mask).float().sum()
        return loss.sum() / n_tokens.clamp(min=1)
    return loss.mean()


def kl_divergence_loss(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """KL divergence: KL(q(z|x) || N(0,I)) with optional free bits.

    Free bits: per-dimension KL is clamped to at least `free_bits` nats,
    preventing posterior collapse by ensuring the latent code carries
    at least some information.
    """
    # Per-dimension KL: [B, latent_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits > 0:
        # Clamp per-dimension KL to at least free_bits
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    return kl_per_dim.mean()


def cyclical_beta(
    epoch: int, warmup_epochs: int, beta_max: float = 1.0,
    n_cycles: int = 1, ratio_ramp: float = 0.5,
) -> float:
    """Cyclical KL annealing schedule.

    When n_cycles=1: linear ramp from 0 to beta_max over warmup_epochs (original).
    When n_cycles>1: repeating ramp-and-hold cycles for better latent utilization.

    Args:
        epoch: Current epoch.
        warmup_epochs: Total epochs for annealing (all cycles).
        beta_max: Maximum beta value.
        n_cycles: Number of annealing cycles.
        ratio_ramp: Fraction of each cycle spent ramping (rest is hold).
    """
    if warmup_epochs <= 0:
        return beta_max
    if epoch >= warmup_epochs:
        return beta_max

    if n_cycles <= 1:
        # Original monotonic ramp
        return beta_max * epoch / warmup_epochs

    # Cyclical: determine position within current cycle
    cycle_len = warmup_epochs / n_cycles
    cycle_pos = (epoch % cycle_len) / cycle_len  # 0.0 to 1.0 within cycle

    if cycle_pos < ratio_ramp:
        # Ramping phase
        return beta_max * cycle_pos / ratio_ramp
    else:
        # Hold phase
        return beta_max


def supervised_contrastive_loss(
    z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon) in latent space.

    Pulls together embeddings with the same label, pushes apart different labels.

    Args:
        z: [B, latent_dim] latent vectors.
        labels: [B] integer class labels (e.g., AMP=1, non-AMP=0).
        temperature: Scaling temperature (0.5 is more stable than 0.1).
    """
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # Normalize embeddings (use float32 for stability)
    z_float = z.float()
    z_norm = F.normalize(z_float, dim=-1)

    # Compute pairwise similarity
    sim_matrix = torch.matmul(z_norm, z_norm.T) / temperature  # [B, B]

    # Positive pair mask: same label = 1, diagonal = 0
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)

    n_positives = pos_mask.sum(dim=1)
    valid = n_positives > 0
    if not valid.any():
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # For numerical stability: subtract max per row
    logits_max = sim_matrix.max(dim=1, keepdim=True).values.detach()
    logits = sim_matrix - logits_max

    # Self-mask: exclude diagonal from denominator
    self_mask = 1.0 - torch.eye(B, device=z.device)

    # Denominator: sum of exp over all non-self pairs
    exp_logits = torch.exp(logits) * self_mask
    log_denominator = torch.log(exp_logits.sum(dim=1).clamp(min=1e-8))

    # Numerator: mean log-prob over positive pairs
    # Use pos_mask to select positive pair logits (no -inf involved)
    log_prob = logits - log_denominator.unsqueeze(1)
    # Mask out self and non-positive pairs, then average over positives
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / n_positives.clamp(min=1)

    loss = -pos_log_prob[valid].mean()
    return loss


def property_prediction_loss(
    prop_preds: dict[str, torch.Tensor],
    prop_targets: torch.Tensor,
    prop_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-property prediction loss.

    Args:
        prop_preds: Dict with keys 'is_amp', 'mic_value', 'is_toxic', 'is_hemolytic'.
        prop_targets: [B, 5] target property values.
        prop_mask: [B, 5] True where the property value is known.
    """
    losses = []

    # is_amp (binary, index 0)
    if prop_mask[:, 0].any():
        m = prop_mask[:, 0]
        loss = F.binary_cross_entropy_with_logits(
            prop_preds["is_amp"][m], prop_targets[m, 0]
        )
        losses.append(loss)

    # mic_value (continuous, index 1)
    if prop_mask[:, 1].any():
        m = prop_mask[:, 1]
        loss = F.mse_loss(prop_preds["mic_value"][m], prop_targets[m, 1])
        losses.append(loss)

    # is_toxic (binary, index 2)
    if prop_mask[:, 2].any():
        m = prop_mask[:, 2]
        loss = F.binary_cross_entropy_with_logits(
            prop_preds["is_toxic"][m], prop_targets[m, 2]
        )
        losses.append(loss)

    # is_hemolytic (binary, index 3)
    if prop_mask[:, 3].any():
        m = prop_mask[:, 3]
        loss = F.binary_cross_entropy_with_logits(
            prop_preds["is_hemolytic"][m], prop_targets[m, 3]
        )
        losses.append(loss)

    if losses:
        return sum(losses) / len(losses)
    return torch.tensor(0.0, device=prop_targets.device)


def length_prediction_loss(
    length_logits: torch.Tensor, seq_lengths: torch.Tensor
) -> torch.Tensor:
    """Length prediction cross-entropy loss.

    Args:
        length_logits: [B, max_len] predicted length logits.
        seq_lengths: [B] true sequence lengths (1-indexed).
    """
    # seq_lengths are 1-indexed, convert to 0-indexed class
    targets = (seq_lengths - 1).clamp(min=0, max=length_logits.size(1) - 1).long()
    return F.cross_entropy(length_logits, targets)


class ESMDiffVAELoss(nn.Module):
    """Combined loss for ESM-DiffVAE VAE training."""

    def __init__(
        self,
        beta_max: float = 1.0,
        beta_warmup_epochs: int = 50,
        lambda_contrastive: float = 0.1,
        lambda_property: float = 0.5,
        lambda_length: float = 0.1,
        label_smoothing: float = 0.0,
        free_bits: float = 0.0,
        kl_n_cycles: int = 1,
        kl_ratio_ramp: float = 0.5,
    ):
        super().__init__()
        self.beta_max = beta_max
        self.beta_warmup_epochs = beta_warmup_epochs
        self.lambda_contrastive = lambda_contrastive
        self.lambda_property = lambda_property
        self.lambda_length = lambda_length
        self.label_smoothing = label_smoothing
        self.free_bits = free_bits
        self.kl_n_cycles = kl_n_cycles
        self.kl_ratio_ramp = kl_ratio_ramp

    def forward(
        self,
        model_output: dict,
        targets: torch.Tensor,
        properties: torch.Tensor,
        prop_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss.

        Returns:
            total_loss: Scalar loss for backprop.
            loss_dict: Dictionary of individual loss components for logging.
        """
        # Reconstruction
        l_recon = reconstruction_loss(
            model_output["logits"], targets, padding_mask, self.label_smoothing
        )

        # KL divergence with cyclical annealing and free bits
        l_kl = kl_divergence_loss(
            model_output["mu"], model_output["logvar"], self.free_bits
        )
        beta = cyclical_beta(
            epoch, self.beta_warmup_epochs, self.beta_max,
            self.kl_n_cycles, self.kl_ratio_ramp,
        )

        # Supervised contrastive loss
        amp_labels = properties[:, 0].long()  # binary AMP labels
        l_contrastive = supervised_contrastive_loss(model_output["z"], amp_labels)

        # Property prediction loss
        l_property = property_prediction_loss(
            model_output["prop_preds"], properties, prop_mask
        )

        # Length prediction loss
        l_length = length_prediction_loss(model_output["length_logits"], seq_lengths)

        # Total
        total = (
            l_recon
            + beta * l_kl
            + self.lambda_contrastive * l_contrastive
            + self.lambda_property * l_property
            + self.lambda_length * l_length
        )

        loss_dict = {
            "total": total.item(),
            "recon": l_recon.item(),
            "kl": l_kl.item(),
            "beta": beta,
            "contrastive": l_contrastive.item(),
            "property": l_property.item(),
            "length": l_length.item(),
        }

        return total, loss_dict
