"""Latent-space Gaussian diffusion module.

Operates on the 128-dim continuous latent vectors from the VAE encoder.
Supports:
- Full denoising (unconditional generation)
- Partial denoising from noised input (variant generation)
- Classifier-free guidance for property conditioning
"""

import torch
import torch.nn as nn

from .noise_schedule import linear_beta_schedule, cosine_beta_schedule, compute_diffusion_params


class DenoiserMLP(nn.Module):
    """MLP-based noise prediction network with time and property conditioning."""

    def __init__(
        self,
        latent_dim: int = 128,
        prop_dim: int = 5,
        hidden_dim: int = 256,
        T: int = 100,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Embedding(T, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prop_embed = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, properties: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise epsilon from noisy latent z_t.

        Args:
            z_t: [B, latent_dim] noisy latent vector at step t.
            t: [B] integer timestep indices.
            properties: [B, prop_dim] property condition vector.

        Returns:
            Predicted noise epsilon [B, latent_dim].
        """
        t_emb = self.time_embed(t)         # [B, hidden_dim]
        p_emb = self.prop_embed(properties)  # [B, hidden_dim]
        x = torch.cat([z_t, t_emb, p_emb], dim=-1)
        return self.net(x)


class LatentDiffusion(nn.Module):
    """Gaussian diffusion in VAE latent space."""

    def __init__(
        self,
        latent_dim: int = 128,
        prop_dim: int = 5,
        T: int = 100,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dim: int = 256,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.cfg_drop_prob = cfg_drop_prob

        # Noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T, beta_start, beta_end)

        params = compute_diffusion_params(betas)
        for name, tensor in params.items():
            self.register_buffer(name, tensor)

        # Denoising network
        self.denoiser = DenoiserMLP(latent_dim, prop_dim, hidden_dim, T)

        # Null property embedding for classifier-free guidance
        self.register_buffer("null_props", torch.zeros(1, prop_dim))

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: add noise to z_0 at timestep t.

        Args:
            z_0: [B, latent_dim] clean latent vectors.
            t: [B] integer timestep.
            noise: Optional pre-sampled noise.

        Returns:
            z_t: [B, latent_dim] noisy latent vectors.
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)       # [B, 1]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # [B, 1]
        return sqrt_alpha * z_0 + sqrt_one_minus * noise

    def predict_noise(self, z_t: torch.Tensor, t: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
        """Predict noise with optional classifier-free guidance dropout during training."""
        if self.training and self.cfg_drop_prob > 0:
            # Randomly drop property conditioning
            mask = torch.rand(z_t.size(0), 1, device=z_t.device) < self.cfg_drop_prob
            properties = torch.where(mask, self.null_props.expand_as(properties), properties)
        return self.denoiser(z_t, t, properties)

    def training_loss(self, z_0: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
        """Compute denoising loss for training.

        Args:
            z_0: [B, latent_dim] clean latent vectors (from VAE encoder).
            properties: [B, prop_dim] property conditions.

        Returns:
            MSE loss between predicted and actual noise.
        """
        B = z_0.size(0)
        t = torch.randint(0, self.T, (B,), device=z_0.device)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        predicted_noise = self.predict_noise(z_t, t, properties)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample_step(
        self, z_t: torch.Tensor, t: int, properties: torch.Tensor, guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """Single reverse diffusion step: z_t -> z_{t-1}.

        Args:
            z_t: [B, latent_dim] current noisy latent.
            t: Current timestep (integer).
            properties: [B, prop_dim] desired properties.
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).

        Returns:
            z_{t-1}: [B, latent_dim] less noisy latent.
        """
        B = z_t.size(0)
        t_batch = torch.full((B,), t, device=z_t.device, dtype=torch.long)

        # Classifier-free guidance
        if guidance_scale != 1.0:
            eps_cond = self.denoiser(z_t, t_batch, properties)
            eps_uncond = self.denoiser(z_t, t_batch, self.null_props.expand(B, -1))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = self.denoiser(z_t, t_batch, properties)

        # Compute z_{t-1}
        beta_t = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        mean = sqrt_recip_alpha * (z_t - beta_t / sqrt_one_minus * eps)

        if t > 0:
            noise = torch.randn_like(z_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int],
        properties: torch.Tensor,
        guidance_scale: float = 2.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Full reverse diffusion: generate latent z from noise.

        Args:
            shape: (B, latent_dim)
            properties: [B, prop_dim] desired properties.
            guidance_scale: CFG scale.
            device: Target device.

        Returns:
            z_0: [B, latent_dim] generated clean latent vectors.
        """
        if device is None:
            device = next(self.parameters()).device
        z_t = torch.randn(shape, device=device)
        properties = properties.to(device)

        for t in reversed(range(self.T)):
            z_t = self.p_sample_step(z_t, t, properties, guidance_scale)
        return z_t

    @torch.no_grad()
    def partial_denoise(
        self,
        z_input: torch.Tensor,
        start_step: int,
        properties: torch.Tensor,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """Partial diffusion for variant generation (mask-and-refine).

        Adds noise to z_input up to start_step, then denoises back.
        Controls how different the variant is from the original.

        Args:
            z_input: [B, latent_dim] input sequence latent.
            start_step: How many noise steps to add (0 = no change, T = fully random).
            properties: [B, prop_dim] desired properties.
            guidance_scale: CFG scale.

        Returns:
            z_variant: [B, latent_dim] variant latent vector.
        """
        B = z_input.size(0)
        start_step = min(start_step, self.T - 1)

        # Forward: add noise up to start_step
        t = torch.full((B,), start_step, device=z_input.device, dtype=torch.long)
        noise = torch.randn_like(z_input)
        z_t = self.q_sample(z_input, t, noise)

        # Reverse: denoise from start_step back to 0
        for step in reversed(range(start_step)):
            z_t = self.p_sample_step(z_t, step, properties, guidance_scale)
        return z_t
