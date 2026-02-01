"""
Multiview Cosmos Model Wrapper for video finetuning.

This module provides a wrapper around the Cosmos Predict 2.5 model that handles:
- Multiview video encoding (multiple camera views concatenated along temporal dimension)
- Per-view learnable embeddings (7 channels concatenated in channel dimension)
- Per-view RoPE construction
- LoRA-based efficient finetuning
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from peft import LoraConfig, get_peft_model, PeftModel


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def logit_normal_sample(size, mu: float = 0.0, sigma: float = 1.0, device='cpu', dtype=None) -> Tensor:
    """Sample from logit-normal distribution for flow time τ_v."""
    z = torch.randn(size, device=device, dtype=dtype) * sigma + mu
    return torch.sigmoid(z)


class ViewEmbedding(nn.Module):
    """Learnable per-view embedding that gets concatenated to latent channels.
    
    Following Cosmos 2.5 multiview: "we concatenate in the latent channel dimension
    a compact per-view learnt embedding (of size 7)".
    
    Args:
        num_views: Number of camera views
        embed_dim: Embedding dimension (default 7 as per Cosmos 2.5)
        latent_h: Latent spatial height
        latent_w: Latent spatial width
    """
    
    def __init__(
        self, 
        num_views: int = 2, 
        embed_dim: int = 7,
    ):
        super().__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        
        # Learnable embeddings for each view - (num_views, embed_dim)
        # These get broadcast across spatial and temporal dimensions
        self.view_embeddings = nn.Parameter(torch.randn(num_views, embed_dim) * 0.02)
    
    def forward(self, latent: Tensor, view_idx: int) -> Tensor:
        """Add view embedding to latent tensor.
        
        Args:
            latent: Latent tensor of shape (B, C, T, H, W)
            view_idx: Index of the view (0 for front, 1 for wrist, etc.)
            
        Returns:
            Tensor of shape (B, C + embed_dim, T, H, W)
        """
        B, C, T, H, W = latent.shape
        
        # Get embedding for this view and expand to match latent shape
        view_emb = self.view_embeddings[view_idx]  # (embed_dim,)
        view_emb = view_emb.view(1, self.embed_dim, 1, 1, 1)  # (1, embed_dim, 1, 1, 1)
        view_emb = view_emb.expand(B, -1, T, H, W)  # (B, embed_dim, T, H, W)
        
        # Concatenate along channel dimension
        return torch.cat([latent, view_emb], dim=1)


class MultiviewRotaryPosEmbed(nn.Module):
    """Rotary Position Embedding with per-view temporal position reset.
    
    Following Cosmos 2.5: "Although we concatenate the views in the latent temporal
    dimension, we construct the RoPE embeddings separately per view."
    
    This means each view gets temporal positions 0, 1, ..., T-1 rather than
    the concatenated sequence getting 0, 1, ..., 2T-1.
    
    Interface matches diffusers.CosmosRotaryPosEmbed so it can replace
    transformer.rope directly.
    
    Args:
        hidden_size: Dimension for RoPE (typically attention_head_dim)
        max_size: Maximum size (T, H, W) in latent space
        patch_size: Patch size (p_t, p_h, p_w)
        base_fps: Base FPS for temporal scaling
        rope_scale: Scaling factors for (T, H, W) dimensions
        num_views: Number of camera views
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        num_views: int = 2,
    ):
        super().__init__()
        
        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.base_fps = base_fps
        self.num_views = num_views
        
        # Split hidden dim into temporal, height, width components
        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w
        
        # NTK-aware scaling factors
        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        fps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings with per-view temporal position reset.
        
        Args:
            hidden_states: Input tensor (B, C, num_frames, H, W)
                For multiview, num_frames = num_views * frames_per_view
            fps: Optional FPS for temporal scaling
            
        Returns:
            Tuple of (cos, sin) embeddings for rotary application
        """
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        device = hidden_states.device
        
        # Compute per-view frame count
        frames_per_view = num_frames // self.num_views
        
        # Post-patch sizes
        pe_h = height // self.patch_size[1]
        pe_w = width // self.patch_size[2]
        pe_t_per_view = frames_per_view // self.patch_size[0]
        
        # Compute frequency bases with NTK scaling
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor
        
        # Create position sequence
        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        
        # Compute frequency ranges
        dim_h_range = (
            torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        )
        dim_w_range = (
            torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        )
        dim_t_range = (
            torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        )
        
        h_spatial_freqs = 1.0 / (h_theta ** dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta ** dim_w_range)
        temporal_freqs = 1.0 / (t_theta ** dim_t_range)
        
        # Spatial embeddings (same for all views, computed for full H, W)
        emb_h = torch.outer(seq[:pe_h], h_spatial_freqs)  # (H, dim_h/2)
        emb_w = torch.outer(seq[:pe_w], w_spatial_freqs)  # (W, dim_w/2)
        
        # Temporal embeddings - KEY DIFFERENCE: positions 0..T-1 per view
        if fps is None:
            emb_t_single = torch.outer(seq[:pe_t_per_view], temporal_freqs)  # (T, dim_t/2)
        else:
            emb_t_single = torch.outer(
                seq[:pe_t_per_view] / fps * self.base_fps, 
                temporal_freqs
            )  # (T, dim_t/2)
        
        # Expand spatial embeddings to full grid for single view
        # Shape: (T, H, W, dim/2)
        emb_h_expanded = emb_h[None, :, None, :].repeat(pe_t_per_view, 1, pe_w, 1)
        emb_w_expanded = emb_w[None, None, :, :].repeat(pe_t_per_view, pe_h, 1, 1)
        emb_t_expanded = emb_t_single[:, None, None, :].repeat(1, pe_h, pe_w, 1)
        
        # Concatenate T, H, W embeddings for single view
        freqs_single = torch.cat([emb_t_expanded, emb_h_expanded, emb_w_expanded] * 2, dim=-1)
        freqs_single = freqs_single.flatten(0, 2)  # (T*H*W, hidden_size)
        
        # Repeat for each view (this is the per-view RoPE: each view gets same positions)
        freqs = freqs_single.repeat(self.num_views, 1)  # (num_views*T*H*W, hidden_size)
        
        # Compute cos and sin in same dtype as hidden_states (e.g. bfloat16) to avoid dtype mismatch
        dtype = hidden_states.dtype
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)

        return cos, sin


class MultiviewCosmosWrapper(nn.Module):
    """Wrapper for Cosmos Predict 2.5 with multiview support.
    
    Handles:
    - Loading pretrained Cosmos model
    - Per-view embedding injection
    - Multiview latent concatenation along temporal dimension
    - LoRA-based finetuning
    - Flow matching training
    
    Args:
        model_name: HuggingFace model name for Cosmos
        num_views: Number of camera views (default 2: front + wrist)
        view_embed_dim: Per-view embedding dimension (default 7)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        freeze_vae: Whether to freeze VAE during training
        freeze_text_encoder: Whether to freeze T5 text encoder
        normalize: Normalization function for video input (default: [0,1] -> [-1,1])
    """
    
    def __init__(
        self,
        model_name: str = 'nvidia/Cosmos-Predict2-2B-Video2World',
        num_views: int = 2,
        view_embed_dim: int = 7,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        freeze_vae: bool = True,
        freeze_text_encoder: bool = True,
        normalize=lambda t: (t - 0.5) * 2.0,  # Same as mimic_video default
    ):
        super().__init__()
        
        self.num_views = num_views
        self.view_embed_dim = view_embed_dim
        self.normalize = normalize
        
        # Load pretrained model components
        self._load_pretrained(model_name)
        
        # Create view embeddings
        self.view_embedding = ViewEmbedding(
            num_views=num_views,
            embed_dim=view_embed_dim,
        )
        
        # Apply LoRA to transformer
        self._apply_lora(lora_r, lora_alpha, lora_dropout)
        
        # Freeze components as needed
        if freeze_vae:
            self.vae.requires_grad_(False)
        if freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
        
        # Store config
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
    def _load_pretrained(self, model_name: str):
        """Load pretrained Cosmos model components directly (bypasses safety checker)."""
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
        from transformers import T5EncoderModel, T5TokenizerFast

        print(f"Loading pretrained model components from: {model_name}")

        # Load each component directly to avoid cosmos_guardrail dependency
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        
        # Store VAE config (AutoencoderKLWan uses different attribute names)
        self.vae_temporal_compression = self.vae.config.scale_factor_temporal
        self.vae_spatial_compression = self.vae.config.scale_factor_spatial
        self.latent_channels = self.vae.config.z_dim

        # Flow-space scaling: use per-channel stats from VAE config (not fixed 0/1)
        # See pipeline_cosmos2_video2world.py L306–311 and vae/config.json (latents_mean, latents_std)
        self.sigma_data = 1.0
        self._latents_mean = self._latents_std = None
        self._register_latent_stats()

        # Replace transformer's RoPE with per-view version
        # This ensures RoPE positions reset for each view instead of being continuous
        self.transformer.rope = MultiviewRotaryPosEmbed(
            hidden_size=self.transformer.config.attention_head_dim,
            max_size=self.transformer.config.max_size,
            patch_size=self.transformer.config.patch_size,
            rope_scale=self.transformer.config.rope_scale,
            num_views=self.num_views,
        )
        print(f"Replaced RoPE with per-view version (num_views={self.num_views})")

        # Resize patch_embed to handle extra channels from view embeddings
        self._resize_patch_embed_for_view_channels()

    def _register_latent_stats(self):
        """Register per-channel latent mean/std from VAE config (flow-space scaling).
        These are model-specific, not fixed: see nvidia/Cosmos-Predict2-2B-Video2World vae/config.json.
        """
        mean = getattr(self.vae.config, "latents_mean", None)
        std = getattr(self.vae.config, "latents_std", None)
        z = self.latent_channels
        if mean is not None and std is not None and len(mean) == z and len(std) == z:
            self._latents_mean = torch.tensor(mean, dtype=torch.float32).view(1, z, 1, 1, 1)
            self._latents_std = torch.tensor(std, dtype=torch.float32).view(1, z, 1, 1, 1)
        else:
            self._latents_mean = torch.zeros(1, z, 1, 1, 1, dtype=torch.float32)
            self._latents_std = torch.ones(1, z, 1, 1, 1, dtype=torch.float32)

    def _normalize_latent(self, latent: Tensor) -> Tensor:
        """(latent - mean) / std * sigma_data. Per-channel: mean/std from VAE config.
        Matches diffusers pipeline_cosmos_video2world prepare_latents: (z - mean) * sigma_data / std."""
        device, dtype = latent.device, latent.dtype
        mean = self._latents_mean.to(device=device, dtype=dtype)
        std = self._latents_std.to(device=device, dtype=dtype)
        return (latent - mean) / std * self.sigma_data

    def _denormalize_latent(self, latent: Tensor) -> Tensor:
        """latent * std / sigma_data + mean. Per-channel: mean/std from VAE config.
        Matches diffusers pipeline decode: latents * latents_std / sigma_data + latents_mean."""
        device, dtype = latent.device, latent.dtype
        mean = self._latents_mean.to(device=device, dtype=dtype)
        std = self._latents_std.to(device=device, dtype=dtype)
        return latent * std / self.sigma_data + mean

    def _edm_c_in(self, sigma: Tensor) -> Tensor:
        """EDM preconditioning: c_in = 1 / sqrt(sigma^2 + sigma_data^2)."""
        return 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5

    def _edm_c_skip_c_out(self, sigma: Tensor) -> Tuple[Tensor, Tensor]:
        """EDM preconditioning: c_skip, c_out for x0 = c_skip*sample + c_out*model_output (epsilon prediction)."""
        c_skip = (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    def _edm_c_noise(self, sigma: Tensor) -> Tensor:
        """EDM timestep embedding: c_noise = 0.25 * log(sigma). Passed to transformer as timestep."""
        return 0.25 * torch.log(sigma.clamp(min=1e-9))

    def _resize_patch_embed_for_view_channels(self):
        """Resize patch_embed projection to handle extra channels from view embeddings.

        The view embeddings add `view_embed_dim` channels to the latent input.
        We need to expand the patch_embed.proj linear layer to accept these extra channels.
        """
        patch_embed = self.transformer.patch_embed
        old_proj = patch_embed.proj

        # Get patch size from config
        patch_size = self.transformer.config.patch_size  # (t, h, w)
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]

        # Calculate new input features based on actual channel structure:
        # VAE latent (16) + view_embed (7) + padding_mask (1) = 24 channels
        old_in_features = old_proj.in_features
        new_channels = self.latent_channels + self.view_embed_dim + 1  # +1 for padding_mask
        new_in_features = new_channels * patch_volume

        # Create new projection layer
        new_proj = nn.Linear(
            new_in_features,
            old_proj.out_features,
            bias=old_proj.bias is not None,
        ).to(device=old_proj.weight.device)

        # Copy original weights for the original channels (cast to float32 for autocast compatibility)
        with torch.no_grad():
            new_proj.weight[:, :old_in_features] = old_proj.weight.float()
            # Initialize new channel weights with small values
            nn.init.normal_(new_proj.weight[:, old_in_features:], std=0.02)
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias.float())

        # Replace the projection
        patch_embed.proj = new_proj
        print(f"Resized patch_embed: {old_in_features} -> {new_in_features} features "
              f"({new_channels} channels: {self.latent_channels} VAE + {self.view_embed_dim} view + 1 padding)")

    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to transformer."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            modules_to_save=["patch_embed"],
            lora_dropout=dropout,
            bias="none",
        )
        
        if not isinstance(self.transformer, PeftModel):
            self.transformer = get_peft_model(self.transformer, lora_config)
            print(f"Applied LoRA with r={r}, alpha={alpha}")
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode_video(self, video: Tensor) -> Tensor:
        """Encode video to latent space using VAE.
        
        Args:
            video: Video tensor (B, T, C, H, W) in [0, 1]
            
        Returns:
            Latent tensor (B, latent_C, T', H', W')
        """
        # Normalize to [-1, 1] range expected by VAE
        # Using same normalization as mimic_video: (t - 0.5) * 2.0
        video = self.normalize(video)
        
        # Rearrange to (B, C, T, H, W) for VAE
        video = rearrange(video, 'b t c h w -> b c t h w')

        # Cast to VAE dtype (typically bfloat16)
        video = video.to(dtype=self.vae.dtype)

        with torch.no_grad():
            latent = self.vae.encode(video).latent_dist.sample()
        # Normalize to flow space using per-channel stats from VAE config (pipeline L306–311)
        return self._normalize_latent(latent)
    
    def decode_latent(self, latent: Tensor) -> Tensor:
        """Decode latent to video using VAE.

        Args:
            latent: Latent tensor (B, C, T, H, W) in flow-normalized space

        Returns:
            Video tensor (B, T, C, H, W) in [0, 1]
        """
        # Denormalize from flow space using per-channel stats from VAE config (pipeline L764–774)
        latent = self._denormalize_latent(latent)
        latent = latent.to(dtype=self.vae.dtype)

        with torch.no_grad():
            video = self.vae.decode(latent).sample
        
        # Rearrange to (B, T, C, H, W)
        video = rearrange(video, 'b c t h w -> b t c h w')
        
        # Normalize from [-1, 1] to [0, 1]
        video = (video + 1.0) / 2.0
        video = video.clamp(0, 1)
        
        return video
    
    def encode_text(self, prompts: List[str]) -> Tensor:
        """Encode text prompts using T5.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Text embeddings tensor
        """
        text_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            encoder_output = self.text_encoder(**text_inputs)
            text_embeds = encoder_output.last_hidden_state
        
        return text_embeds
    
    def prepare_multiview_latents(
        self,
        front_video: Tensor,
        wrist_video: Tensor,
    ) -> Tuple[Tensor, int]:
        """Encode and prepare multiview latents.
        
        Following Cosmos 2.5: "we re-purpose the latent temporal dimension by 
        concatenating multiple views along it, effectively treating views as 
        sequential frames."
        
        Args:
            front_video: Front camera video (B, T, C, H, W) in [0, 1]
            wrist_video: Wrist camera video (B, T, C, H, W) in [0, 1]
            
        Returns:
            Tuple of:
                - multiview_latent: (B, C + view_embed_dim, 2*T', H', W')
                - frames_per_view: Number of latent frames per view
        """
        # Encode each view independently
        front_latent = self.encode_video(front_video)  # (B, C, T', H', W')
        wrist_latent = self.encode_video(wrist_video)  # (B, C, T', H', W')
        
        frames_per_view = front_latent.shape[2]
        
        # Add per-view embeddings
        front_latent_with_emb = self.view_embedding(front_latent, view_idx=0)
        wrist_latent_with_emb = self.view_embedding(wrist_latent, view_idx=1)
        
        # Concatenate along temporal dimension
        multiview_latent = torch.cat([front_latent_with_emb, wrist_latent_with_emb], dim=2)
        
        return multiview_latent, frames_per_view
    
    def forward_transformer(
        self,
        latent: Tensor,
        timestep: Tensor,
        text_embeds: Tensor,
        frames_per_view: Optional[int] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through transformer.
        
        Args:
            latent: Input latent (B, C, T, H, W), already scaled by c_in when using EDM
            timestep: EDM c_noise = 0.25*log(sigma) (B,) — pass as-is; diffusers does NOT use [0,1000]
            text_embeds: Text embeddings from T5
            frames_per_view: For per-view RoPE construction
            fps: FPS for RoPE (default 16)
            condition_mask: (B, 1, T, H, W) ones=conditioned, zeros=generated; concat to channels if provided
            
        Returns:
            Model output (B, C_lat, T, H, W) — combine with c_skip/c_out for x0 in EDM
        """
        B, C, T, H, W = latent.shape
        
        if timestep.dim() > 1:
            timestep = timestep[:, 0]
        # Diffusers Cosmos uses EDMEulerScheduler: timestep = c_noise = 0.25*log(sigma). Do NOT scale by 1000.

        _, _, _, H, W = latent.shape
        padding_mask = torch.ones(1, 1, H, W, device=latent.device, dtype=latent.dtype)

        output = self.transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            fps=fps if fps is not None else 16,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        
        return output
    
    def compute_flow_loss(
        self,
        front_video: Tensor,
        wrist_video: Tensor,
        prompts: List[str],
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> Tensor:
        """Compute EDM denoising loss (aligns with diffusers Cosmos / EDMEulerScheduler).
        
        Sample sigma, xt = x0 + sigma*noise; model input = c_in*xt, timestep = c_noise;
        target model output satisfies c_skip*xt + c_out*pred = x0 => pred = (x0 - c_skip*xt)/c_out.
        
        Args:
            front_video: Front camera video (B, T, C, H, W) in [0, 1]
            wrist_video: Wrist camera video (B, T, C, H, W) in [0, 1]
            prompts: List of text prompts
            sigma_min, sigma_max: Sigma range for training (log-uniform sample)
            
        Returns:
            MSE loss between predicted and target model output
        """
        B = front_video.shape[0]
        device = front_video.device
        
        text_embeds = self.encode_text(prompts)
        latents, frames_per_view = self.prepare_multiview_latents(front_video, wrist_video)
        C_lat = self.latent_channels
        
        # Sample sigma (log-uniform in [sigma_min, sigma_max])
        u = torch.rand(B, device=device, dtype=latents.dtype)
        log_sigma = torch.log(torch.tensor(sigma_min, device=device, dtype=latents.dtype)) + u * (
            torch.log(torch.tensor(sigma_max, device=device, dtype=latents.dtype)) - torch.log(torch.tensor(sigma_min, device=device, dtype=latents.dtype))
        )
        sigma = log_sigma.exp()
        sigma = sigma.view(B, 1, 1, 1, 1)
        
        # xt = x0 + sigma*noise (EDM forward; only latent channels)
        noise = torch.randn(B, C_lat, latents.shape[2], latents.shape[3], latents.shape[4], device=device, dtype=latents.dtype)
        x0_lat = latents[:, :C_lat]
        xt_lat = x0_lat + sigma * noise
        
        # Replace latent channels in full tensor (keep view embedding channels)
        noisy_latents = latents.clone()
        noisy_latents[:, :C_lat] = xt_lat
        
        # EDM preconditioning: scale input by c_in, pass c_noise as timestep
        c_in = self._edm_c_in(sigma)
        scaled_latents = noisy_latents.clone()
        scaled_latents[:, :C_lat] = noisy_latents[:, :C_lat] * c_in
        
        c_noise = self._edm_c_noise(sigma)
        c_noise_b = c_noise.squeeze()
        
        pred = self.forward_transformer(
            scaled_latents,
            c_noise_b,
            text_embeds,
            frames_per_view=frames_per_view,
        )
        
        c_skip, c_out = self._edm_c_skip_c_out(sigma)
        # Target: c_skip*xt + c_out*target = x0 => target = (x0 - c_skip*xt) / c_out
        target = (x0_lat - c_skip * xt_lat) / c_out.clamp(min=1e-7)
        loss = F.mse_loss(pred[:, :C_lat], target)
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        front_context: Tensor,
        wrist_context: Tensor,
        prompts: List[str],
        num_inference_steps: int = 10,
        num_frames_to_generate: int = 5,
        sigma_max: float = 80.0,
        fps: int = 16,
    ) -> Tuple[Tensor, Tensor]:
        """Generate future frames using EDM sampling (aligns with diffusers Cosmos pipeline).
        
        Uses EDMEulerScheduler logic: sigma schedule, c_in/c_skip/c_out preconditioning,
        timestep = c_noise = 0.25*log(sigma), and scheduler.step for ODE integration.
        
        Args:
            front_context: Front camera context frames (B, T_ctx, C, H, W) in [0, 1]
            wrist_context: Wrist camera context frames (B, T_ctx, C, H, W) in [0, 1]
            prompts: List of text prompts
            num_inference_steps: Number of denoising steps
            num_frames_to_generate: Number of future frames to generate per view
            sigma_max: Max sigma for initial noise (pipeline: latents * sigma_max)
            fps: FPS for transformer RoPE
            
        Returns:
            Tuple of (front_generated, wrist_generated) both (B, T_gen, C, H, W) in [0, 1]
        """
        from diffusers import EDMEulerScheduler

        B = front_context.shape[0]
        device = front_context.device
        
        text_embeds = self.encode_text(prompts)
        front_ctx_latent = self.encode_video(front_context)
        wrist_ctx_latent = self.encode_video(wrist_context)
        
        _, C_lat, T_ctx_lat, H_lat, W_lat = front_ctx_latent.shape
        
        T_gen_lat = max(1, num_frames_to_generate // self.vae_temporal_compression)
        
        # Initial noise scaled by sigma_max (pipeline_cosmos_video2world prepare_latents)
        front_future_latent = torch.randn(B, C_lat, T_gen_lat, H_lat, W_lat, device=device, dtype=front_ctx_latent.dtype)
        wrist_future_latent = torch.randn(B, C_lat, T_gen_lat, H_lat, W_lat, device=device, dtype=wrist_ctx_latent.dtype)
        front_future_latent = front_future_latent * sigma_max
        wrist_future_latent = wrist_future_latent * sigma_max
        
        front_ctx_with_emb = self.view_embedding(front_ctx_latent, view_idx=0)
        wrist_ctx_with_emb = self.view_embedding(wrist_ctx_latent, view_idx=1)
        
        # Scheduler: same config as pipeline (sigma_data must match our normalize)
        scheduler = EDMEulerScheduler(
            sigma_min=0.002,
            sigma_max=sigma_max,
            sigma_data=self.sigma_data,
            sigma_schedule="karras",
            num_train_timesteps=1000,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # c_noise = 0.25*log(sigma)
        sigmas = scheduler.sigmas
        
        for i in range(num_inference_steps):
            t = timesteps[i]
            sigma = sigmas[i]
            if not isinstance(sigma, torch.Tensor):
                sigma = torch.tensor(sigma, device=device, dtype=front_ctx_latent.dtype)
            sigma = sigma.to(device).to(front_ctx_latent.dtype)
            
            front_future_with_emb = self.view_embedding(front_future_latent, view_idx=0)
            wrist_future_with_emb = self.view_embedding(wrist_future_latent, view_idx=1)
            front_full = torch.cat([front_ctx_with_emb, front_future_with_emb], dim=2)
            wrist_full = torch.cat([wrist_ctx_with_emb, wrist_future_with_emb], dim=2)
            multiview_latent = torch.cat([front_full, wrist_full], dim=2)
            
            T_total = multiview_latent.shape[2]
            T_ctx = front_ctx_with_emb.shape[2]
            T_future = front_future_with_emb.shape[2]
            
            # EDM scale model input (c_in) — pipeline calls scheduler.scale_model_input; only scale latent channels
            c_in = self._edm_c_in(sigma)
            c_in_5d = c_in.reshape(1, 1, 1, 1, 1).to(device=multiview_latent.device, dtype=multiview_latent.dtype)
            scaled_latent = multiview_latent.clone()
            scaled_latent[:, :C_lat] = multiview_latent[:, :C_lat] * c_in_5d
            
            # Condition mask: 1 = context, 0 = future (pipeline cond_indicator * ones + (1-cond_indicator)*zeros)
            cond_indicator = torch.zeros(1, 1, T_total, 1, 1, device=device, dtype=multiview_latent.dtype)
            cond_indicator[:, :, :T_ctx] = 1.0
            cond_indicator[:, :, T_ctx + T_future : T_ctx + T_future + T_ctx] = 1.0
            ones_pad = scaled_latent.new_ones(1, 1, T_total, H_lat, W_lat)
            zeros_pad = scaled_latent.new_zeros(1, 1, T_total, H_lat, W_lat)
            condition_mask = (cond_indicator * ones_pad + (1 - cond_indicator) * zeros_pad).expand(B, 1, T_total, H_lat, W_lat)
            
            timestep_b = t.expand(B).to(multiview_latent.dtype)
            
            model_output = self.forward_transformer(
                scaled_latent,
                timestep_b,
                text_embeds,
                fps=fps,
                condition_mask=condition_mask,
            )
            
            # Full current sample (latent channels only) for scheduler
            full_sample = multiview_latent[:, :C_lat].clone()
            full_sample[:, :, :T_ctx] = front_ctx_latent
            full_sample[:, :, T_ctx : T_ctx + T_future] = front_future_latent
            full_sample[:, :, T_ctx + T_future : T_ctx + T_future + T_ctx] = wrist_ctx_latent
            full_sample[:, :, T_ctx + T_future + T_ctx :] = wrist_future_latent
            
            # EDM pred_x0 = c_skip*sample + c_out*model_output; scheduler.step does this if pred_original_sample is None
            c_skip, c_out = self._edm_c_skip_c_out(sigma)
            c_skip_5d = c_skip.reshape(1, 1, 1, 1, 1).to(device=model_output.device, dtype=model_output.dtype)
            c_out_5d = c_out.reshape(1, 1, 1, 1, 1).to(device=model_output.device, dtype=model_output.dtype)
            pred_x0 = c_skip_5d * full_sample + c_out_5d * model_output[:, :C_lat]
            # Pin context: pred_original_sample equals sample on context frames so derivative is 0 there
            pred_x0[:, :, :T_ctx] = full_sample[:, :, :T_ctx]
            pred_x0[:, :, T_ctx + T_future : T_ctx + T_future + T_ctx] = full_sample[:, :, T_ctx + T_future : T_ctx + T_future + T_ctx]
            
            prev_sample = scheduler.step(
                model_output[:, :C_lat],
                t,
                full_sample,
                return_dict=False,
                pred_original_sample=pred_x0,
            )[0]
            
            front_future_latent = prev_sample[:, :, T_ctx : T_ctx + T_future]
            wrist_future_latent = prev_sample[:, :, T_ctx + T_future + T_ctx :]
        
        front_generated = self.decode_latent(front_future_latent)
        wrist_generated = self.decode_latent(wrist_future_latent)
        return front_generated, wrist_generated
    
    def save_checkpoint(self, path: str, step: int):
        """Save model checkpoint including LoRA weights and view embeddings.
        
        Args:
            path: Directory to save checkpoint
            step: Current training step
        """
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapter
        lora_path = os.path.join(path, f"lora_step_{step}")
        self.transformer.save_pretrained(lora_path)
        
        # Save view embeddings
        view_emb_path = os.path.join(path, f"view_embeddings_step_{step}.pt")
        torch.save(self.view_embedding.state_dict(), view_emb_path)
        
        print(f"Saved checkpoint at step {step} to {path}")
    
    def load_checkpoint(self, path: str, step: int):
        """Load model checkpoint.
        
        Args:
            path: Directory containing checkpoint
            step: Training step to load
        """
        # Load LoRA adapter
        lora_path = os.path.join(path, f"lora_step_{step}")
        if os.path.exists(lora_path):
            self.transformer.load_adapter(lora_path, adapter_name="default")
            print(f"Loaded LoRA from {lora_path}")
        
        # Load view embeddings
        view_emb_path = os.path.join(path, f"view_embeddings_step_{step}.pt")
        if os.path.exists(view_emb_path):
            self.view_embedding.load_state_dict(torch.load(view_emb_path))
            print(f"Loaded view embeddings from {view_emb_path}")
    
    def trainable_parameters(self):
        """Get trainable parameters (LoRA + view embeddings)."""
        params = []
        
        # LoRA parameters
        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                params.append(param)
        
        # View embedding parameters
        for param in self.view_embedding.parameters():
            params.append(param)
        
        return params
    
    def print_trainable_params(self):
        """Print number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.trainable_parameters())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
