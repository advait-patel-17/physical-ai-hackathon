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
from diffusers import DiffusionPipeline

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


def logit_normal_sample(size, mu: float = 0.0, sigma: float = 1.0, device='cpu') -> Tensor:
    """Sample from logit-normal distribution for flow time τ_v."""
    z = torch.randn(size, device=device) * sigma + mu
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
        
        # Compute cos and sin
        cos = torch.cos(freqs).float()
        sin = torch.sin(freqs).float()
        
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
        model_name: str = 'nvidia/Cosmos-Predict2.5-2B',
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
        """Load pretrained Cosmos model components."""
        from diffusers import CosmosVideoToWorldPipeline
        
        print(f"Loading pretrained model: {model_name}")
        DiffusionPipeline.from_pretrained("nvidia/Cosmos-Predict2.5-2B", dtype=torch.bfloat16, device_map="cuda")
        
        self.vae = pipeline.vae
        self.transformer = pipeline.transformer
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        
        # Store VAE config
        self.vae_temporal_compression = self.vae.config.temporal_compression_ratio
        self.vae_spatial_compression = self.vae.config.spatial_compression_ratio
        self.latent_channels = self.vae.config.latent_channels
        
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
        
        del pipeline
        
    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to transformer."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
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
        
        with torch.no_grad():
            latent = self.vae.encode(video).latent_dist.sample()
        
        return latent
    
    def decode_latent(self, latent: Tensor) -> Tensor:
        """Decode latent to video using VAE.
        
        Args:
            latent: Latent tensor (B, C, T, H, W)
            
        Returns:
            Video tensor (B, T, C, H, W) in [0, 1]
        """
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
    ) -> Tensor:
        """Forward pass through transformer.
        
        Args:
            latent: Input latent (B, C, T, H, W)
            timestep: Flow timestep (B,) or (B, T)
            text_embeds: Text embeddings from T5
            frames_per_view: For per-view RoPE construction
            
        Returns:
            Predicted flow/velocity (B, C, T, H, W)
        """
        B, C, T, H, W = latent.shape
        
        # For multiview, we need to construct RoPE separately per view
        # The transformer handles this internally if we pass appropriate hints
        # For now, we'll use the standard forward pass
        
        # Prepare timestep for transformer
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).expand(-1, T)  # (B, T)
        
        # Scale timestep to [0, 1000] range expected by diffusers
        timestep_scaled = timestep * 1000.0
        
        # Forward through transformer
        output = self.transformer(
            hidden_states=latent,
            timestep=timestep_scaled,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]
        
        return output
    
    def compute_flow_loss(
        self,
        front_video: Tensor,
        wrist_video: Tensor,
        prompts: List[str],
        tau: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute flow matching loss for training.
        
        Args:
            front_video: Front camera video (B, T, C, H, W) in [0, 1]
            wrist_video: Wrist camera video (B, T, C, H, W) in [0, 1]
            prompts: List of text prompts
            tau: Optional pre-sampled flow time, otherwise sampled from logit-normal
            
        Returns:
            MSE loss between predicted and target flow
        """
        B = front_video.shape[0]
        device = front_video.device
        
        # Encode text
        text_embeds = self.encode_text(prompts)
        
        # Prepare multiview latents
        latents, frames_per_view = self.prepare_multiview_latents(front_video, wrist_video)
        
        # Sample flow time from logit-normal if not provided
        if tau is None:
            tau = logit_normal_sample((B,), mu=0.0, sigma=1.0, device=device)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Interpolate between noise and data
        # noisy_latents = tau * latents + (1 - tau) * noise
        tau_expanded = tau.view(B, 1, 1, 1, 1)
        noisy_latents = tau_expanded * latents + (1 - tau_expanded) * noise
        
        # Target flow: latents - noise (velocity from noise to data)
        flow_target = latents - noise
        
        # Predict flow
        pred_flow = self.forward_transformer(
            noisy_latents,
            tau,
            text_embeds,
            frames_per_view=frames_per_view,
        )
        
        # Compute MSE loss
        loss = F.mse_loss(pred_flow, flow_target)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        front_context: Tensor,
        wrist_context: Tensor,
        prompts: List[str],
        num_inference_steps: int = 10,
        num_frames_to_generate: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """Generate future frames using flow matching sampling.
        
        Args:
            front_context: Front camera context frames (B, T_ctx, C, H, W) in [0, 1]
            wrist_context: Wrist camera context frames (B, T_ctx, C, H, W) in [0, 1]
            prompts: List of text prompts
            num_inference_steps: Number of ODE integration steps (more = better quality)
            num_frames_to_generate: Number of future frames to generate per view
            
        Returns:
            Tuple of (front_generated, wrist_generated) both (B, T_gen, C, H, W) in [0, 1]
        """
        B = front_context.shape[0]
        device = front_context.device
        
        # Encode text
        text_embeds = self.encode_text(prompts)
        
        # Encode context frames to latent
        front_ctx_latent = self.encode_video(front_context)  # (B, C_lat, T'_ctx, H', W')
        wrist_ctx_latent = self.encode_video(wrist_context)
        
        _, C_lat, T_ctx_lat, H_lat, W_lat = front_ctx_latent.shape
        
        # Compute number of latent frames to generate
        # Temporal compression = 8 for Cosmos VAE
        T_gen_lat = num_frames_to_generate // self.vae_temporal_compression
        T_gen_lat = max(1, T_gen_lat)
        
        # Initialize future latent frames with pure noise
        front_future_latent = torch.randn(B, C_lat, T_gen_lat, H_lat, W_lat, device=device, dtype=front_ctx_latent.dtype)
        wrist_future_latent = torch.randn(B, C_lat, T_gen_lat, H_lat, W_lat, device=device, dtype=wrist_ctx_latent.dtype)
        
        # Add view embeddings to context
        front_ctx_with_emb = self.view_embedding(front_ctx_latent, view_idx=0)
        wrist_ctx_with_emb = self.view_embedding(wrist_ctx_latent, view_idx=1)
        
        # Flow matching ODE integration: t goes from 1 (noise) to 0 (data)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        
        for i in range(num_inference_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current  # Negative (going from 1 to 0)
            
            # Add view embeddings to current future latents
            front_future_with_emb = self.view_embedding(front_future_latent, view_idx=0)
            wrist_future_with_emb = self.view_embedding(wrist_future_latent, view_idx=1)
            
            # Concatenate context + future along temporal dimension
            front_full = torch.cat([front_ctx_with_emb, front_future_with_emb], dim=2)
            wrist_full = torch.cat([wrist_ctx_with_emb, wrist_future_with_emb], dim=2)
            
            # Concatenate views along temporal dimension
            multiview_latent = torch.cat([front_full, wrist_full], dim=2)
            
            # Create timestep tensor: context at t=0 (clean), future at t_current
            T_total = multiview_latent.shape[2]
            T_ctx = front_ctx_with_emb.shape[2]
            T_future = front_future_with_emb.shape[2]
            
            # Build per-frame timesteps
            timestep_tensor = torch.zeros(B, T_total, device=device, dtype=multiview_latent.dtype)
            # Future frames of view 1: T_ctx to T_ctx + T_future
            timestep_tensor[:, T_ctx:T_ctx + T_future] = t_current
            # Future frames of view 2: T_ctx + T_future + T_ctx to end
            timestep_tensor[:, T_ctx + T_future + T_ctx:] = t_current
            
            # Predict flow
            pred_flow = self.forward_transformer(
                multiview_latent,
                timestep_tensor[:, 0],  # Scalar timestep for now (simplified)
                text_embeds,
            )
            
            # Extract predicted flow for future frames only
            # View 1 future: positions T_ctx to T_ctx + T_future
            # View 2 future: positions T_ctx + T_future + T_ctx to end
            C_with_emb = front_ctx_with_emb.shape[1]
            
            front_future_flow = pred_flow[:, :C_with_emb, T_ctx:T_ctx + T_future, :, :]
            wrist_future_flow = pred_flow[:, :C_with_emb, T_ctx + T_future + T_ctx:, :, :]
            
            # Remove view embedding channels from flow (keep only latent channels)
            front_future_flow = front_future_flow[:, :C_lat, :, :, :]
            wrist_future_flow = wrist_future_flow[:, :C_lat, :, :, :]
            
            # Euler step: x_next = x_current + dt * flow
            # dt is negative, and flow points from noise to data
            front_future_latent = front_future_latent + dt * front_future_flow
            wrist_future_latent = wrist_future_latent + dt * wrist_future_flow
        
        # Decode generated latents to pixels
        front_generated = self.decode_latent(front_future_latent)  # (B, T_gen, C, H, W)
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
