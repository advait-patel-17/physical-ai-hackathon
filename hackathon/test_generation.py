#!/usr/bin/env python
"""
Test script for multiview Cosmos model generation.

Runs a single generation with dummy or optional real data to verify:
- No dtype errors (float vs bfloat16) in RoPE / transformer
- End-to-end generate() path works

Usage (from repo root):
    # Default: dummy data, 1 sample, 2 inference steps (quick)
    PYTHONPATH=. uv run python hackathon/test_generation.py

    # More steps and save outputs
    PYTHONPATH=. uv run python hackathon/test_generation.py --num_inference_steps 4 --save_dir ./test_out

    # Optional: resume from checkpoint
    PYTHONPATH=. uv run python hackathon/test_generation.py --resume_from_checkpoint ./output/checkpoint_step_500 --checkpoint_step 500
"""

from __future__ import annotations

import argparse
import os

import torch

from hackathon.multiview_cosmos import MultiviewCosmosWrapper


def save_video_to_disk(video: torch.Tensor, path: str, fps: int = 8) -> None:
    """Save video (T, C, H, W) in [0, 1] to MP4."""
    try:
        import imageio.v3 as iio
    except ImportError:
        import imageio as iio

    # Ensure float32 and CPU for saving
    video_np = video.float().permute(0, 2, 3, 1).cpu().numpy()
    video_np = (video_np * 255).clip(0, 255).astype("uint8")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    iio.imwrite(path, video_np, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Test multiview Cosmos generation")
    parser.add_argument("--model_name", type=str, default="nvidia/Cosmos-Predict2-2B-Video2World")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--context_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="test_generation_output",
        help="Folder to save generated videos (default: test_generation_output)",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Required if --resume_from_checkpoint is set")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dummy context: (B, T_ctx, C, H, W) in [0, 1]
    B, T_ctx, C, H, W = args.batch_size, args.context_frames, 3, args.image_size, args.image_size
    front_context = torch.rand(B, T_ctx, C, H, W, device=device, dtype=torch.float32)
    wrist_context = torch.rand(B, T_ctx, C, H, W, device=device, dtype=torch.float32)
    prompts = ["Pick up the red block and place it in the bin."] * B

    print("Loading model...")
    model = MultiviewCosmosWrapper(
        model_name=args.model_name,
        num_views=2,
        view_embed_dim=7,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_vae=True,
        freeze_text_encoder=True,
    )
    model = model.to(device)

    if args.resume_from_checkpoint:
        step = args.checkpoint_step
        if step is None:
            raise ValueError("--checkpoint_step is required when using --resume_from_checkpoint")
        print(f"Loading checkpoint: {args.resume_from_checkpoint} (step {step})")
        model.load_checkpoint(args.resume_from_checkpoint, step)

    model.eval()

    # Match training: run generation under bfloat16 autocast to trigger RoPE dtype path
    num_latent_frames = max(1, (args.num_future_frames * 8) // 8)  # vae_temporal_compression = 8

    print("Running generate() under autocast(bfloat16)...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            front_gen, wrist_gen = model.generate(
                front_context,
                wrist_context,
                prompts,
                num_inference_steps=args.num_inference_steps,
                num_frames_to_generate=num_latent_frames * 8,
            )
    except Exception as e:
        print("Generation failed with error:")
        print(type(e).__name__, str(e))
        raise

    print("Generation succeeded.")
    print(f"  front_gen: {front_gen.shape} dtype={front_gen.dtype}")
    print(f"  wrist_gen: {wrist_gen.shape} dtype={wrist_gen.dtype}")

    save_dir = os.path.abspath(args.save_dir)
    front_path = os.path.join(save_dir, "test_front_gen.mp4")
    wrist_path = os.path.join(save_dir, "test_wrist_gen.mp4")
    save_video_to_disk(front_gen[0], front_path)
    save_video_to_disk(wrist_gen[0], wrist_path)
    print(f"Saved to {save_dir}:")
    print(f"  {front_path}")
    print(f"  {wrist_path}")

    print("Done.")


if __name__ == "__main__":
    main()
