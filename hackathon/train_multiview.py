# !/usr/bin/env python
"""
Multiview Video Backbone Finetuning Script.

Finetunes Cosmos Predict 2.5 with multiview support using LoRA and flow matching.

Hyperparameters (from plan):
- LR: 1.778e-4
- Warmup steps: 1000
- Training steps: 5000
- Checkpoints every: 500 steps
- Weight decay: 0.1
- Gradient clipping: 10.0
- Batch size: 64
- Optimizer: AdamW
- Flow time sampling: Logit-normal
- LoRA rank: 8
- LoRA alpha: 16
- LR schedule: Constant (after warmup)

Usage:
    # Single GPU with gradient accumulation
    python train_multiview.py \
        --dataset_path /path/to/dataset \
        --batch_size 8 \
        --gradient_accumulation_steps 8
    
    # Multi-GPU with accelerate
    accelerate launch --num_processes 4 train_multiview.py \
        --dataset_path /path/to/dataset \
        --batch_size 16
"""

import argparse
import os
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import wandb

from hackathon.dataset import MimicVideoDataset
from hackathon.video_finetune_dataset import (
    VideoFinetuneDataset,
    collate_video_finetune,
)
from hackathon.multiview_cosmos import MultiviewCosmosWrapper, logit_normal_sample


# Default hyperparameters from the plan
DEFAULT_LR = 1.778e-4
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_MAX_STEPS = 5000
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    constant_after_warmup: bool = True,
):
    """
    Create a schedule with warmup then constant LR.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Not used if constant_after_warmup=True
        constant_after_warmup: If True, use constant LR after warmup
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        if constant_after_warmup:
            # Constant after warmup
            return 1.0
        # Cosine decay (not used in this plan but kept for reference)
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def save_video_to_disk(
    video: torch.Tensor,
    path: str,
    fps: int = 8,
):
    """Save video tensor to MP4 file.
    
    Args:
        video: Video tensor (T, C, H, W) in [0, 1]
        path: Output path for MP4 file
        fps: Frames per second
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        import imageio as iio
    
    # Convert to numpy uint8: (T, C, H, W) -> (T, H, W, C)
    video_np = video.permute(0, 2, 3, 1).cpu().numpy()
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    
    # Save video
    os.makedirs(os.path.dirname(path), exist_ok=True)
    iio.imwrite(path, video_np, fps=fps)


def create_comparison_grid(
    gt_video: torch.Tensor,
    gen_video: torch.Tensor,
) -> torch.Tensor:
    """Create side-by-side comparison grid of GT and generated video.
    
    Args:
        gt_video: Ground truth video (T, C, H, W)
        gen_video: Generated video (T_gen, C, H, W)
        
    Returns:
        Grid video (T, C, H, W*2) with GT on left, generated on right
    """
    T_gt, C, H, W = gt_video.shape
    T_gen = gen_video.shape[0]
    
    # Use minimum of the two lengths
    T = min(T_gt, T_gen)
    
    # Concatenate horizontally
    grid = torch.cat([gt_video[:T], gen_video[:T]], dim=3)  # (T, C, H, W*2)
    
    return grid


@torch.no_grad()
def run_evaluation(
    model,
    dataloader,
    accelerator: Accelerator,
    global_step: int,
    output_dir: str,
    split: str = "val",
    num_samples: int = 4,
    num_batches_for_loss: int = 10,
    num_inference_steps: int = 10,
):
    """Run evaluation and generate sample videos.
    
    Args:
        model: The model (may be wrapped by accelerator)
        dataloader: Dataloader for the split (train or val)
        accelerator: Accelerator instance
        global_step: Current training step
        output_dir: Directory to save samples
        split: Name of the split ("train" or "val")
        num_samples: Number of samples to generate
        num_batches_for_loss: Number of batches to compute loss over
        num_inference_steps: Number of inference steps for generation
        
    Returns:
        Dictionary with loss and paths to saved samples
    """
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Compute loss over a few batches
    losses = []
    data_iter = iter(dataloader)
    
    for _ in range(min(num_batches_for_loss, len(dataloader))):
        try:
            front_videos, wrist_videos, prompts = next(data_iter)
        except StopIteration:
            break
        
        front_videos = front_videos.to(accelerator.device)
        wrist_videos = wrist_videos.to(accelerator.device)
        
        loss = unwrapped_model.compute_flow_loss(front_videos, wrist_videos, prompts)
        losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    
    # Generate sample videos
    samples_dir = os.path.join(output_dir, "samples", f"step_{global_step}", split)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Get a few samples from the dataloader
    data_iter = iter(dataloader)
    generated_samples = []
    
    for sample_idx in range(num_samples):
        try:
            front_videos, wrist_videos, prompts = next(data_iter)
        except StopIteration:
            break
        
        # Take first item from batch
        front_video = front_videos[0:1].to(accelerator.device)  # (1, T, C, H, W)
        wrist_video = wrist_videos[0:1].to(accelerator.device)
        prompt = [prompts[0]]
        
        # Split into context and future
        T = front_video.shape[1]
        T_ctx = T // 2  # Use first half as context
        T_future = T - T_ctx
        
        front_context = front_video[:, :T_ctx]
        wrist_context = wrist_video[:, :T_ctx]
        front_future_gt = front_video[:, T_ctx:]
        wrist_future_gt = wrist_video[:, T_ctx:]
        
        # Generate future frames
        try:
            front_gen, wrist_gen = unwrapped_model.generate(
                front_context,
                wrist_context,
                prompt,
                num_inference_steps=num_inference_steps,
                num_frames_to_generate=T_future * 8,  # Account for temporal compression
            )
        except Exception as e:
            print(f"Generation failed for sample {sample_idx}: {e}")
            continue
        
        # Save videos to disk
        # Front camera
        front_gt_path = os.path.join(samples_dir, f"sample_{sample_idx}_front_gt.mp4")
        front_gen_path = os.path.join(samples_dir, f"sample_{sample_idx}_front_gen.mp4")
        save_video_to_disk(front_future_gt[0], front_gt_path)
        save_video_to_disk(front_gen[0], front_gen_path)
        
        # Wrist camera
        wrist_gt_path = os.path.join(samples_dir, f"sample_{sample_idx}_wrist_gt.mp4")
        wrist_gen_path = os.path.join(samples_dir, f"sample_{sample_idx}_wrist_gen.mp4")
        save_video_to_disk(wrist_future_gt[0], wrist_gt_path)
        save_video_to_disk(wrist_gen[0], wrist_gen_path)
        
        generated_samples.append({
            "front_gt": front_future_gt[0],
            "front_gen": front_gen[0],
            "wrist_gt": wrist_future_gt[0],
            "wrist_gen": wrist_gen[0],
            "prompt": prompt[0],
        })
        
        if accelerator.is_main_process:
            print(f"  Saved sample {sample_idx}: {prompt[0][:50]}...")
    
    # Log to WandB
    if accelerator.is_main_process and generated_samples:
        # Log videos to wandb
        wandb_videos = {}
        for idx, sample in enumerate(generated_samples):
            # Convert tensors to numpy for wandb.Video
            # Format: (T, C, H, W) -> (T, H, W, C) for wandb
            front_gt_np = (sample["front_gt"].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            front_gen_np = (sample["front_gen"].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            wrist_gt_np = (sample["wrist_gt"].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            wrist_gen_np = (sample["wrist_gen"].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            
            # Create side-by-side comparison videos
            T_min = min(front_gt_np.shape[0], front_gen_np.shape[0])
            front_comparison = np.concatenate([front_gt_np[:T_min], front_gen_np[:T_min]], axis=2)  # Side by side
            T_min_wrist = min(wrist_gt_np.shape[0], wrist_gen_np.shape[0])
            wrist_comparison = np.concatenate([wrist_gt_np[:T_min_wrist], wrist_gen_np[:T_min_wrist]], axis=2)
            
            wandb_videos[f"{split}_samples/front_{idx}"] = wandb.Video(front_comparison, fps=8, format="mp4")
            wandb_videos[f"{split}_samples/wrist_{idx}"] = wandb.Video(wrist_comparison, fps=8, format="mp4")
        
        # Log all videos at once
        wandb.log(wandb_videos, step=global_step)
    
    model.train()
    
    return {
        "loss": avg_loss,
        "samples_dir": samples_dir,
        "num_samples": len(generated_samples),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Multiview video backbone finetuning")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to LeRobot v3.0 dataset or HuggingFace dataset ID",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=480,
        help="Image resolution (480 for Cosmos 2.5)",
    )
    parser.add_argument(
        "--visual_context_length",
        type=int,
        default=5,
        help="Number of past frames",
    )
    parser.add_argument(
        "--num_future_frames",
        type=int,
        default=5,
        help="Number of future frames",
    )
    parser.add_argument(
        "--temporal_stride",
        type=int,
        default=6,
        help="Temporal stride between frames",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/Cosmos-Predict2.5-2B",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=DEFAULT_LORA_R,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * num_gpus * grad_accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
        help="Gradient clipping max norm",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Save checkpoint every N steps",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/multiview_cosmos",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
        help="Run evaluation and generate samples every N steps",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=4,
        help="Number of samples to generate during evaluation",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="multiview-cosmos",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (auto-generated if not provided)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize accelerator with wandb
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Log configuration
    if accelerator.is_main_process:
        print("=" * 60)
        print("Multiview Video Backbone Finetuning")
        print("=" * 60)
        print(f"Dataset: {args.dataset_path}")
        print(f"Model: {args.model_name}")
        print(f"Output: {args.output_dir}")
        print()
        print("Hyperparameters:")
        print(f"  Learning rate: {args.lr}")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Gradient clipping: {args.grad_clip}")
        print(f"  Per-device batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  Mixed precision: {args.mixed_precision}")
        print("=" * 60)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize datasets (train and val splits)
    if accelerator.is_main_process:
        print("Loading datasets...")
    
    # Training dataset
    train_base_dataset = MimicVideoDataset(
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        visual_context_length=args.visual_context_length,
        num_future_frames=args.num_future_frames,
        temporal_stride=args.temporal_stride,
        split="train",
        val_fraction=0.1,
    )
    train_dataset = VideoFinetuneDataset(train_base_dataset)
    
    # Validation dataset
    val_base_dataset = MimicVideoDataset(
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        visual_context_length=args.visual_context_length,
        num_future_frames=args.num_future_frames,
        temporal_stride=args.temporal_stride,
        split="val",
        val_fraction=0.1,
    )
    val_dataset = VideoFinetuneDataset(val_base_dataset)
    
    if accelerator.is_main_process:
        print(f"Train dataset: {len(train_dataset)} samples ({train_base_dataset.num_episodes} episodes)")
        print(f"Val dataset: {len(val_dataset)} samples ({val_base_dataset.num_episodes} episodes)")
        print(f"Frames per view: {train_dataset.num_frames_per_view}")
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_video_finetune,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_video_finetune,
        drop_last=False,
        persistent_workers=False,
    )
    
    # Initialize model
    if accelerator.is_main_process:
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
    
    if accelerator.is_main_process:
        model.print_trainable_params()
    
    # Create optimizer (only for trainable parameters)
    trainable_params = model.trainable_parameters()
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Create scheduler (warmup + constant)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        constant_after_warmup=True,
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Resume from checkpoint if specified
    starting_step = 0
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Extract step number from checkpoint path
        try:
            step_str = args.resume_from_checkpoint.split("step_")[-1]
            starting_step = int(step_str)
        except:
            starting_step = 0
        accelerator.load_state(args.resume_from_checkpoint)
    
    # Initialize wandb tracking
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={
                "learning_rate": args.lr,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "model_name": args.model_name,
                "dataset_path": args.dataset_path,
            },
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )
    
    # Training loop
    if accelerator.is_main_process:
        print("\nStarting training...")
    
    model.train()
    global_step = starting_step
    progress_bar = tqdm(
        total=args.max_steps,
        initial=starting_step,
        desc="Training",
        disable=not accelerator.is_main_process,
    )
    
    running_loss = 0.0
    num_batches_since_log = 0
    
    while global_step < args.max_steps:
        for batch in train_dataloader:
            if global_step >= args.max_steps:
                break
            
            front_videos, wrist_videos, prompts = batch
            
            with accelerator.accumulate(model):
                # Compute loss
                loss = model.module.compute_flow_loss(
                    front_video=front_videos,
                    wrist_video=wrist_videos,
                    prompts=prompts,
                )
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update tracking
            if accelerator.sync_gradients:
                global_step += 1
                running_loss += loss.detach().item()
                num_batches_since_log += 1
                progress_bar.update(1)
                
                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / num_batches_since_log
                    current_lr = scheduler.get_last_lr()[0]
                    
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train_loss": avg_loss,
                                "learning_rate": current_lr,
                                "step": global_step,
                            },
                            step=global_step,
                        )
                        progress_bar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            lr=f"{current_lr:.2e}",
                        )
                    
                    running_loss = 0.0
                    num_batches_since_log = 0
                
                # Evaluation and sample generation
                if global_step % args.eval_every == 0:
                    if accelerator.is_main_process:
                        print(f"\nRunning evaluation at step {global_step}...")
                    
                    # Run on validation set
                    val_results = run_evaluation(
                        model=model,
                        dataloader=val_dataloader,
                        accelerator=accelerator,
                        global_step=global_step,
                        output_dir=args.output_dir,
                        split="val",
                        num_samples=args.num_eval_samples,
                    )
                    
                    # Run on train set (for sample generation comparison)
                    train_results = run_evaluation(
                        model=model,
                        dataloader=train_dataloader,
                        accelerator=accelerator,
                        global_step=global_step,
                        output_dir=args.output_dir,
                        split="train",
                        num_samples=args.num_eval_samples,
                    )
                    
                    if accelerator.is_main_process:
                        accelerator.log({
                            "val_loss": val_results["loss"],
                            "train_eval_loss": train_results["loss"],
                        }, step=global_step)
                        print(f"  Val loss: {val_results['loss']:.4f}")
                        print(f"  Train eval loss: {train_results['loss']:.4f}")
                        print(f"  Saved {val_results['num_samples']} val samples to {val_results['samples_dir']}")
                        print(f"  Saved {train_results['num_samples']} train samples to {train_results['samples_dir']}")
                
                # Checkpointing
                if global_step % args.checkpoint_every == 0:
                    if accelerator.is_main_process:
                        print(f"\nSaving checkpoint at step {global_step}...")
                    
                    # Save accelerator state
                    checkpoint_dir = os.path.join(
                        args.output_dir, f"checkpoint_step_{global_step}"
                    )
                    accelerator.save_state(checkpoint_dir)
                    
                    # Also save LoRA and view embeddings separately for easy loading
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_checkpoint(args.output_dir, global_step)
    
    progress_bar.close()
    
    # Final save
    if accelerator.is_main_process:
        print("\nSaving final checkpoint...")
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
        accelerator.save_state(checkpoint_dir)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_checkpoint(args.output_dir, global_step)
        
        print(f"\nTraining complete! Final step: {global_step}")
        print(f"Checkpoints saved to: {args.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
