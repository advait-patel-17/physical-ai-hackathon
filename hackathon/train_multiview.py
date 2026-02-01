#!/usr/bin/env python
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
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
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
    
    # Initialize dataset
    if accelerator.is_main_process:
        print("Loading dataset...")
    
    base_dataset = MimicVideoDataset(
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        visual_context_length=args.visual_context_length,
        num_future_frames=args.num_future_frames,
        temporal_stride=args.temporal_stride,
    )
    
    dataset = VideoFinetuneDataset(base_dataset)
    
    if accelerator.is_main_process:
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Frames per view: {dataset.num_frames_per_view}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_video_finetune,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
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
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
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
    
    # Initialize tracking
    if accelerator.is_main_process:
        accelerator.init_trackers("multiview_cosmos_finetune")
    
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
        for batch in dataloader:
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
