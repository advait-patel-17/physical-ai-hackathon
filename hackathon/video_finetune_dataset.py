"""
Thin wrapper around MimicVideoDataset for video backbone finetuning.

This module provides a dataset adapter that reformats the existing MimicVideoDataset
output into the format expected by the multiview video finetuning training loop.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hackathon.dataset import MimicVideoDataset, MimicVideoItem


@dataclass
class VideoFinetuneItem:
    """A single training sample for video finetuning.
    
    Contains full video sequences (past + future concatenated) for each view.
    """
    front_video: np.ndarray    # (T_total, H, W, 3) - full sequence from front camera
    wrist_video: np.ndarray    # (T_total, H, W, 3) - full sequence from wrist camera  
    text_prompt: str           # task description


class VideoFinetuneDataset(Dataset):
    """Thin wrapper around MimicVideoDataset for video finetuning.
    
    Concatenates past + future frames into full video sequences and returns
    them in the format expected by the multiview training loop.
    
    Args:
        base_dataset: An existing MimicVideoDataset instance
        
    Returns per sample:
        VideoFinetuneItem with front_video, wrist_video, and text_prompt
    """
    
    def __init__(self, base_dataset: MimicVideoDataset):
        self.base = base_dataset
    
    def __len__(self) -> int:
        return len(self.base)
    
    def __getitem__(self, idx: int) -> VideoFinetuneItem:
        """Get a training sample with concatenated video sequences.
        
        Args:
            idx: Sample index
            
        Returns:
            VideoFinetuneItem with full video sequences for each view
        """
        item: MimicVideoItem = self.base[idx]
        
        # Concatenate past + future frames along time dimension
        # Past frames are already in chronological order (oldest to current)
        # Future frames are in chronological order (current+1 to future)
        front_video = np.concatenate([item.cam_high_past, item.cam_high_future], axis=0)
        wrist_video = np.concatenate([item.wrist_cam_past, item.wrist_cam_future], axis=0)
        
        return VideoFinetuneItem(
            front_video=front_video,
            wrist_video=wrist_video,
            text_prompt=item.command,
        )
    
    @property
    def num_frames_per_view(self) -> int:
        """Total number of frames per view (past + future)."""
        return self.base.visual_context_length + self.base.num_future_frames
    
    @property
    def image_size(self) -> int:
        """Image resolution (square)."""
        return self.base.image_size


def collate_video_finetune(
    batch: list[VideoFinetuneItem],
) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate VideoFinetuneItems into batched tensors.
    
    Converts numpy arrays to torch tensors and rearranges dimensions
    from (B, T, H, W, C) to (B, T, C, H, W) for model input.
    Images are normalized from [0, 255] uint8 to [0, 1] float32.
    
    Args:
        batch: List of VideoFinetuneItem instances
        
    Returns:
        Tuple of:
            - front_videos: (B, T, C, H, W) float32 in [0, 1]
            - wrist_videos: (B, T, C, H, W) float32 in [0, 1]
            - prompts: List[str] of length B
    """
    # Stack numpy arrays and convert to tensors
    # Input shape: (T, H, W, C) -> Output shape: (T, C, H, W)
    front_videos = torch.from_numpy(
        np.stack([item.front_video for item in batch])
    ).permute(0, 1, 4, 2, 3).float() / 255.0
    
    wrist_videos = torch.from_numpy(
        np.stack([item.wrist_video for item in batch])
    ).permute(0, 1, 4, 2, 3).float() / 255.0
    
    prompts = [item.text_prompt for item in batch]
    
    return front_videos, wrist_videos, prompts


def create_video_finetune_dataloader(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 480,
    visual_context_length: int = 5,
    num_future_frames: int = 5,
    temporal_stride: int = 6,
    shuffle: bool = True,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for video finetuning.
    
    Args:
        dataset_path: Path to LeRobot v3.0 dataset
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        image_size: Target image resolution (480 for Cosmos 2.5)
        visual_context_length: Number of past frames
        num_future_frames: Number of future frames
        temporal_stride: Spacing between sampled frames
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        **dataset_kwargs: Additional args for MimicVideoDataset
        
    Returns:
        DataLoader configured for video finetuning
    """
    # Create base dataset
    base_dataset = MimicVideoDataset(
        dataset_path=dataset_path,
        image_size=image_size,
        visual_context_length=visual_context_length,
        num_future_frames=num_future_frames,
        temporal_stride=temporal_stride,
        **dataset_kwargs,
    )
    
    # Wrap with video finetune adapter
    dataset = VideoFinetuneDataset(base_dataset)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_finetune,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    return dataloader
