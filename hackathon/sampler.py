"""
Distributed sampling utilities for MimicVideo training.

Usage:
    # In training script
    from hackathon.sampler import get_dataloader
    
    dataloader = get_dataloader(
        dataset_path="/path/to/data",
        batch_size=8,  # Per-GPU batch size
        num_workers=8,
        distributed=True,
    )
    
    # Training loop
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # Important for shuffling
        for batch in dataloader:
            ...

Launch with: torchrun --nproc_per_node=4 train.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Sampler

if TYPE_CHECKING:
    from hackathon.dataset import MimicVideoDataset, MimicVideoItem


def collate_mimic_video(batch: list[MimicVideoItem]) -> dict[str, torch.Tensor | list[str]]:
    """Collate MimicVideoItems into batched tensors.
    
    Converts numpy arrays to torch tensors and rearranges dimensions
    from (B, T, H, W, C) to (B, T, C, H, W) for PyTorch conv layers.
    Images are normalized from [0, 255] to [0, 1].
    
    Args:
        batch: List of MimicVideoItem dataclass instances
        
    Returns:
        Dictionary with batched tensors:
            - cam_high_past: (B, T_past, C, H, W) float32 in [0, 1]
            - wrist_cam_past: (B, T_past, C, H, W) float32 in [0, 1]
            - cam_high_future: (B, T_future, C, H, W) float32 in [0, 1]
            - wrist_cam_future: (B, T_future, C, H, W) float32 in [0, 1]
            - proprio: (B, T_past, state_dim) float32
            - actions: (B, chunk_size, action_dim) float32
            - command: List[str] of length B
    """
    # Stack numpy arrays and convert to tensors
    # Input shape: (T, H, W, C) -> Output shape: (T, C, H, W)
    cam_high_past = torch.from_numpy(
        np.stack([item.cam_high_past for item in batch])
    ).permute(0, 1, 4, 2, 3).float() / 255.0
    
    wrist_cam_past = torch.from_numpy(
        np.stack([item.wrist_cam_past for item in batch])
    ).permute(0, 1, 4, 2, 3).float() / 255.0
    
    cam_high_future = torch.from_numpy(
        np.stack([item.cam_high_future for item in batch])
    ).permute(0, 1, 4, 2, 3).float() / 255.0
    
    wrist_cam_future = torch.from_numpy(
        np.stack([item.wrist_cam_future for item in batch])
    ).permute(0, 1, 4, 2, 3).float() / 255.0
    
    proprio = torch.from_numpy(
        np.stack([item.proprio for item in batch])
    )
    
    actions = torch.from_numpy(
        np.stack([item.actions for item in batch])
    )
    
    commands = [item.command for item in batch]
    
    return {
        "cam_high_past": cam_high_past,
        "wrist_cam_past": wrist_cam_past,
        "cam_high_future": cam_high_future,
        "wrist_cam_future": wrist_cam_future,
        "proprio": proprio,
        "actions": actions,
        "command": commands,
    }


def get_dataloader(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 8,
    distributed: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    **dataset_kwargs,
) -> DataLoader:
    """Factory function to create a dataloader with distributed support.
    
    Args:
        dataset_path: Path to LeRobot v3.0 format dataset
        batch_size: Batch size per GPU (effective batch = batch_size * num_gpus)
        num_workers: Number of data loading workers
        distributed: Whether to use DistributedSampler for multi-GPU training
        drop_last: Drop last incomplete batch (recommended for DDP)
        pin_memory: Pin memory for faster CPU→GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        **dataset_kwargs: Additional arguments passed to MimicVideoDataset
        
    Returns:
        DataLoader configured for training
        
    Example:
        >>> dataloader = get_dataloader(
        ...     dataset_path="/data/lerobot_dataset",
        ...     batch_size=8,
        ...     chunk_size=16,
        ...     image_size=224,
        ... )
        >>> for epoch in range(10):
        ...     dataloader.sampler.set_epoch(epoch)
        ...     for batch in dataloader:
        ...         loss = model(batch)
    """
    # Import here to avoid circular imports
    from hackathon.dataset import MimicVideoDataset
    
    dataset = MimicVideoDataset(dataset_path, **dataset_kwargs)
    
    # Set up sampler based on distributed mode
    sampler: Sampler | None = None
    shuffle = True
    
    if distributed and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=drop_last,
        )
        shuffle = False  # Sampler handles shuffling
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_mimic_video,
    )
    
    return dataloader


def get_dataloader_from_dataset(
    dataset: MimicVideoDataset,
    batch_size: int = 8,
    num_workers: int = 8,
    distributed: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a dataloader from an existing dataset instance.
    
    Use this when you need to reuse the same dataset object (e.g., for
    validation dataloader with different batch size).
    
    Args:
        dataset: Pre-created MimicVideoDataset instance
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        distributed: Whether to use DistributedSampler
        drop_last: Drop last incomplete batch
        pin_memory: Pin memory for faster transfer
        prefetch_factor: Batches to prefetch per worker
        
    Returns:
        DataLoader configured for training
    """
    sampler: Sampler | None = None
    shuffle = True
    
    if distributed and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=drop_last,
        )
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_mimic_video,
    )


class EpisodeAwareSampler(Sampler):
    """Sampler that respects episode boundaries.
    
    Optionally drops frames from the beginning/end of episodes to avoid
    edge cases where temporal context is incomplete.
    
    Args:
        dataset: MimicVideoDataset instance
        drop_n_first_frames: Frames to skip at episode start
        drop_n_last_frames: Frames to skip at episode end
        shuffle: Whether to shuffle indices
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: MimicVideoDataset,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.drop_n_first_frames = drop_n_first_frames
        self.drop_n_last_frames = drop_n_last_frames
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Build list of valid indices
        self.indices = []
        for episode_idx, episode_length in dataset._episode_lengths.items():
            start_idx = dataset._episode_start_indices[episode_idx]
            
            # Apply frame dropping
            valid_start = drop_n_first_frames
            valid_end = episode_length - drop_n_last_frames
            
            if valid_start < valid_end:
                for frame_idx in range(valid_start, valid_end):
                    global_idx = start_idx + frame_idx
                    self.indices.append(global_idx)
    
    def __iter__(self):
        if self.shuffle:
            # Deterministic shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.indices), generator=g)
            yield from (self.indices[i] for i in perm)
        else:
            yield from self.indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch


class DistributedEpisodeAwareSampler(Sampler):
    """Distributed sampler that respects episode boundaries.
    
    Combines EpisodeAwareSampler with distributed sharding for multi-GPU training.
    
    Args:
        dataset: MimicVideoDataset instance
        num_replicas: Number of distributed processes (GPUs)
        rank: Rank of current process
        drop_n_first_frames: Frames to skip at episode start
        drop_n_last_frames: Frames to skip at episode end  
        shuffle: Whether to shuffle indices
        seed: Random seed for reproducibility
        drop_last: Drop samples to make dataset evenly divisible
    """
    
    def __init__(
        self,
        dataset: MimicVideoDataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            if not dist.is_initialized():
                raise RuntimeError("Distributed not initialized. Call dist.init_process_group() first.")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError("Distributed not initialized. Call dist.init_process_group() first.")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_n_first_frames = drop_n_first_frames
        self.drop_n_last_frames = drop_n_last_frames
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Build list of valid indices (same as EpisodeAwareSampler)
        self.indices = []
        for episode_idx, episode_length in dataset._episode_lengths.items():
            start_idx = dataset._episode_start_indices[episode_idx]
            
            valid_start = drop_n_first_frames
            valid_end = episode_length - drop_n_last_frames
            
            if valid_start < valid_end:
                for frame_idx in range(valid_start, valid_end):
                    global_idx = start_idx + frame_idx
                    self.indices.append(global_idx)
        
        # Calculate samples per replica
        total_size = len(self.indices)
        if self.drop_last and total_size % self.num_replicas != 0:
            self.num_samples = total_size // self.num_replicas
        else:
            self.num_samples = (total_size + self.num_replicas - 1) // self.num_replicas
        
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Deterministic shuffle
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in perm]
        else:
            indices = list(self.indices)
        
        # Pad to make evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across all replicas."""
        self.epoch = epoch


def get_dataloader_with_episode_sampler(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 8,
    distributed: bool = True,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    **dataset_kwargs,
) -> DataLoader:
    """Create dataloader with episode-aware sampling.
    
    Use this when you need to drop frames at episode boundaries (e.g., when
    temporal context at the start/end of episodes would be incomplete).
    
    Args:
        dataset_path: Path to dataset
        batch_size: Batch size per GPU
        num_workers: Number of workers
        distributed: Use distributed sampling
        drop_n_first_frames: Skip N frames at episode start
        drop_n_last_frames: Skip N frames at episode end
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch batches per worker
        **dataset_kwargs: Args for MimicVideoDataset
        
    Returns:
        DataLoader with episode-aware sampling
    """
    from hackathon.dataset import MimicVideoDataset
    
    dataset = MimicVideoDataset(dataset_path, **dataset_kwargs)
    
    if distributed and dist.is_initialized():
        sampler = DistributedEpisodeAwareSampler(
            dataset,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=True,
        )
    else:
        sampler = EpisodeAwareSampler(
            dataset,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=True,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
        collate_fn=collate_mimic_video,
    )
