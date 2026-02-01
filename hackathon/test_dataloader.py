#!/usr/bin/env python
"""
Test script for MimicVideo dataloader.

Usage:
    # Basic test (non-distributed)
    python -m hackathon.test_dataloader --dataset_path /path/to/dataset
    
    # Test with distributed (single node, 2 GPUs)
    torchrun --nproc_per_node=2 -m hackathon.test_dataloader --dataset_path /path/to/dataset --distributed
    
    # Quick test with small batch
    python -m hackathon.test_dataloader --dataset_path /path/to/dataset --num_batches 2 --batch_size 2
"""

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist


def test_dataset_basic(dataset_path: str, num_samples: int = 5):
    """Test basic dataset functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Dataset Loading")
    print("="*60)
    
    from hackathon.dataset import MimicVideoDataset
    
    print(f"Loading dataset from: {dataset_path}")
    start = time.time()
    dataset = MimicVideoDataset(
        dataset_path=dataset_path,
        chunk_size=16,
        temporal_stride=6,
        visual_context_length=5,
        num_future_frames=5,
        image_size=224,
    )
    load_time = time.time() - start
    print(f"✓ Dataset loaded in {load_time:.2f}s")
    
    print(f"\nDataset info:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Total episodes: {dataset.total_episodes}")
    print(f"  - FPS: {dataset.fps}")
    print(f"  - Action dim: {dataset.action_dim}")
    print(f"  - State dim: {dataset.state_dim}")
    
    print(f"\nTesting {num_samples} random samples...")
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    for i, idx in enumerate(indices):
        start = time.time()
        sample = dataset[idx]
        sample_time = time.time() - start
        
        print(f"\n  Sample {i+1} (idx={idx}):")
        print(f"    - cam_high_past: {sample.cam_high_past.shape}, dtype={sample.cam_high_past.dtype}")
        print(f"    - wrist_cam_past: {sample.wrist_cam_past.shape}, dtype={sample.wrist_cam_past.dtype}")
        print(f"    - cam_high_future: {sample.cam_high_future.shape}, dtype={sample.cam_high_future.dtype}")
        print(f"    - wrist_cam_future: {sample.wrist_cam_future.shape}, dtype={sample.wrist_cam_future.dtype}")
        print(f"    - proprio: {sample.proprio.shape}, dtype={sample.proprio.dtype}")
        print(f"    - actions: {sample.actions.shape}, dtype={sample.actions.dtype}")
        print(f"    - command: '{sample.command}'")
        print(f"    - Load time: {sample_time*1000:.1f}ms")
    
    print("\n✓ Basic dataset test PASSED")
    return dataset


def test_collate_function(dataset, batch_size: int = 4):
    """Test the collate function."""
    print("\n" + "="*60)
    print("TEST 2: Collate Function")
    print("="*60)
    
    from hackathon.sampler import collate_mimic_video
    
    # Get a few samples
    indices = torch.randperm(len(dataset))[:batch_size].tolist()
    samples = [dataset[i] for i in indices]
    
    print(f"Collating {batch_size} samples...")
    start = time.time()
    batch = collate_mimic_video(samples)
    collate_time = time.time() - start
    
    print(f"\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"min={value.min():.3f}, max={value.max():.3f}")
        else:
            print(f"  - {key}: {type(value).__name__} of length {len(value)}")
    
    print(f"\nCollate time: {collate_time*1000:.1f}ms")
    
    # Verify shapes
    B = batch_size
    T_past = dataset.visual_context_length
    T_future = dataset.num_future_frames
    H = W = dataset.image_size
    C = 3
    
    assert batch["cam_high_past"].shape == (B, T_past, C, H, W), "cam_high_past shape mismatch"
    assert batch["wrist_cam_past"].shape == (B, T_past, C, H, W), "wrist_cam_past shape mismatch"
    assert batch["cam_high_future"].shape == (B, T_future, C, H, W), "cam_high_future shape mismatch"
    assert batch["wrist_cam_future"].shape == (B, T_future, C, H, W), "wrist_cam_future shape mismatch"
    assert batch["proprio"].shape == (B, T_past, dataset.state_dim), "proprio shape mismatch"
    assert batch["actions"].shape == (B, dataset.chunk_size, dataset.action_dim), "actions shape mismatch"
    assert len(batch["command"]) == B, "command length mismatch"
    
    # Verify normalization
    assert batch["cam_high_past"].min() >= 0.0, "Images should be >= 0"
    assert batch["cam_high_past"].max() <= 1.0, "Images should be <= 1"
    
    print("\n✓ Collate function test PASSED")
    return batch


def test_dataloader_non_distributed(dataset_path: str, batch_size: int = 4, num_batches: int = 5):
    """Test dataloader without distributed."""
    print("\n" + "="*60)
    print("TEST 3: DataLoader (Non-Distributed)")
    print("="*60)
    
    from hackathon.sampler import get_dataloader
    
    print(f"Creating dataloader with batch_size={batch_size}, num_workers=2...")
    dataloader = get_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=2,
        distributed=False,  # Non-distributed
        chunk_size=16,
        image_size=224,
    )
    
    print(f"Total batches: {len(dataloader)}")
    
    print(f"\nIterating through {num_batches} batches...")
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start = time.time()
        
        # Simulate GPU transfer
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        
        batch_time = time.time() - start
        times.append(batch_time)
        
        print(f"  Batch {i+1}: cam_high_past={batch['cam_high_past'].shape}, "
              f"time={batch_time*1000:.1f}ms")
    
    avg_time = sum(times) / len(times) * 1000
    print(f"\nAverage batch time: {avg_time:.1f}ms")
    print(f"Throughput: {batch_size / (avg_time/1000):.1f} samples/sec")
    
    print("\n✓ Non-distributed dataloader test PASSED")


def test_dataloader_distributed(dataset_path: str, batch_size: int = 4, num_batches: int = 5):
    """Test dataloader with distributed (must be launched with torchrun)."""
    print("\n" + "="*60)
    print("TEST 4: DataLoader (Distributed)")
    print("="*60)
    
    if not dist.is_initialized():
        print("Initializing distributed...")
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size}, local_rank={local_rank}")
    
    from hackathon.sampler import get_dataloader
    
    dataloader = get_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=2,
        distributed=True,
        chunk_size=16,
        image_size=224,
    )
    
    print(f"[Rank {rank}] Total batches per GPU: {len(dataloader)}")
    print(f"[Rank {rank}] Effective batch size: {batch_size * world_size}")
    
    # Test epoch setting
    dataloader.sampler.set_epoch(0)
    
    print(f"\n[Rank {rank}] Iterating through {num_batches} batches...")
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start = time.time()
        
        # Move to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Synchronize for accurate timing
        torch.cuda.synchronize()
        batch_time = time.time() - start
        times.append(batch_time)
        
        if rank == 0:
            print(f"  Batch {i+1}: shape={batch['cam_high_past'].shape}, "
                  f"time={batch_time*1000:.1f}ms")
    
    # Gather timing info
    avg_time = sum(times) / len(times) * 1000
    
    if rank == 0:
        print(f"\n[Rank 0] Average batch time: {avg_time:.1f}ms")
        print(f"[Rank 0] Per-GPU throughput: {batch_size / (avg_time/1000):.1f} samples/sec")
        print(f"[Rank 0] Total throughput: {batch_size * world_size / (avg_time/1000):.1f} samples/sec")
    
    dist.barrier()
    
    if rank == 0:
        print("\n✓ Distributed dataloader test PASSED")
    
    dist.destroy_process_group()


def test_episode_aware_sampler(dataset_path: str, batch_size: int = 4):
    """Test episode-aware sampling."""
    print("\n" + "="*60)
    print("TEST 5: Episode-Aware Sampler")
    print("="*60)
    
    from hackathon.sampler import get_dataloader_with_episode_sampler
    
    # Test dropping frames at episode boundaries
    dataloader = get_dataloader_with_episode_sampler(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=0,  # Use 0 for debugging
        distributed=False,
        drop_n_first_frames=5,  # Skip first 5 frames
        drop_n_last_frames=5,   # Skip last 5 frames
        chunk_size=16,
        image_size=224,
    )
    
    original_len = dataloader.dataset._total_steps
    sampler_len = len(dataloader.sampler)
    
    print(f"Original dataset length: {original_len}")
    print(f"Sampler length (after dropping): {sampler_len}")
    print(f"Dropped frames: {original_len - sampler_len}")
    
    # Verify we can iterate
    batch = next(iter(dataloader))
    print(f"\nFirst batch shape: {batch['cam_high_past'].shape}")
    
    print("\n✓ Episode-aware sampler test PASSED")


def benchmark_throughput(dataset_path: str, batch_size: int = 8, num_batches: int = 50):
    """Benchmark dataloader throughput."""
    print("\n" + "="*60)
    print("BENCHMARK: Dataloader Throughput")
    print("="*60)
    
    from hackathon.sampler import get_dataloader
    
    for num_workers in [0, 2, 4, 8]:
        print(f"\nTesting with num_workers={num_workers}...")
        
        dataloader = get_dataloader(
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            distributed=False,
            chunk_size=16,
            image_size=224,
        )
        
        # Warmup
        for i, _ in enumerate(dataloader):
            if i >= 3:
                break
        
        # Benchmark
        start = time.time()
        samples = 0
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            samples += batch["cam_high_past"].shape[0]
        
        elapsed = time.time() - start
        throughput = samples / elapsed
        
        print(f"  num_workers={num_workers}: {throughput:.1f} samples/sec "
              f"({elapsed:.2f}s for {samples} samples)")


def main():
    parser = argparse.ArgumentParser(description="Test MimicVideo dataloader")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to LeRobot v3.0 dataset")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing")
    parser.add_argument("--num_batches", type=int, default=5,
                        help="Number of batches to iterate")
    parser.add_argument("--distributed", action="store_true",
                        help="Test distributed mode (requires torchrun)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip tests that require video files")
    args = parser.parse_args()
    
    # Check dataset exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return 1
    
    if not (dataset_path / "meta" / "info.json").exists():
        print(f"ERROR: Not a valid LeRobot dataset (missing meta/info.json)")
        return 1
    
    print(f"Testing dataloader with dataset: {dataset_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        if args.distributed:
            # Distributed test only
            test_dataloader_distributed(
                str(dataset_path),
                batch_size=args.batch_size,
                num_batches=args.num_batches,
            )
        else:
            # Run all non-distributed tests
            dataset = test_dataset_basic(str(dataset_path), num_samples=3)
            test_collate_function(dataset, batch_size=args.batch_size)
            
            if not args.skip_video:
                test_dataloader_non_distributed(
                    str(dataset_path),
                    batch_size=args.batch_size,
                    num_batches=args.num_batches,
                )
                test_episode_aware_sampler(str(dataset_path), batch_size=args.batch_size)
            
            if args.benchmark:
                benchmark_throughput(
                    str(dataset_path),
                    batch_size=args.batch_size,
                    num_batches=50,
                )
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
