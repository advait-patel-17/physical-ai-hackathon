from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def resolve_dataset_path(dataset_path: str) -> Path:
    """Resolve dataset path, downloading from HuggingFace Hub if needed.
    
    Args:
        dataset_path: Local path or HuggingFace dataset ID (e.g., "username/dataset-name")
        
    Returns:
        Path to local dataset directory
    """
    path = Path(dataset_path)
    
    # Check if it's already a local path that exists
    if path.exists():
        return path
    
    # Check if it looks like a HuggingFace dataset ID (contains "/" but doesn't exist locally)
    if "/" in dataset_path and not path.is_absolute():
        print(f"Dataset '{dataset_path}' not found locally, downloading from HuggingFace Hub...")
        try:
            from huggingface_hub import snapshot_download
            
            # Download to HF cache
            local_path = snapshot_download(
                repo_id=dataset_path,
                repo_type="dataset",
            )
            print(f"Downloaded to: {local_path}")
            return Path(local_path)
        except Exception as e:
            raise ValueError(
                f"Could not find dataset at '{dataset_path}' locally or download from HuggingFace Hub. "
                f"Error: {e}"
            )
    
    # Local path that doesn't exist
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")


@dataclass
class EpisodeData:
    """Cached data for a single episode."""
    actions: np.ndarray        # (episode_length, action_dim)
    states: np.ndarray         # (episode_length, state_dim)
    timestamps: np.ndarray     # (episode_length,)
    task_index: int            # Index into tasks table
    global_indices: np.ndarray # (episode_length,) - maps local frame_idx to global video frame


@dataclass
class MimicVideoItem:
    """A single training sample for MimicVideo.
    
    All images are resized to (image_size, image_size, 3).
    Batch dimension is handled by DataLoader, not here.
    """
    cam_high_past: np.ndarray      # (T_past, H, W, 3) - past frames from high/front camera
    wrist_cam_past: np.ndarray     # (T_past, H, W, 3) - past frames from wrist camera
    cam_high_future: np.ndarray    # (T_future, H, W, 3) - future frames from high/front camera
    wrist_cam_future: np.ndarray   # (T_future, H, W, 3) - future frames from wrist camera
    proprio: np.ndarray            # (T_past, D) - proprioceptive state at past timesteps
    actions: np.ndarray            # (chunk_size, action_dim) - action chunk
    command: str                   # task description


class MimicVideoDataset(Dataset):
    """PyTorch Dataset for MimicVideo training.
    
    Loads LeRobot v3.0 format datasets with configurable temporal sampling.
    
    Args:
        dataset_path: Path to LeRobot dataset directory
        chunk_size: Number of consecutive actions to return
        temporal_stride: Spacing between sampled frames
        visual_context_length: Number of past frames to include
        num_future_frames: Number of future frames to include
        image_size: Target size for resized images (square)
        video_key_high: Key for the high/front camera video
        video_key_wrist: Key for the wrist camera video
        split: Data split - "train", "val", or "all"
        val_fraction: Fraction of episodes for validation (default 0.1)
        split_seed: Random seed for reproducible train/val split
    """
    
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 32,
        temporal_stride: int = 6,
        visual_context_length: int = 5,
        num_future_frames: int = 5,
        image_size: int = 224,
        video_key_high: str = "observation.images.front",
        video_key_wrist: str = "observation.images.top",
        split: str = "all",
        val_fraction: float = 0.1,
        split_seed: int = 42,
    ):
        # Resolve dataset path (download from HF Hub if needed)
        self.dataset_path = resolve_dataset_path(dataset_path)
        self.chunk_size = chunk_size
        self.temporal_stride = temporal_stride
        self.visual_context_length = visual_context_length
        self.num_future_frames = num_future_frames
        self.image_size = image_size
        self.video_key_high = video_key_high
        self.video_key_wrist = video_key_wrist
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        
        # Load metadata
        self._load_metadata()
        
        # Build step index mapping (with train/val split)
        self._build_step_index()
        
        # Load all parquet data into memory for fast access
        self._load_parquet_data()
        
        # Video reader cache: path -> VideoReader (decord) or decoded array (imageio)
        self._video_cache: dict = {}
    
    def _load_metadata(self):
        """Load dataset metadata from info.json."""
        info_path = self.dataset_path / "meta" / "info.json"
        with open(info_path, "r") as f:
            self.info = json.load(f)
        
        self.total_frames = self.info["total_frames"]
        self.total_episodes = self.info["total_episodes"]
        self.fps = self.info["fps"]
        self.chunks_size = self.info["chunks_size"]
        self.data_path_pattern = self.info["data_path"]
        self.video_path_pattern = self.info["video_path"]
        
        # Get action and state dimensions from features
        self.action_dim = self.info["features"]["action"]["shape"][0]
        self.state_dim = self.info["features"]["observation.state"]["shape"][0]
        
        # Load tasks
        tasks_path = self.dataset_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            self.tasks_df = pd.read_parquet(tasks_path)
        else:
            self.tasks_df = None
    
    def _build_step_index(self):
        """Build mapping from global index to (episode_id, frame_index).
        
        This allows us to sample any frame across all episodes using a single index.
        Applies train/val split based on episode boundaries.
        """
        self._step_to_episode_map = {}
        self._episode_lengths = {}
        self._episode_start_indices = {}
        self._all_episode_lengths = {}  # Store all episode lengths before filtering
        
        # Load episode metadata from parquet files
        episodes_dir = self.dataset_path / "meta" / "episodes"
        
        # First pass: collect all episode indices and lengths
        all_episodes = []
        for chunk_dir in sorted(episodes_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for parquet_file in sorted(chunk_dir.iterdir()):
                if not parquet_file.suffix == ".parquet":
                    continue
                episodes_df = pd.read_parquet(parquet_file)
                for _, row in episodes_df.iterrows():
                    episode_idx = int(row["episode_index"])
                    episode_length = int(row["length"])
                    all_episodes.append((episode_idx, episode_length))
                    self._all_episode_lengths[episode_idx] = episode_length
        
        # Sort by episode index for consistency
        all_episodes.sort(key=lambda x: x[0])
        all_episode_indices = [ep[0] for ep in all_episodes]
        
        # Compute train/val split
        if self.split != "all":
            # Deterministic shuffle for split
            rng = np.random.RandomState(self.split_seed)
            shuffled_indices = all_episode_indices.copy()
            rng.shuffle(shuffled_indices)
            
            # Split point
            num_val = max(1, int(len(shuffled_indices) * self.val_fraction))
            val_episodes = set(shuffled_indices[:num_val])
            train_episodes = set(shuffled_indices[num_val:])
            
            if self.split == "train":
                selected_episodes = train_episodes
            elif self.split == "val":
                selected_episodes = val_episodes
            else:
                raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'all'")
        else:
            selected_episodes = set(all_episode_indices)
        
        self._selected_episodes = selected_episodes
        
        # Second pass: build index only for selected episodes
        global_idx = 0
        for episode_idx, episode_length in all_episodes:
            if episode_idx not in selected_episodes:
                continue
            
            self._episode_lengths[episode_idx] = episode_length
            self._episode_start_indices[episode_idx] = global_idx
            
            for frame_idx in range(episode_length):
                self._step_to_episode_map[global_idx] = (episode_idx, frame_idx)
                global_idx += 1
        
        self._total_steps = global_idx
    
    def _load_parquet_data(self):
        """Load parquet data files for selected episodes into memory."""
        self._episode_data: dict[int, EpisodeData] = {}
        
        data_dir = self.dataset_path / "data"
        for chunk_dir in sorted(data_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for parquet_file in sorted(chunk_dir.iterdir()):
                if not parquet_file.suffix == ".parquet":
                    continue
                df = pd.read_parquet(parquet_file)
                
                # Group by episode_index and sort to ensure order
                for episode_idx, episode_df in sorted(df.groupby("episode_index")):
                    # Skip episodes not in our split
                    if episode_idx not in self._selected_episodes:
                        continue
                    
                    # Sort by frame_index
                    episode_df = episode_df.sort_values("frame_index").reset_index(drop=True)
                    
                    # Extract actions and states as numpy arrays
                    actions = np.stack(episode_df["action"].values).astype(np.float32)
                    states = np.stack(episode_df["observation.state"].values).astype(np.float32)
                    timestamps = episode_df["timestamp"].values.astype(np.float32)
                    task_indices = episode_df["task_index"].values.astype(np.int64)
                    global_indices = episode_df["index"].values.astype(np.int64)
                    
                    self._episode_data[episode_idx] = EpisodeData(
                        actions=actions,
                        states=states,
                        timestamps=timestamps,
                        task_index=int(task_indices[0]) if len(task_indices) > 0 else 0,
                        global_indices=global_indices,
                    )
    
    @property
    def num_episodes(self) -> int:
        """Return number of episodes in this split."""
        return len(self._selected_episodes)
    
    def _get_episode_chunk(self, episode_idx: int) -> int:
        """Get chunk index for an episode."""
        return episode_idx // self.chunks_size
    
    def _get_video_path(self, episode_idx: int, video_key: str) -> Path:
        """Get path to video file for an episode and camera.
        
        Note: In LeRobot v3.0, all episodes in a chunk share ONE video file.
        The file_index is 0 for the first file in each chunk.
        """
        chunk_idx = self._get_episode_chunk(episode_idx)
        # All episodes in a chunk are in file-000.mp4
        video_filename = self.video_path_pattern.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=0,  # Always 0 - one video file per chunk
        )
        return self.dataset_path / video_filename
    
    def __len__(self) -> int:
        """Return total number of samples (one per frame)."""
        return self._total_steps
    
    def __getitem__(self, idx: int) -> MimicVideoItem:
        """Get a single training sample.
        
        Args:
            idx: Global frame index (within this split)
            
        Returns:
            MimicVideoItem with past/future frames, actions, proprio, and command
        """
        # Map global index to episode and frame
        episode_idx, frame_idx = self._step_to_episode_map[idx]
        episode_length = self._episode_lengths[episode_idx]
        episode_data = self._episode_data[episode_idx]  # Dict lookup now
        
        # Compute past frame indices with stride
        past_indices = self._get_past_indices(frame_idx, episode_length)
        
        # Compute future frame indices with stride
        future_indices = self._get_future_indices(frame_idx, episode_length)
        
        # Load video frames (convert local indices to global video frame indices)
        past_global = episode_data.global_indices[past_indices]
        future_global = episode_data.global_indices[future_indices]
        
        cam_high_past = self._load_video_frames(episode_idx, past_global, self.video_key_high)
        wrist_cam_past = self._load_video_frames(episode_idx, past_global, self.video_key_wrist)
        cam_high_future = self._load_video_frames(episode_idx, future_global, self.video_key_high)
        wrist_cam_future = self._load_video_frames(episode_idx, future_global, self.video_key_wrist)
        
        # Get proprio at past timesteps
        proprio = self._get_proprio(episode_data, past_indices)
        
        # Get action chunk starting at current frame
        actions = self._get_action_chunk(episode_data, frame_idx, episode_length)
        
        # Get task description
        command = self._get_command(episode_data)
        
        return MimicVideoItem(
            cam_high_past=cam_high_past,
            wrist_cam_past=wrist_cam_past,
            cam_high_future=cam_high_future,
            wrist_cam_future=wrist_cam_future,
            proprio=proprio,
            actions=actions,
            command=command,
        )
    
    def _get_past_indices(self, frame_idx: int, episode_length: int) -> np.ndarray:
        """Compute past frame indices with temporal stride.
        
        Returns indices for frames at: t-(n-1)*stride, ..., t-stride, t
        where n = visual_context_length.
        
        Indices < 0 are clamped to 0 (repeat first frame).
        """
        # Generate indices going backwards from current frame
        offsets = np.arange(self.visual_context_length - 1, -1, -1) * self.temporal_stride
        indices = frame_idx - offsets
        
        # Clamp to valid range
        indices = np.clip(indices, 0, episode_length - 1)
        
        return indices
    
    def _get_future_indices(self, frame_idx: int, episode_length: int) -> np.ndarray:
        """Compute future frame indices with temporal stride.
        
        Returns indices for frames at: t+stride, t+2*stride, ..., t+n*stride
        where n = num_future_frames.
        
        Indices >= episode_length are clamped to episode_length-1 (repeat last frame).
        """
        # Generate indices going forward from current frame
        offsets = np.arange(1, self.num_future_frames + 1) * self.temporal_stride
        indices = frame_idx + offsets
        
        # Clamp to valid range
        indices = np.clip(indices, 0, episode_length - 1)
        
        return indices
    
    def _load_video_frames(
        self, 
        episode_idx: int, 
        frame_indices: np.ndarray, 
        video_key: str
    ) -> np.ndarray:
        """Load and resize video frames at specified global indices.
        
        Args:
            episode_idx: Episode index (used to find correct video file)
            frame_indices: Array of GLOBAL frame indices into the video file
            video_key: Camera key (e.g., "observation.images.front")
            
        Returns:
            Array of shape (T, image_size, image_size, 3)
        """
        # Get video path
        video_path = self._get_video_path(episode_idx, video_key)
        
        # Load frames using decord or imageio
        frames = self._decode_video_frames(video_path, frame_indices)
        
        # Resize frames
        resized_frames = np.stack([
            self._resize_frame(frame) for frame in frames
        ])
        
        return resized_frames
    
    def _decode_video_frames(self, video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
        """Decode specific frames from a video file.
        
        Uses caching to avoid re-opening video files repeatedly.
        
        Args:
            video_path: Path to video file
            frame_indices: Indices of frames to extract
            
        Returns:
            Array of shape (T, H, W, 3) with uint8 RGB frames
        """
        cache_key = str(video_path)
        
        try:
            import decord
            decord.bridge.set_bridge("numpy")
            
            # Check cache for VideoReader
            if cache_key not in self._video_cache:
                self._video_cache[cache_key] = decord.VideoReader(str(video_path))
            
            vr = self._video_cache[cache_key]
            frames = vr.get_batch(frame_indices).asnumpy()
        except ImportError:
            # Fallback to imageio - cache the entire decoded video
            if cache_key not in self._video_cache:
                import imageio.v3 as iio
                self._video_cache[cache_key] = iio.imread(str(video_path), plugin="pyav")
            
            video = self._video_cache[cache_key]
            frames = video[frame_indices]
        
        return frames
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a single frame to target size.
        
        Args:
            frame: Input frame of shape (H, W, 3)
            
        Returns:
            Resized frame of shape (image_size, image_size, 3)
        """
        img = Image.fromarray(frame)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img)
    
    def _get_proprio(self, episode_data: EpisodeData, frame_indices: np.ndarray) -> np.ndarray:
        """Get proprioceptive state at specified frame indices.
        
        Args:
            episode_data: Episode data
            frame_indices: Array of frame indices
            
        Returns:
            Array of shape (T, state_dim)
        """
        return episode_data.states[frame_indices].astype(np.float32)
    
    def _get_action_chunk(
        self, 
        episode_data: EpisodeData, 
        frame_idx: int, 
        episode_length: int
    ) -> np.ndarray:
        """Get chunk of consecutive actions starting at frame_idx.
        
        If chunk extends beyond episode, pad with last action.
        
        Args:
            episode_data: Episode data
            frame_idx: Starting frame index
            episode_length: Total episode length
            
        Returns:
            Array of shape (chunk_size, action_dim)
        """
        actions = episode_data.actions
        
        # How many actions we can get before end of episode
        remaining = episode_length - frame_idx
        
        if remaining >= self.chunk_size:
            # Full chunk available
            action_chunk = actions[frame_idx:frame_idx + self.chunk_size]
        else:
            # Need to pad with last action
            available = actions[frame_idx:]
            padding_needed = self.chunk_size - remaining
            padding = np.tile(actions[-1], (padding_needed, 1))
            action_chunk = np.concatenate([available, padding], axis=0)
        
        return action_chunk.astype(np.float32)
    
    def _get_command(self, episode_data: EpisodeData) -> str:
        """Get task description for an episode.
        
        In LeRobot v3.0, the tasks.parquet has task descriptions as the index
        and task_index as a column.
        
        Args:
            episode_data: Episode data
            
        Returns:
            Task description string
        """
        if self.tasks_df is None:
            return ""
        
        task_idx = episode_data.task_index
        # Find the row where task_index matches
        matching_rows = self.tasks_df[self.tasks_df["task_index"] == task_idx]
        if len(matching_rows) > 0:
            # Task description is the index (row label)
            return str(matching_rows.index[0])
        return ""
