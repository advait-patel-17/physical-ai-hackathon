import pytest
import numpy as np
from dataclasses import asdict
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from hackathon.dataset import MimicVideoDataset, MimicVideoItem

# Download and cache the dataset from Hugging Face Hub
REPO_ID = "advpatel/vam-test"
DATASET_PATH = snapshot_download(repo_id=REPO_ID, repo_type="dataset")


def collate_mimic_items(batch: list[MimicVideoItem]) -> dict:
    """Custom collate function for MimicVideoItem dataclass."""
    return {
        key: np.stack([asdict(item)[key] for item in batch if not isinstance(asdict(item)[key], str)])
        if not isinstance(asdict(batch[0])[key], str)
        else [asdict(item)[key] for item in batch]
        for key in asdict(batch[0]).keys()
    }


class TestMimicVideoDataset:

    def test_dataset_loads(self):
        """Dataset initializes without error and has correct length."""
        dataset = MimicVideoDataset(DATASET_PATH)
        assert len(dataset) == 2128  # total_frames from info.json

    def test_getitem_returns_mimic_video_item(self):
        """__getitem__ returns MimicVideoItem dataclass."""
        dataset = MimicVideoDataset(DATASET_PATH)
        item = dataset[0]
        assert isinstance(item, MimicVideoItem)

    def test_image_shapes(self):
        """Images are resized to 224x224."""
        dataset = MimicVideoDataset(
            DATASET_PATH,
            visual_context_length=4,
            num_future_frames=3,
            image_size=224,
        )
        item = dataset[100]
        assert item.cam_high_past.shape == (4, 224, 224, 3)
        assert item.wrist_cam_past.shape == (4, 224, 224, 3)
        assert item.cam_high_future.shape == (3, 224, 224, 3)
        assert item.wrist_cam_future.shape == (3, 224, 224, 3)

    def test_action_shape(self):
        """Actions have correct chunk size and dimension."""
        dataset = MimicVideoDataset(DATASET_PATH, chunk_size=16)
        item = dataset[100]
        assert item.actions.shape == (16, 6)  # 6-dim actions

    def test_proprio_shape(self):
        """Proprio matches visual context length."""
        dataset = MimicVideoDataset(DATASET_PATH, visual_context_length=4)
        item = dataset[100]
        assert item.proprio.shape == (4, 6)  # 6-dim proprio

    def test_padding_at_start(self):
        """First frames of episode pad correctly (repeat first frame)."""
        dataset = MimicVideoDataset(
            DATASET_PATH,
            visual_context_length=4,
            temporal_stride=2,
        )
        item = dataset[0]  # First frame of first episode
        # Past frames should all be the same (padded with first frame)
        assert np.allclose(item.cam_high_past[0], item.cam_high_past[1])

    def test_padding_at_end(self):
        """Last frames of episode pad correctly (repeat last frame)."""
        dataset = MimicVideoDataset(
            DATASET_PATH,
            num_future_frames=3,
            chunk_size=16,
        )
        # Get a sample near the end of an episode
        item = dataset[len(dataset) - 1]
        # Should not raise, padding handles overflow
        assert item.actions.shape[0] == 16

    def test_dataloader_compatible(self):
        """Dataset works with PyTorch DataLoader."""
        dataset = MimicVideoDataset(DATASET_PATH)
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True,
            collate_fn=collate_mimic_items,
        )
        batch = next(iter(dataloader))
        # Batch should have 4 samples
        assert batch["cam_high_past"].shape[0] == 4

    def test_temporal_stride(self):
        """Temporal stride spaces frames correctly."""
        dataset = MimicVideoDataset(
            DATASET_PATH,
            visual_context_length=3,
            temporal_stride=5,
        )
        # With stride=5 and context=3, we sample t-10, t-5, t
        # This test verifies frames are not identical (unless at boundary)
        item = dataset[50]  # Mid-episode sample
        # Frames should be different due to stride
        assert not np.allclose(item.cam_high_past[0], item.cam_high_past[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
