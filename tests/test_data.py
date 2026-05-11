"""Tests for the data pipeline."""

import torch
import pytest

from cv_quixer.config.schema import DataConfig
from cv_quixer.data.transforms import extract_patches
from cv_quixer.data.mnist import PatchedDataset, get_dataloaders


class TestExtractPatches:
    def test_output_shape(self):
        # 28x28 image, patch_size=4 → 49 patches of dim 16
        images = torch.zeros(2, 1, 28, 28)
        patches = extract_patches(images, patch_size=4)
        assert patches.shape == (2, 49, 16)

    def test_patch_size_7(self):
        # 28x28 image, patch_size=7 → 16 patches of dim 49
        images = torch.zeros(1, 1, 28, 28)
        patches = extract_patches(images, patch_size=7)
        assert patches.shape == (1, 16, 49)

    def test_invalid_patch_size_raises(self):
        images = torch.zeros(1, 1, 28, 28)
        with pytest.raises(AssertionError):
            extract_patches(images, patch_size=5)  # 28 not divisible by 5


class TestPatchedDataset:
    @pytest.fixture
    def config(self, tmp_path):
        return DataConfig(dataset="fashionmnist", patch_size=4, data_root=str(tmp_path))

    def test_item_shapes(self, config):
        dataset = PatchedDataset(config, train=True)
        patches, label = dataset[0]
        assert patches.shape == (49, 16)
        assert isinstance(label, int)

    def test_length(self, config):
        dataset = PatchedDataset(config, train=True)
        assert len(dataset) == 60_000

    def test_test_split_length(self, config):
        dataset = PatchedDataset(config, train=False)
        assert len(dataset) == 10_000

    def test_unknown_dataset_raises(self, tmp_path):
        config = DataConfig(dataset="imagenet", data_root=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown dataset"):
            PatchedDataset(config, train=True)

    def test_normalize_false_range(self, tmp_path):
        config = DataConfig(dataset="fashionmnist", normalize=False,
                            patch_size=4, data_root=str(tmp_path))
        dataset = PatchedDataset(config, train=True)
        patches, _ = dataset[0]
        assert patches.min() >= 0.0
        assert patches.max() <= 1.0

    def test_stats_cache_written(self, config):
        import pathlib
        PatchedDataset(config, train=True)
        cache = pathlib.Path(config.data_root) / "fashionmnist_norm_stats.json"
        assert cache.exists()


class TestGetDataloaders:
    def test_batch_shapes(self, tmp_path):
        config = DataConfig(dataset="fashionmnist", patch_size=4,
                            batch_size=8, data_root=str(tmp_path))
        train_loader, test_loader = get_dataloaders(config)
        patches, labels = next(iter(train_loader))
        assert patches.shape == (8, 49, 16)
        assert labels.shape == (8,)
