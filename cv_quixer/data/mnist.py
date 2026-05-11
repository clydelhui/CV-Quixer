"""Generic patched dataset loader for MNIST-family datasets."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from cv_quixer.config.schema import DataConfig
from cv_quixer.data.transforms import extract_patches

_DATASET_REGISTRY: dict[str, type] = {
    "mnist":        datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
}


def _load_or_compute_stats(
    cls: type,
    data_root: str,
    dataset_name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (mean, std), loading from cache or computing from training split.

    Cache file: {data_root}/{dataset_name}_norm_stats.json
    """
    cache_path = Path(data_root) / f"{dataset_name}_norm_stats.json"

    if cache_path.exists():
        stored = json.loads(cache_path.read_text())
        return tuple(stored["mean"]), tuple(stored["std"])

    raw = cls(root=data_root, train=True, download=True)
    data = raw.data.float() / 255.0   # (N, H, W), [0, 1]
    mean = (float(data.mean()),)
    std  = (float(data.std()),)

    cache_path.write_text(json.dumps({"mean": list(mean), "std": list(std)}, indent=2))
    return mean, std


class PatchedDataset(Dataset):
    """Wraps any registered torchvision dataset to return image patches."""

    def __init__(self, config: DataConfig, train: bool = True) -> None:
        dataset_name = config.dataset.lower()
        if dataset_name not in _DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{config.dataset}'. "
                f"Supported: {sorted(_DATASET_REGISTRY)}"
            )
        cls = _DATASET_REGISTRY[dataset_name]

        if config.normalize:
            mean, std = _load_or_compute_stats(cls, config.data_root, dataset_name)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = cls(root=config.data_root, train=train,
                           download=True, transform=transform)
        self.patch_size = config.patch_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        patches = extract_patches(image.unsqueeze(0), self.patch_size)
        return patches.squeeze(0), label


def get_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    """Return train and test DataLoaders for the dataset specified in config."""
    Path(config.data_root).mkdir(parents=True, exist_ok=True)
    train_dataset = PatchedDataset(config, train=True)
    test_dataset  = PatchedDataset(config, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
