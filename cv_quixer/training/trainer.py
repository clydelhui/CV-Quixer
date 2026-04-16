"""Generic Trainer — model-agnostic training loop used by both models.

Accepting any BaseVisionTransformer ensures the classical and quantum models
are trained under identical conditions, which is essential for a fair
experimental comparison in the thesis.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.utils.reproducibility import set_seed


class Trainer:
    """Trains and evaluates a BaseVisionTransformer on MNIST patches.

    Args:
        model:       The model to train.
        config:      Full ExperimentConfig (training section used here).
        train_loader: DataLoader yielding (patches, labels) batches.
        test_loader:  DataLoader yielding (patches, labels) batches.
    """

    def __init__(
        self,
        model: BaseVisionTransformer,
        config: ExperimentConfig,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(config.training.seed)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

        self.checkpoint_dir = Path(config.training.checkpoint_dir) / config.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [],
        }

    def train(self) -> dict[str, list]:
        """Run the full training loop.

        Returns:
            history dict with train/test loss and accuracy per epoch.
        """
        for epoch in range(1, self.config.training.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(epoch)
            test_loss, test_acc = self._evaluate()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)

            print(
                f"Epoch {epoch:03d}/{self.config.training.epochs} "
                f"| train loss {train_loss:.4f} acc {train_acc:.3f} "
                f"| test loss {test_loss:.4f} acc {test_acc:.3f} "
                f"| {elapsed:.1f}s"
            )

            self._save_checkpoint(epoch, test_acc)

        return self.history

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        log_interval = self.config.training.log_interval

        for batch_idx, (patches, labels) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        ):
            patches = patches.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(patches)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for patches, labels in self.test_loader:
            patches = patches.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(patches)
            total_loss += self.criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, test_acc: float) -> None:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_acc{test_acc:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.training
        if tc.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
            )
        elif tc.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer '{tc.optimizer}'")
