"""Mini experiment: 10× parameter scale, 200 train / 50 test, 100 epochs.

Compared to smoke_test.py:
  - ~13,760 trainable parameters
  - cutoff_dim raised from 4 → 6 for greater Fock-space expressivity
  - num_heads 2 → 4, decoder_hidden_dim 16 → 32
  - cnn_channels_2 auto-scaled via target_params=13760
  - patch_size=7 (16 patches per image), CNN hypernetwork replaces MLP
  - 200 train / 50 test FashionMNIST samples, batch_size=32
  - 100 epochs; per-epoch train + test loss / accuracy tracked

Outputs:
  results/figures/mini_experiment_curve.png                     — two-panel training curve
  results/logs/mini_experiment/history.json                     — per-epoch metrics
  results/checkpoints/mini_experiment/final_model.pt            — final model weights
  results/checkpoints/mini_experiment/checkpoint_epoch_NNNN.pt  — periodic checkpoints

Run (fresh):
    uv run python experiments/mini_experiment.py

Resume from checkpoint:
    uv run python experiments/mini_experiment.py \\
        --resume results/checkpoints/mini_experiment/checkpoint_epoch_0050.pt
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv_quixer.config.schema import DataConfig, ExperimentConfig, QuantumConfig, TrainingConfig
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.models import build_model
from cv_quixer.utils import print_parameter_table

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCHS = 100
TRAIN_SIZE = 200
TEST_SIZE = 50
BATCH_SIZE = 32
TARGET_PARAMS = 13_760
CHECKPOINT_INTERVAL = 10

_parser = argparse.ArgumentParser()
_parser.add_argument("--resume", type=str, default=None,
                     help="path to checkpoint .pt file to resume from")
_parser.add_argument("--epochs", type=int, default=None,
                     help=f"override the default EPOCHS={EPOCHS} (useful for short regression runs)")
_args = _parser.parse_args()

if _args.epochs is not None:
    EPOCHS = _args.epochs

data_cfg = DataConfig(
    dataset="fashionmnist",
    normalize=True,
    patch_size=7,
    batch_size=BATCH_SIZE,
    num_workers=0,
    data_root="data/",
)
quantum_cfg = QuantumConfig(
    num_modes=2,
    cutoff_dim=6,
    num_heads=4,
    cnn_channels_1=8,
    cnn_channels_2=16,          # overridden by target_params auto-scaling
    cnn_kernel_size=3,
    decoder_hidden_dim=32,
    poly_degree=2,
    dtype="complex64",
    trunc_penalty="norm",
    trunc_lambda=0.01,
    target_params=TARGET_PARAMS,
)
config = ExperimentConfig(
    name="mini_experiment",
    model="quantum",
    data=data_cfg,
    quantum=quantum_cfg,
    training=TrainingConfig(lr=1e-3, epochs=EPOCHS, seed=42),
    use_wandb=False,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}\n")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

train_ds = Subset(PatchedDataset(data_cfg, train=True),  indices=list(range(TRAIN_SIZE)))
test_ds  = Subset(PatchedDataset(data_cfg, train=False), indices=list(range(TEST_SIZE)))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=TEST_SIZE,  shuffle=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = build_model(config).to(device)
print_parameter_table(model)
n_params = model.get_num_parameters()
print(f"\nTrainable parameters: {n_params:,}  (target: {TARGET_PARAMS:,})\n")

# Shape check
_patches, _ = next(iter(train_loader))
_logits = model(_patches.to(device))
assert _logits.shape == (BATCH_SIZE, 10), f"Expected ({BATCH_SIZE}, 10), got {_logits.shape}"
print(f"Forward shape OK — logits: {tuple(_logits.shape)}\n")

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

# ---------------------------------------------------------------------------
# Directories (created before training so checkpoints can be written mid-run)
# ---------------------------------------------------------------------------

log_dir  = Path("results/logs/mini_experiment")
ckpt_dir = Path("results/checkpoints/mini_experiment")
fig_dir  = Path("results/figures")
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Resume from checkpoint (if requested)
# ---------------------------------------------------------------------------

history: dict[str, list] = {
    "train_loss": [], "train_acc": [],
    "test_loss":  [], "test_acc":  [],
    "trunc_loss": [],
}

start_epoch = 1
if _args.resume:
    ckpt = torch.load(_args.resume, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt["history"]
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from {_args.resume}  (continuing from epoch {start_epoch})\n")


def train_epoch(epoch: int) -> tuple[float, float, float]:
    model.train()
    total_ce, total_trunc, correct, total = 0.0, 0.0, 0, 0
    for patches, labels in tqdm(
        train_loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", leave=False, unit="batch"
    ):
        patches, labels = patches.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, trunc_loss = model(patches, return_trunc_loss=True)
        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss + quantum_cfg.trunc_lambda * trunc_loss
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_ce    += ce_loss.item() * n
        total_trunc += trunc_loss.item() * n
        correct     += (logits.argmax(dim=-1) == labels).sum().item()
        total       += n

    return total_ce / total, total_trunc / total, correct / total


@torch.no_grad()
def evaluate() -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for patches, labels in test_loader:
        patches, labels = patches.to(device), labels.to(device)
        logits = model(patches)
        total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"{'Epoch':<7} {'Train loss':<12} {'Train acc':<11} "
      f"{'Test loss':<11} {'Test acc':<11} {'Trunc loss':<12} {'Time'}")
print("─" * 76)

run_start = time.time()

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()
    train_loss, trunc_loss, train_acc = train_epoch(epoch)
    test_loss, test_acc = evaluate()
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["trunc_loss"].append(trunc_loss)

    print(f"{epoch:<7} {train_loss:<12.4f} {train_acc:<11.3f} "
          f"{test_loss:<11.4f} {test_acc:<11.3f} {trunc_loss:<12.4f} {elapsed:.1f}s")

    if epoch % CHECKPOINT_INTERVAL == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt")

total_runtime = time.time() - run_start

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

# JSON log
with open(log_dir / "history.json", "w") as f:
    json.dump(history, f, indent=2)

# Final model weights
torch.save(
    {"model_state_dict": model.state_dict(), "history": history},
    ckpt_dir / "final_model.pt",
)

# Training curve
epochs_x = list(range(1, len(history["train_loss"]) + 1))
fig, (ax_loss, ax_acc, ax_trunc) = plt.subplots(1, 3, figsize=(18, 4))

ax_loss.plot(epochs_x, history["train_loss"], label="train loss")
ax_loss.plot(epochs_x, history["test_loss"],  label="test loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("Loss")
ax_loss.legend()

ax_acc.plot(epochs_x, history["train_acc"], label="train acc")
ax_acc.plot(epochs_x, history["test_acc"],  label="test acc")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy")
ax_acc.legend()

ax_trunc.plot(epochs_x, history["trunc_loss"], label="trunc loss", color="tab:orange")
ax_trunc.set_xlabel("Epoch")
ax_trunc.set_ylabel("Avg per-patch truncation loss")
ax_trunc.set_title("Truncation Loss")
ax_trunc.legend()

fig.suptitle(f"CV-Quixer mini experiment  |  {n_params:,} params  |  {TRAIN_SIZE} train samples")
fig.tight_layout()
fig.savefig(fig_dir / "mini_experiment_curve.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

best_test_acc = max(history["test_acc"])
best_epoch    = history["test_acc"].index(best_test_acc) + 1

print(f"\nTotal runtime:    {total_runtime / 60:.1f} min")
print(f"Best test acc:    {best_test_acc:.4f} (epoch {best_epoch})")
print(f"Logs saved       →  {log_dir}/")
print(f"Checkpoints saved →  {ckpt_dir}/")
print(f"Curve saved      →  {fig_dir}/mini_experiment_curve.png")
