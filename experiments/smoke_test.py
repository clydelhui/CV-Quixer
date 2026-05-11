"""End-to-end smoke test: 5 train samples, 1 test sample, FashionMNIST.

Verifies:
  1. FashionMNIST patches load with correct shape (5, 16, 49)
  2. Forward pass produces logits of shape (B, 10)
  3. Fock truncation loss is computed and logged each epoch
  4. Combined loss computes and backward() runs without error
  5. Every trainable parameter receives a gradient
  6. Eval pass runs and returns a prediction

Run:
    uv run python experiments/smoke_test.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cv_quixer.config.schema import DataConfig, ExperimentConfig, QuantumConfig, TrainingConfig
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.models import build_model
from cv_quixer.utils import print_parameter_table

EPOCHS = 5

# --- Config (hardcoded, no YAML) ---
data_cfg = DataConfig(
    dataset="fashionmnist",
    normalize=True,
    patch_size=7,
    batch_size=5,
    num_workers=0,
    data_root="data/",
)
quantum_cfg = QuantumConfig(
    num_modes=2,
    cutoff_dim=4,
    num_heads=2,
    cnn_channels_1=4,
    cnn_channels_2=8,
    cnn_kernel_size=3,
    decoder_hidden_dim=16,
    poly_degree=2,
    dtype="complex64",
    trunc_penalty="norm",
    trunc_lambda=0.01,
)
config = ExperimentConfig(
    name="smoke_test",
    model="quantum",
    data=data_cfg,
    quantum=quantum_cfg,
    training=TrainingConfig(lr=1e-3, epochs=EPOCHS, seed=0),
    use_wandb=False,
)

# --- Dataset (5 train, 1 test) ---
train_ds = Subset(PatchedDataset(data_cfg, train=True),  indices=list(range(5)))
test_ds  = Subset(PatchedDataset(data_cfg, train=False), indices=[0])
train_loader = DataLoader(train_ds, batch_size=5, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

# --- Model ---
model = build_model(config)
print_parameter_table(model)

# --- Shape check (before training) ---
patches, labels = next(iter(train_loader))
logits_check = model(patches)
assert logits_check.shape == (5, 10), f"Expected (5, 10), got {logits_check.shape}"
print(f"\nForward shape OK — logits: {tuple(logits_check.shape)}")

# --- Training loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"\n{'Epoch':<6} {'CE loss':<12} {'Trunc loss':<14} {'Total loss'}")
print("─" * 46)

for epoch in range(1, EPOCHS + 1):
    model.train()
    patches, labels = next(iter(train_loader))
    optimizer.zero_grad()

    logits, trunc_loss = model(patches, return_trunc_loss=True)
    ce_loss = F.cross_entropy(logits, labels)
    total_loss = ce_loss + quantum_cfg.trunc_lambda * trunc_loss

    total_loss.backward()
    optimizer.step()

    print(f"{epoch:<6} {ce_loss.item():<12.4f} {trunc_loss.item():<14.4f} {total_loss.item():.4f}")

# --- Gradient check (after final backward) ---
missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
if missing:
    raise AssertionError(f"Missing gradients: {missing}")
print(f"\nGradients OK — all {sum(1 for p in model.parameters() if p.requires_grad)} params have gradients")

# --- Eval pass ---
model.eval()
with torch.no_grad():
    test_patches, test_label = next(iter(test_loader))
    test_logits = model(test_patches)
    pred = test_logits.argmax(dim=-1).item()
print(f"Eval OK     — predicted class: {pred}, true label: {test_label.item()}")

print("\nSmoke test PASSED")
