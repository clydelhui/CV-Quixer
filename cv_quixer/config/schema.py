from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset: str = "fashionmnist"  # "fashionmnist" | "mnist"
    normalize: bool = True         # compute & cache stats on first load; False → ToTensor only
    image_size: int = 28
    patch_size: int = 4            # must divide image_size evenly
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 2
    data_root: str = "data/"


@dataclass
class QuantumConfig:
    num_modes: int = 4           # number of bosonic modes (qumodes)
    num_layers: int = 4          # reserved — not read until multi-layer stacking is implemented
    cutoff_dim: int = 6          # Fock space truncation (memory ~ cutoff_dim^num_modes)
    grad_mode: str = "backprop"          # "backprop" | "parameter_shift"
    param_shift_shift: float = 1.5708   # shift s for PSR (default π/2)
    bs_topology: str = "linear"         # "linear" | "ring"
    dtype: str = "complex128"           # "complex64" | "complex128"

    # Multi-head attention
    num_heads: int = 4            # parallel CV attention heads
    decoder_hidden_dim: int = 64  # hidden dim of readout→logits MLP

    # CNN hypernetwork: Conv(1→C1) → Conv(C1→C2) → flatten → 2D PE → Linear → gate params
    # No padding; spatial output: h_out = patch_size - 2*(cnn_kernel_size - 1)
    cnn_channels_1: int = 8    # output channels of first conv layer
    cnn_channels_2: int = 16   # output channels of second conv layer (auto-scaled if target_params > 0)
    cnn_kernel_size: int = 3   # kernel size for both conv layers

    # Matrix polynomial degree for LCU attention (P(M) = Σ c_j M^j, j=0..d)
    poly_degree: int = 2           # keep ≤ 4; d=2 or d=3 recommended

    # Auto-scaling: set > 0 to auto-adjust cnn_channels_2 to hit this budget
    target_params: int = -1

    # Fock truncation penalty added to training loss
    trunc_penalty: str = "none"   # "none" | "norm" | "photon_number"
    trunc_lambda: float = 0.01

    # Quantum-circuit readout used as input to the classical decoder.
    #   "quadrature_x"     — 〈x̂〉 per mode                       (num_modes scalars)
    #   "photon_number"    — 〈n̂〉 per mode                       (num_modes scalars)
    #   "pnr_distribution" — P(n_mode=k) for k=0..cutoff_dim-1   (num_modes × cutoff_dim values)
    readout_observable: str = "quadrature_x"

    def __post_init__(self) -> None:
        valid_readouts = {"quadrature_x", "photon_number", "pnr_distribution"}
        if self.readout_observable not in valid_readouts:
            raise ValueError(
                f"readout_observable must be one of {sorted(valid_readouts)}, "
                f"got {self.readout_observable!r}"
            )


@dataclass
class ClassicalConfig:
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 30
    optimizer: str = "adam"      # "adam" or "sgd"
    weight_decay: float = 1e-4
    seed: int = 42
    checkpoint_dir: str = "results/checkpoints/"
    log_dir: str = "results/logs/"
    log_interval: int = 10       # log every N batches


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    model: str = "quantum"       # "quantum" or "classical"
    data: DataConfig = field(default_factory=DataConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_wandb: bool = False
