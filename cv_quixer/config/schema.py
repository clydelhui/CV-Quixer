from dataclasses import dataclass, field


@dataclass
class DataConfig:
    image_size: int = 28
    patch_size: int = 4          # must divide image_size evenly
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 2
    data_root: str = "data/"


@dataclass
class QuantumConfig:
    num_modes: int = 8           # number of bosonic modes (qumodes)
    num_layers: int = 4          # number of CV transformer layers
    cutoff_dim: int = 10         # Fock space truncation dimension
    backend: str = "strawberryfields.fock"  # PennyLane device string


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
