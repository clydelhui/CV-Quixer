from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.classical.vit import ClassicalViT
from cv_quixer.models.quantum.cv_quixer import CVQuixer, SharedCVQuixer
from cv_quixer.models.quantum.cv_seq2seq import StackedCVQuixer


def build_model(config: ExperimentConfig) -> BaseVisionTransformer:
    """Instantiate the correct model from an ExperimentConfig.

    This is the single place where the model string is resolved to a concrete
    class. Everything else in the codebase is polymorphic via
    BaseVisionTransformer.

      - "quantum"         — CVQuixer (per-head CNN hypernetworks)
      - "quantum_shared"  — SharedCVQuixer (shared patch CNN + per-head linears)
      - "quantum_stacked" — StackedCVQuixer (seq-to-seq blocks, ADR-0002)
      - "classical"       — ClassicalViT

    Args:
        config: Fully populated ExperimentConfig.

    Returns:
        An instantiated BaseVisionTransformer subclass.
    """
    if config.model == "quantum":
        return CVQuixer(config.quantum, config.data)
    elif config.model == "quantum_shared":
        return SharedCVQuixer(config.quantum, config.data)
    elif config.model == "quantum_stacked":
        return StackedCVQuixer(config.quantum, config.data)
    elif config.model == "classical":
        return ClassicalViT(config.classical, config.data)
    else:
        raise ValueError(
            f"Unknown model '{config.model}'. Expected 'quantum', "
            "'quantum_shared', 'quantum_stacked', or 'classical'."
        )
