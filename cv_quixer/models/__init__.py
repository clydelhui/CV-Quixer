from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.classical.vit import ClassicalViT
from cv_quixer.models.quantum.cv_quixer import CVQuixer


def build_model(config: ExperimentConfig) -> BaseVisionTransformer:
    """Instantiate the correct model from an ExperimentConfig.

    This is the single place where the string "quantum" / "classical" is
    resolved to a concrete class. Everything else in the codebase is
    polymorphic via BaseVisionTransformer.

    Args:
        config: Fully populated ExperimentConfig.

    Returns:
        An instantiated BaseVisionTransformer subclass.
    """
    if config.model == "quantum":
        return CVQuixer(config.quantum, config.data)
    elif config.model == "classical":
        return ClassicalViT(config.classical, config.data)
    else:
        raise ValueError(
            f"Unknown model '{config.model}'. Expected 'quantum' or 'classical'."
        )
