"""Shared pytest fixtures for the non-audit test suite.

These fixtures are immutable dataclasses, safe to share across the whole
session.
"""

import pytest

from cv_quixer.config.schema import DataConfig


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    """Canonical 28×28 image / 7×7 patch config used by most tests."""
    return DataConfig(image_size=28, patch_size=7, num_classes=10)


@pytest.fixture(scope="session")
def tiny_data_config() -> DataConfig:
    """Smaller 14×14 image / 7×7 patch → 4 patches per forward pass.

    Use for shape/finiteness/gradient-flow tests where the specific patch
    count doesn't matter — about 4× cheaper per CVQuixer forward pass than
    the canonical 16-patch config.
    """
    return DataConfig(image_size=14, patch_size=7, num_classes=10)
