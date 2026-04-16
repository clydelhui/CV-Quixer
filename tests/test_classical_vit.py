"""Tests for the classical ViT model."""

import torch
import pytest

from cv_quixer.config.schema import ClassicalConfig, DataConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.classical.vit import ClassicalViT


@pytest.fixture
def data_config():
    return DataConfig(image_size=28, patch_size=4, num_classes=10)


@pytest.fixture
def classical_config():
    return ClassicalConfig(embed_dim=32, num_heads=4, num_layers=2)


@pytest.fixture
def model(classical_config, data_config):
    return ClassicalViT(classical_config, data_config)


class TestClassicalViT:
    def test_is_base_model(self, model):
        assert isinstance(model, BaseVisionTransformer)

    def test_forward_shape(self, model):
        patches = torch.randn(4, 49, 16)   # batch=4, 49 patches, dim=16
        logits = model(patches)
        assert logits.shape == (4, 10)

    def test_no_nan_in_output(self, model):
        patches = torch.randn(2, 49, 16)
        logits = model(patches)
        assert not torch.isnan(logits).any()

    def test_get_num_parameters(self, model):
        n = model.get_num_parameters()
        assert isinstance(n, int) and n > 0

    def test_predict_shape(self, model):
        patches = torch.randn(3, 49, 16)
        preds = model.predict(patches)
        assert preds.shape == (3,)
        assert preds.dtype == torch.int64
