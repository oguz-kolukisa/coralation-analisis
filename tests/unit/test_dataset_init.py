"""Tests for dataset.py __init__ field detection branches."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_dataset(field_names):
    """Create a mock HF dataset with given field names in features."""
    ds = MagicMock()
    features = MagicMock()
    features.__contains__ = lambda self, key: key in field_names
    features.keys.return_value = list(field_names)
    ds.features = features
    return ds


def _patch_imports_and_create(features, hf_token=None):
    """Patch all external imports and create an ImageNetSampler."""
    mock_datasets = MagicMock()
    mock_ds = _mock_dataset(features)
    mock_datasets.load_dataset.return_value = mock_ds
    mock_datasets.disable_progress_bar = MagicMock()

    mock_hf_hub = MagicMock()

    with patch.dict(sys.modules, {
        "datasets": mock_datasets,
        "huggingface_hub": mock_hf_hub,
    }):
        # Need to reimport after patching
        from src.dataset import ImageNetSampler
        sampler = ImageNetSampler(
            dataset_name="test", split="test",
            hf_token=hf_token, max_scan=10, seed=42,
        )
        return sampler, mock_hf_hub


class TestInitFieldDetection:
    def test_cifar100_fields(self):
        sampler, _ = _patch_imports_and_create({"fine_label", "img"})
        assert sampler._label_field == "fine_label"
        assert sampler._image_field == "img"

    def test_imagenet_fields(self):
        sampler, _ = _patch_imports_and_create({"label", "image"})
        assert sampler._label_field == "label"
        assert sampler._image_field == "image"

    def test_unknown_label_field_raises(self):
        with pytest.raises(ValueError, match="label field"):
            _patch_imports_and_create({"unknown_col", "image"})

    def test_unknown_image_field_raises(self):
        with pytest.raises(ValueError, match="image field"):
            _patch_imports_and_create({"label", "unknown_col"})

    def test_hf_token_triggers_login(self):
        sampler, mock_hf_hub = _patch_imports_and_create(
            {"label", "image"}, hf_token="fake_token"
        )
        mock_hf_hub.login.assert_called_once()
