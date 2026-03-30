"""Tests for pluggable classifier registry and model loading."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.classifier import (
    ImageNetClassifier, _MODEL_REGISTRY, available_classifiers,
)


class TestModelRegistry:
    def test_resnet50_in_registry(self):
        assert "resnet50" in _MODEL_REGISTRY

    def test_dinov2_in_registry(self):
        assert "dinov2_vitb14_lc" in _MODEL_REGISTRY

    def test_vit_l_16_in_registry(self):
        assert "vit_l_16" in _MODEL_REGISTRY

    def test_available_classifiers(self):
        names = available_classifiers()
        assert "resnet50" in names
        assert "dinov2_vitb14_lc" in names
        assert "vit_l_16" in names

    def test_registry_has_required_keys(self):
        for name, spec in _MODEL_REGISTRY.items():
            assert "loader" in spec, f"{name} missing loader"
            assert "target_layer" in spec, f"{name} missing target_layer"
            assert "arch" in spec, f"{name} missing arch"
            assert "preprocess" in spec, f"{name} missing preprocess"
            assert spec["arch"] in ("cnn", "vit"), f"{name} has invalid arch"


class TestUnknownModel:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            ImageNetClassifier(model_name="nonexistent_model", device="cpu")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="resnet50"):
            ImageNetClassifier(model_name="bad", device="cpu")


class TestResNet50Loading:
    @patch("src.classifier.models.resnet50")
    @patch("src.classifier.get_attention_generator")
    def test_loads_resnet50_weights(self, mock_attn, mock_resnet):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_resnet.return_value = mock_model

        clf = ImageNetClassifier(model_name="resnet50", device="cpu")
        mock_resnet.assert_called_once()
        assert clf.loaded is True
        assert len(clf.labels) == 1000

    @patch("src.classifier.models.resnet50")
    @patch("src.classifier.get_attention_generator")
    def test_passes_target_layer_to_attention(self, mock_attn, mock_resnet):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_resnet.return_value = mock_model

        ImageNetClassifier(model_name="resnet50", device="cpu", attention_method="gradcam")
        mock_attn.assert_called_once_with("gradcam", target_layer="layer4.2", arch="cnn")


class TestDINOv2Loading:
    @patch("src.classifier.torch.hub.load")
    @patch("src.classifier.get_attention_generator")
    def test_loads_dinov2_from_hub(self, mock_attn, mock_hub):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_hub.return_value = mock_model

        clf = ImageNetClassifier(model_name="dinov2_vitb14_lc", device="cpu")
        mock_hub.assert_called_once_with(
            "facebookresearch/dinov2", "dinov2_vitb14_lc", pretrained=True,
        )
        assert clf.loaded is True
        assert len(clf.labels) == 1000

    @patch("src.classifier.torch.hub.load")
    @patch("src.classifier.get_attention_generator")
    def test_dinov2_uses_vit_arch(self, mock_attn, mock_hub):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_hub.return_value = mock_model

        ImageNetClassifier(model_name="dinov2_vitb14_lc", device="cpu")
        mock_attn.assert_called_once_with(
            "gradcam++", target_layer="backbone.blocks.11", arch="vit",
        )


class TestViTL16Loading:
    @patch("src.classifier.models.vit_l_16")
    @patch("src.classifier.get_attention_generator")
    def test_loads_vit_l_16_swag(self, mock_attn, mock_vit):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_vit.return_value = mock_model

        clf = ImageNetClassifier(model_name="vit_l_16", device="cpu")
        mock_vit.assert_called_once()
        assert clf.loaded is True
        assert len(clf.labels) == 1000

    @patch("src.classifier.models.vit_l_16")
    @patch("src.classifier.get_attention_generator")
    def test_vit_l_16_uses_vit_arch(self, mock_attn, mock_vit):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_vit.return_value = mock_model

        ImageNetClassifier(model_name="vit_l_16", device="cpu")
        mock_attn.assert_called_once_with(
            "gradcam++", target_layer="encoder.layers.encoder_layer_23", arch="vit",
        )

    @patch("src.classifier.models.vit_l_16")
    @patch("src.classifier.get_attention_generator")
    def test_vit_l_16_uses_512_preprocess(self, mock_attn, mock_vit):
        from src.classifier import _PREPROCESS_512
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_vit.return_value = mock_model

        clf = ImageNetClassifier(model_name="vit_l_16", device="cpu")
        assert clf._preprocess is _PREPROCESS_512


class TestPerModelPreprocess:
    @patch("src.classifier.models.resnet50")
    @patch("src.classifier.get_attention_generator")
    def test_resnet50_uses_224_preprocess(self, mock_attn, mock_resnet):
        from src.classifier import _PREPROCESS
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_resnet.return_value = mock_model

        clf = ImageNetClassifier(model_name="resnet50", device="cpu")
        assert clf._preprocess is _PREPROCESS


class TestOffloadReload:
    @patch("src.classifier.models.resnet50")
    @patch("src.classifier.get_attention_generator")
    @patch("src.classifier.torch")
    def test_offload_sets_loaded_false(self, mock_torch, mock_attn, mock_resnet):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_resnet.return_value = mock_model

        clf = ImageNetClassifier(model_name="resnet50", device="cpu")
        assert clf.loaded is True
        clf.offload()
        assert clf.loaded is False
