"""Tests for ModelManager._ensure_only and loaded-flag integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.model_manager import ModelManager


@pytest.fixture
def config():
    return Config(device="cpu", low_vram=True)


@pytest.fixture
def config_high_vram():
    return Config(device="cpu", low_vram=False)


class TestEnsureOnly:
    @patch("src.model_manager.torch")
    def test_offloads_loaded_models(self, mock_torch, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm._editor = MagicMock()
        mm._editor.loaded = True
        mm._ensure_only("classifier")
        mm._vlm.offload.assert_called_once()
        mm._editor.offload.assert_called_once()

    @patch("src.model_manager.torch")
    def test_skips_unloaded_models(self, mock_torch, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = False
        mm._ensure_only("classifier")
        mm._vlm.offload.assert_not_called()

    @patch("src.model_manager.torch")
    def test_skips_none_models(self, mock_torch, config):
        mm = ModelManager(config)
        mm._vlm = None
        mm._ensure_only("classifier")
        mock_torch.cuda.empty_cache.assert_called()

    @patch("src.model_manager.torch")
    def test_does_not_offload_requested_model(self, mock_torch, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm._ensure_only("vlm")
        mm._vlm.offload.assert_not_called()

    @patch("src.model_manager.torch")
    def test_noop_in_high_vram(self, mock_torch, config_high_vram):
        mm = ModelManager(config_high_vram)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm._ensure_only("classifier")
        mm._vlm.offload.assert_not_called()

    @patch("src.model_manager.torch")
    def test_clears_cuda_cache(self, mock_torch, config):
        mm = ModelManager(config)
        mm._ensure_only("classifier")
        mock_torch.cuda.empty_cache.assert_called_once()


class TestLoadToGpuIntegration:
    @patch("src.model_manager.ImageNetClassifier")
    @patch("src.model_manager.torch")
    def test_classifier_load_to_gpu_when_offloaded(self, mock_torch, MockCls, config_high_vram):
        mm = ModelManager(config_high_vram)
        mock_instance = MagicMock()
        mock_instance.loaded = False
        MockCls.return_value = mock_instance
        mm._classifier = mock_instance
        mm.classifier()
        mock_instance.load_to_gpu.assert_called_once()

    @patch("src.model_manager.QwenVLAnalyzer")
    @patch("src.model_manager.torch")
    def test_vlm_load_to_gpu_when_offloaded(self, mock_torch, MockVLM, config_high_vram):
        mm = ModelManager(config_high_vram)
        mock_instance = MagicMock()
        mock_instance.loaded = False
        MockVLM.return_value = mock_instance
        mm._vlm = mock_instance
        mm.vlm()
        mock_instance.load_to_gpu.assert_called_once()

    @patch("src.model_manager.ImageEditor")
    @patch("src.model_manager.torch")
    def test_editor_load_to_gpu_when_offloaded(self, mock_torch, MockEditor, config_high_vram):
        mm = ModelManager(config_high_vram)
        mock_instance = MagicMock()
        mock_instance.loaded = False
        MockEditor.return_value = mock_instance
        mm._editor = mock_instance
        mm.editor()
        mock_instance.load_to_gpu.assert_called_once()

    @patch("src.model_manager.ImageNetClassifier")
    @patch("src.model_manager.torch")
    def test_classifier_no_reload_when_loaded(self, mock_torch, MockCls, config_high_vram):
        mm = ModelManager(config_high_vram)
        mock_instance = MagicMock()
        mock_instance.loaded = True
        mm._classifier = mock_instance
        result = mm.classifier()
        mock_instance.load_to_gpu.assert_not_called()
        assert result is mock_instance


class TestOffloadNoop:
    @patch("src.model_manager.torch")
    def test_offload_classifier_noop_when_not_loaded(self, mock_torch, config):
        mm = ModelManager(config)
        mm._classifier = MagicMock()
        mm._classifier.loaded = False
        mm.offload_classifier()
        mm._classifier.offload.assert_not_called()

    @patch("src.model_manager.torch")
    def test_offload_vlm_noop_when_not_loaded(self, mock_torch, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = False
        mm.offload_vlm()
        mm._vlm.offload.assert_not_called()

    @patch("src.model_manager.torch")
    def test_offload_editor_noop_when_not_loaded(self, mock_torch, config):
        mm = ModelManager(config)
        mm._editor = MagicMock()
        mm._editor.loaded = False
        mm.offload_editor()
        mm._editor.offload.assert_not_called()
