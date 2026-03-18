"""Tests for ModelManager."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import Config
from src.model_manager import ModelManager


@pytest.fixture
def config():
    return Config(device="cpu", low_vram=True)


@pytest.fixture
def config_high_vram():
    return Config(device="cpu", low_vram=False)


# =========================================================================
# Initialization
# =========================================================================

class TestInit:
    def test_all_models_start_none(self, config):
        mm = ModelManager(config)
        assert mm._classifier is None
        assert mm._vlm is None
        assert mm._editor is None
        assert mm._sampler is None

    def test_stores_config(self, config):
        mm = ModelManager(config)
        assert mm.cfg is config


# =========================================================================
# Lazy loading
# =========================================================================

class TestLazyLoading:
    @patch("src.model_manager.ImageNetClassifier")
    @patch("src.model_manager.torch")
    def test_classifier_loads_on_first_call(self, mock_torch, MockClassifier, config_high_vram):
        mm = ModelManager(config_high_vram)
        result = mm.classifier()
        MockClassifier.assert_called_once()
        assert result is mm._classifier

    @patch("src.model_manager.ImageNetClassifier")
    @patch("src.model_manager.torch")
    def test_classifier_reuses_on_second_call(self, mock_torch, MockClassifier, config_high_vram):
        mm = ModelManager(config_high_vram)
        first = mm.classifier()
        second = mm.classifier()
        assert MockClassifier.call_count == 1
        assert first is second

    @patch("src.model_manager.QwenVLAnalyzer")
    @patch("src.model_manager.torch")
    def test_vlm_loads_on_first_call(self, mock_torch, MockVLM, config_high_vram):
        mm = ModelManager(config_high_vram)
        result = mm.vlm()
        MockVLM.assert_called_once()
        assert result is mm._vlm

    @patch("src.model_manager.ImageEditor")
    @patch("src.model_manager.torch")
    def test_editor_loads_on_first_call(self, mock_torch, MockEditor, config_high_vram):
        mm = ModelManager(config_high_vram)
        result = mm.editor()
        MockEditor.assert_called_once()
        assert result is mm._editor

    @patch("src.model_manager.ImageNetSampler")
    def test_sampler_loads_on_first_call(self, MockSampler, config):
        mm = ModelManager(config)
        result = mm.sampler()
        MockSampler.assert_called_once()
        assert result is mm._sampler

    @patch("src.model_manager.ImageNetSampler")
    def test_sampler_reuses(self, MockSampler, config):
        mm = ModelManager(config)
        mm.sampler()
        mm.sampler()
        assert MockSampler.call_count == 1


# =========================================================================
# Low VRAM offloading
# =========================================================================

class TestLowVramOffloading:
    @patch("src.model_manager.ImageNetClassifier")
    @patch("src.model_manager.torch")
    def test_classifier_offloads_vlm_in_low_vram(self, mock_torch, MockCls, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm.classifier()
        mm._vlm.offload.assert_called_once()

    @patch("src.model_manager.QwenVLAnalyzer")
    @patch("src.model_manager.torch")
    def test_vlm_offloads_classifier_and_editor(self, mock_torch, MockVLM, config):
        mm = ModelManager(config)
        mm._classifier = MagicMock()
        mm._classifier.loaded = True
        mm._editor = MagicMock()
        mm._editor.loaded = True
        mm.vlm()
        mm._classifier.offload.assert_called_once()
        mm._editor.offload.assert_called_once()

    @patch("src.model_manager.ImageEditor")
    @patch("src.model_manager.torch")
    def test_editor_offloads_vlm(self, mock_torch, MockEditor, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm.editor()
        mm._vlm.offload.assert_called_once()


# =========================================================================
# Offload methods
# =========================================================================

class TestOffloading:
    @patch("src.model_manager.torch")
    def test_offload_classifier_calls_offload(self, mock_torch, config):
        mm = ModelManager(config)
        mm._classifier = MagicMock()
        mm._classifier.loaded = True
        mm.offload_classifier()
        mm._classifier.offload.assert_called_once()

    @patch("src.model_manager.torch")
    def test_offload_classifier_none_is_noop(self, mock_torch, config):
        mm = ModelManager(config)
        mm.offload_classifier()

    @patch("src.model_manager.torch")
    def test_offload_vlm_calls_offload(self, mock_torch, config):
        mm = ModelManager(config)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm.offload_vlm()
        mm._vlm.offload.assert_called_once()

    @patch("src.model_manager.torch")
    def test_offload_vlm_none_is_noop(self, mock_torch, config):
        mm = ModelManager(config)
        mm.offload_vlm()

    @patch("src.model_manager.torch")
    def test_offload_editor_calls_offload(self, mock_torch, config):
        mm = ModelManager(config)
        mm._editor = MagicMock()
        mm._editor.loaded = True
        mm.offload_editor()
        mm._editor.offload.assert_called_once()

    @patch("src.model_manager.torch")
    def test_offload_editor_none_is_noop(self, mock_torch, config):
        mm = ModelManager(config)
        mm.offload_editor()

    @patch("src.model_manager.torch")
    def test_offload_all(self, mock_torch, config):
        mm = ModelManager(config)
        mm._classifier = MagicMock()
        mm._classifier.loaded = True
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm._editor = MagicMock()
        mm._editor.loaded = True
        mm.offload_all()
        mm._classifier.offload.assert_called_once()
        mm._vlm.offload.assert_called_once()
        mm._editor.offload.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called()


# =========================================================================
# High VRAM mode (no offloading)
# =========================================================================

class TestHighVram:
    @patch("src.model_manager.ImageNetClassifier")
    @patch("src.model_manager.torch")
    def test_classifier_does_not_offload_vlm(self, mock_torch, MockCls, config_high_vram):
        mm = ModelManager(config_high_vram)
        mm._vlm = MagicMock()
        mm._vlm.loaded = True
        mm.classifier()
        mm._vlm.offload.assert_not_called()
