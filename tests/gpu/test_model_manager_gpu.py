"""GPU integration tests for src/models/model_manager.py.

Tests ModelManager loading/offloading with real model weights.
"""
from __future__ import annotations

import pytest
import torch

from src.models.model_manager import ModelManager, ModelState

pytestmark = pytest.mark.gpu


# ============================================================================
# Init
# ============================================================================

class TestManagerInit:
    def test_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "cache"
        mm = ModelManager(cache_dir=cache, device="cpu", low_vram=False)
        assert cache.exists()

    def test_default_state_empty(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        assert mm._models == {}
        assert mm._states == {}


# ============================================================================
# Classifier (small, always works on CPU)
# ============================================================================

class TestLoadClassifier:
    def test_loads_resnet50(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        model, weights = mm.load_classifier()
        assert model is not None
        assert mm._states["classifier"] == ModelState.ON_GPU

    def test_returns_cached_on_second_call(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        r1 = mm.load_classifier()
        r2 = mm.load_classifier()
        # Second call returns same model (tuple of model, weights)
        assert r1[0] is r2[0]

    def test_offload_classifier(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        mm.load_classifier()
        mm.offload("classifier")
        assert mm._states["classifier"] == ModelState.UNLOADED

    def test_offload_nonexistent_is_noop(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        mm.offload("nonexistent")  # Should not raise


# ============================================================================
# VRAM usage
# ============================================================================

class TestVRAMUsage:
    def test_returns_dict_on_cpu(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        usage = mm.get_vram_usage()
        if not torch.cuda.is_available():
            assert "error" in usage
        else:
            assert "allocated_gb" in usage
            assert "free_gb" in usage

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vram_keys_on_gpu(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cuda", low_vram=False)
        usage = mm.get_vram_usage()
        assert all(k in usage for k in ("allocated_gb", "reserved_gb", "total_gb", "free_gb"))


# ============================================================================
# Cleanup
# ============================================================================

class TestCleanup:
    def test_cleanup_all(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        mm.load_classifier()
        mm.cleanup()
        assert mm._states["classifier"] == ModelState.UNLOADED

    def test_cleanup_empty_is_noop(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False)
        mm.cleanup()  # Should not raise


# ============================================================================
# Low VRAM mode
# ============================================================================

class TestLowVRAM:
    def test_low_vram_offloads_vlm_on_classifier_load(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=True)
        # Simulate VLM being loaded
        mm._models["vlm"] = "fake_model"
        mm._states["vlm"] = ModelState.ON_GPU
        # Loading classifier should offload VLM
        mm.load_classifier()
        assert mm._states["vlm"] == ModelState.UNLOADED
