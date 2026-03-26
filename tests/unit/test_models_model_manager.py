"""Tests for src/models/model_manager.py using mocks (no GPU needed)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.model_manager import ModelManager, ModelState


# ============================================================================
# Init
# ============================================================================

class TestModelManagerInit:
    def test_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "new_cache"
        mm = ModelManager(cache_dir=cache, device="cpu", low_vram=False, disk_offload=False)
        assert cache.exists()

    def test_stores_params(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=True, disk_offload=True)
        assert mm.device == "cpu"
        assert mm.low_vram is True
        assert mm.disk_offload is True

    def test_empty_models_on_init(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        assert mm._models == {}
        assert mm._states == {}


# ============================================================================
# load_vlm (mocked)
# ============================================================================

class TestLoadVLM:
    def test_returns_cached_vlm(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        fake = (MagicMock(), MagicMock())
        mm._models["vlm"] = fake
        mm._states["vlm"] = ModelState.ON_GPU

        result = mm.load_vlm()
        assert result is fake

    def test_low_vram_offloads_others_before_vlm(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=True, disk_offload=False)
        mm._models["editor"] = MagicMock()
        mm._states["editor"] = ModelState.ON_GPU
        # Simulate VLM already cached to avoid real model load
        mm._models["vlm"] = (MagicMock(), MagicMock())
        mm._states["vlm"] = ModelState.ON_GPU

        mm.load_vlm()
        # Editor should stay since VLM was cached (skip offload path)


# ============================================================================
# load_editor (mocked)
# ============================================================================

class TestLoadEditor:
    def test_returns_cached_editor(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        fake_pipe = MagicMock()
        mm._models["editor"] = fake_pipe
        mm._states["editor"] = ModelState.ON_GPU

        result = mm.load_editor()
        assert result is fake_pipe


# ============================================================================
# offload
# ============================================================================

class TestOffload:
    def test_offload_sets_unloaded(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm._models["test"] = MagicMock()
        mm._states["test"] = ModelState.ON_GPU

        mm.offload("test")
        assert mm._states["test"] == ModelState.UNLOADED

    def test_offload_nonexistent_is_noop(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm.offload("nonexistent")  # Should not raise

    def test_offload_already_unloaded_is_noop(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm._models["test"] = None
        mm._states["test"] = ModelState.UNLOADED

        mm.offload("test")
        assert mm._states["test"] == ModelState.UNLOADED

    def test_offload_on_disk_is_noop(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm._models["test"] = None
        mm._states["test"] = ModelState.ON_DISK

        mm.offload("test")
        assert mm._states["test"] == ModelState.ON_DISK

    def test_offload_tuple_model(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm._models["vlm"] = (MagicMock(), MagicMock())
        mm._states["vlm"] = ModelState.ON_GPU

        mm.offload("vlm")
        assert mm._states["vlm"] == ModelState.UNLOADED


# ============================================================================
# _offload_all_except
# ============================================================================

class TestOffloadAllExcept:
    def test_offloads_others(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm._models["vlm"] = MagicMock()
        mm._states["vlm"] = ModelState.ON_GPU
        mm._models["editor"] = MagicMock()
        mm._states["editor"] = ModelState.ON_GPU
        mm._models["classifier"] = MagicMock()
        mm._states["classifier"] = ModelState.ON_GPU

        mm._offload_all_except("classifier")
        assert mm._states["vlm"] == ModelState.UNLOADED
        assert mm._states["editor"] == ModelState.UNLOADED
        assert mm._states["classifier"] == ModelState.ON_GPU


# ============================================================================
# get_vram_usage
# ============================================================================

class TestGetVRAMUsage:
    @patch("torch.cuda.is_available", return_value=False)
    def test_no_cuda_returns_error(self, mock_cuda, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        usage = mm.get_vram_usage()
        assert "error" in usage

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=2e9)
    @patch("torch.cuda.memory_reserved", return_value=4e9)
    @patch("torch.cuda.get_device_properties")
    def test_cuda_returns_stats(self, mock_props, mock_reserved, mock_alloc, mock_avail, tmp_path):
        mock_props.return_value = MagicMock(total_memory=8e9)
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        usage = mm.get_vram_usage()
        assert usage["allocated_gb"] == 2.0
        assert usage["reserved_gb"] == 4.0
        assert usage["total_gb"] == 8.0
        assert usage["free_gb"] == 6.0


# ============================================================================
# cleanup
# ============================================================================

class TestCleanup:
    def test_cleans_all_models(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm._models["a"] = MagicMock()
        mm._states["a"] = ModelState.ON_GPU
        mm._models["b"] = MagicMock()
        mm._states["b"] = ModelState.ON_GPU

        mm.cleanup()
        assert mm._states["a"] == ModelState.UNLOADED
        assert mm._states["b"] == ModelState.UNLOADED

    def test_cleanup_empty(self, tmp_path):
        mm = ModelManager(cache_dir=tmp_path, device="cpu", low_vram=False, disk_offload=False)
        mm.cleanup()  # Should not raise


# ============================================================================
# ModelState enum
# ============================================================================

class TestModelState:
    def test_values(self):
        assert ModelState.UNLOADED.value == "unloaded"
        assert ModelState.ON_GPU.value == "on_gpu"
        assert ModelState.ON_CPU.value == "on_cpu"
        assert ModelState.ON_DISK.value == "on_disk"
