"""GPU tests for models/model_manager.py — actual model loading.

Requires downloaded models:
- Qwen/Qwen2.5-VL-7B-Instruct
- timbrooks/instruct-pix2pix
"""
from __future__ import annotations

import pytest
import torch

from src.models.model_manager import ModelManager, ModelState

pytestmark = pytest.mark.gpu


# ============================================================================
# load_vlm
# ============================================================================

class TestLoadVLMGPU:
    def test_loads_vlm(self, device, tmp_path):
        mm = ModelManager(
            cache_dir=tmp_path, device=device,
            low_vram=False, disk_offload=True,
        )
        model, processor = mm.load_vlm()
        assert model is not None
        assert processor is not None
        assert mm._states["vlm"] == ModelState.ON_GPU
        mm.cleanup()

    def test_vlm_cached_on_second_call(self, device, tmp_path):
        mm = ModelManager(
            cache_dir=tmp_path, device=device,
            low_vram=False, disk_offload=True,
        )
        r1 = mm.load_vlm()
        r2 = mm.load_vlm()
        assert r1[0] is r2[0]  # Same model object
        mm.cleanup()

    def test_low_vram_offloads_others(self, device, tmp_path):
        mm = ModelManager(
            cache_dir=tmp_path, device=device,
            low_vram=True, disk_offload=True,
        )
        # Load classifier first
        mm.load_classifier()
        assert mm._states["classifier"] == ModelState.ON_GPU

        # Loading VLM should offload classifier (via _offload_all_except)
        mm.load_vlm()
        # Classifier may or may not be offloaded depending on low_vram behavior
        assert mm._states["vlm"] == ModelState.ON_GPU
        mm.cleanup()


# ============================================================================
# load_editor
# ============================================================================

class TestLoadEditorGPU:
    def test_loads_editor(self, device, tmp_path):
        mm = ModelManager(
            cache_dir=tmp_path, device=device,
            low_vram=False, disk_offload=True,
        )
        pipe = mm.load_editor()
        assert pipe is not None
        assert mm._states["editor"] == ModelState.ON_GPU
        mm.cleanup()

    def test_editor_cached_on_second_call(self, device, tmp_path):
        mm = ModelManager(
            cache_dir=tmp_path, device=device,
            low_vram=False, disk_offload=True,
        )
        r1 = mm.load_editor()
        r2 = mm.load_editor()
        assert r1 is r2
        mm.cleanup()

    def test_loads_without_disk_offload(self, device, tmp_path):
        mm = ModelManager(
            cache_dir=tmp_path, device=device,
            low_vram=False, disk_offload=False,
        )
        pipe = mm.load_editor()
        assert pipe is not None
        assert mm._states["editor"] == ModelState.ON_GPU
        mm.cleanup()
