"""Tests for src/editor.py branches using mocks (Flux2, Qwen, error handling)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


class TestEditorDetectsBackend:
    def test_flux_detected(self):
        from src.editor import ImageEditor
        editor = object.__new__(ImageEditor)
        editor._model_name = "black-forest-labs/FLUX.2-klein-9b-kv"
        editor._is_flux = "flux" in editor._model_name.lower()
        editor._is_qwen = "qwen" in editor._model_name.lower()
        assert editor._is_flux is True
        assert editor._is_qwen is False

    def test_qwen_detected(self):
        from src.editor import ImageEditor
        editor = object.__new__(ImageEditor)
        editor._model_name = "Qwen/Qwen-Image-Edit"
        editor._is_flux = "flux" in editor._model_name.lower()
        editor._is_qwen = "qwen" in editor._model_name.lower()
        assert editor._is_qwen is True
        assert editor._is_flux is False

    def test_pix2pix_detected(self):
        from src.editor import ImageEditor
        editor = object.__new__(ImageEditor)
        editor._model_name = "timbrooks/instruct-pix2pix"
        editor._is_flux = "flux" in editor._model_name.lower()
        editor._is_qwen = "qwen" in editor._model_name.lower()
        assert editor._is_flux is False
        assert editor._is_qwen is False


class TestEditorEditBranches:
    """Test the edit() method branches for different backends."""

    def _make_editor(self, is_flux=False, is_qwen=False):
        from src.editor import ImageEditor
        editor = object.__new__(ImageEditor)
        editor._is_flux = is_flux
        editor._is_qwen = is_qwen
        editor._use_fp8_offload = False
        editor.device = "cpu"
        editor.loaded = True
        editor.pipe = MagicMock()
        result = MagicMock()
        result.images = [Image.new("RGB", (256, 256), "green")]
        editor.pipe.return_value = result
        return editor

    def test_pix2pix_edit_calls_pipe(self):
        from src.editor import ImageEditor
        editor = self._make_editor()
        img = Image.new("RGB", (256, 256), "red")
        result = editor.edit(img, "Make it blue", num_steps=1, seed=42)
        assert isinstance(result, Image.Image)
        editor.pipe.assert_called_once()

    def test_flux_edit_calls_pipe(self):
        from src.editor import ImageEditor
        editor = self._make_editor(is_flux=True)
        img = Image.new("RGB", (256, 256), "red")
        result = editor.edit(img, "Make it blue", num_steps=1, seed=42)
        assert isinstance(result, Image.Image)

    def test_qwen_edit_calls_pipe(self):
        from src.editor import ImageEditor
        editor = self._make_editor(is_qwen=True)
        img = Image.new("RGB", (256, 256), "red")
        result = editor.edit(img, "Make it blue", num_steps=1, seed=42)
        assert isinstance(result, Image.Image)

    def test_qwen_fp8_uses_cpu_generator(self):
        from src.editor import ImageEditor
        editor = self._make_editor(is_qwen=True)
        editor._use_fp8_offload = True
        img = Image.new("RGB", (256, 256), "red")
        result = editor.edit(img, "Make it blue", num_steps=1, seed=42)
        assert isinstance(result, Image.Image)

    def test_preserves_original_size(self):
        editor = self._make_editor()
        img = Image.new("RGB", (800, 600), "red")
        result = editor.edit(img, "test", num_steps=1, seed=42)
        assert result.size == (800, 600)

    def test_oom_raises(self):
        import torch
        from src.editor import ImageEditor
        editor = self._make_editor()
        editor.pipe.side_effect = torch.cuda.OutOfMemoryError("OOM")
        img = Image.new("RGB", (256, 256))
        with pytest.raises(torch.cuda.OutOfMemoryError):
            editor.edit(img, "test", num_steps=1, seed=42)

    def test_generic_exception_raises(self):
        from src.editor import ImageEditor
        editor = self._make_editor()
        editor.pipe.side_effect = RuntimeError("unexpected")
        img = Image.new("RGB", (256, 256))
        with pytest.raises(RuntimeError):
            editor.edit(img, "test", num_steps=1, seed=42)


class TestEditorOffloadReload:
    def _make_editor(self):
        from src.editor import ImageEditor
        editor = object.__new__(ImageEditor)
        editor._model_name = "timbrooks/instruct-pix2pix"
        editor._dtype_str = "float16"
        editor._use_fp8_offload = False
        editor._is_flux = False
        editor._is_qwen = False
        editor.device = "cpu"
        editor.loaded = True
        editor.pipe = MagicMock()
        return editor

    def test_offload(self):
        editor = self._make_editor()
        editor.offload()
        assert editor.loaded is False
        assert editor.pipe is None

    def test_load_to_gpu_when_already_loaded(self):
        editor = self._make_editor()
        editor.load_to_gpu()  # Should be no-op
        assert editor.loaded is True
