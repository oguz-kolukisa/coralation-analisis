"""Tests for src/editor.py generate_from_text and its helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image


def _make_editor(is_flux=True, is_qwen=False, fp8=False):
    from src.editor import ImageEditor
    editor = object.__new__(ImageEditor)
    editor._model_name = "black-forest-labs/FLUX.2-klein-9b-kv" if is_flux else "qwen"
    editor._is_flux = is_flux
    editor._is_qwen = is_qwen
    editor._use_fp8_offload = fp8
    editor.device = "cpu"
    editor.loaded = True
    editor.pipe = MagicMock()
    result = MagicMock()
    result.images = [Image.new("RGB", (256, 256), "cyan")]
    editor.pipe.return_value = result
    return editor


class TestSnapDims:
    def test_snaps_to_multiples_of_8(self):
        from src.editor import ImageEditor
        w, h = ImageEditor._snap_dims(513, 513)
        assert w % 8 == 0 and h % 8 == 0

    def test_clamps_to_max_dimension(self):
        from src.editor import ImageEditor
        w, h = ImageEditor._snap_dims(4096, 4096)
        assert w <= 768 and h <= 768

    def test_enforces_minimum_size(self):
        from src.editor import ImageEditor
        w, h = ImageEditor._snap_dims(40, 40)
        assert w >= 256 and h >= 256

    def test_preserves_aspect_ratio_down(self):
        from src.editor import ImageEditor
        # 2000x1000 → should be roughly 768x384
        w, h = ImageEditor._snap_dims(2000, 1000)
        assert w > h
        # aspect ratio preserved within 10%
        assert 1.8 < w / h < 2.2


class TestMakeGenerator:
    def test_returns_generator_with_manual_seed(self):
        editor = _make_editor(is_flux=True)
        gen = editor._make_generator(7)
        import torch
        assert isinstance(gen, torch.Generator)
        # Two generators with same seed produce same first random
        g2 = editor._make_generator(7)
        assert gen.initial_seed() == g2.initial_seed()

    def test_flux_uses_device_generator(self):
        editor = _make_editor(is_flux=True, fp8=True)
        gen = editor._make_generator(1)
        # FLUX never uses CPU generator (only qwen+fp8 does)
        # On cpu device, both are "cpu"; this asserts no crash path
        assert gen is not None


class TestGenerateFromText:
    def test_raises_on_non_flux_backend(self):
        editor = _make_editor(is_flux=False, is_qwen=True)
        with pytest.raises(NotImplementedError):
            editor.generate_from_text("a red apple on a table")

    def test_calls_pipe_with_image_none(self):
        editor = _make_editor(is_flux=True)
        editor.generate_from_text("a red apple", seed=42, size=(512, 512))
        editor.pipe.assert_called_once()
        kwargs = editor.pipe.call_args.kwargs
        assert kwargs["image"] is None

    def test_uses_flux_step_count(self):
        from src.editor import _FLUX_STEPS
        editor = _make_editor(is_flux=True)
        editor.generate_from_text("a red apple", seed=42)
        kwargs = editor.pipe.call_args.kwargs
        assert kwargs["num_inference_steps"] == _FLUX_STEPS

    def test_snaps_size_to_multiples_of_8(self):
        editor = _make_editor(is_flux=True)
        editor.generate_from_text("a red apple", seed=42, size=(513, 259))
        kwargs = editor.pipe.call_args.kwargs
        assert kwargs["height"] % 8 == 0
        assert kwargs["width"] % 8 == 0

    def test_propagates_seed_to_generator(self):
        editor = _make_editor(is_flux=True)
        editor.generate_from_text("a red apple", seed=77)
        kwargs = editor.pipe.call_args.kwargs
        assert kwargs["generator"].initial_seed() == 77

    def test_returns_image_from_pipe(self):
        editor = _make_editor(is_flux=True)
        img = editor.generate_from_text("a red apple", seed=42)
        assert isinstance(img, Image.Image)

    def test_reraises_oom(self):
        import torch
        editor = _make_editor(is_flux=True)
        editor.pipe.side_effect = torch.cuda.OutOfMemoryError("test")
        with pytest.raises(torch.cuda.OutOfMemoryError):
            editor.generate_from_text("a red apple", seed=42)

    def test_reraises_generic_exception(self):
        editor = _make_editor(is_flux=True)
        editor.pipe.side_effect = RuntimeError("bad prompt")
        with pytest.raises(RuntimeError):
            editor.generate_from_text("a red apple", seed=42)
