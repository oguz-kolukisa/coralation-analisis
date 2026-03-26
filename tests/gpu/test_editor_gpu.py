"""GPU integration tests for src/editor.py.

Tests ImageEditor with actual diffusion model weights.
Most tests require significant VRAM (8GB+).
"""
from __future__ import annotations

import pytest
from PIL import Image

from src.editor import ImageEditor, _MAX_DIMENSION

pytestmark = pytest.mark.gpu


# ============================================================================
# Editor initialization (requires model download)
# ============================================================================

class TestEditorInit:
    def test_loads_pix2pix(self, device):
        editor = ImageEditor(
            model_name="timbrooks/instruct-pix2pix",
            device=device,
            dtype="float16",
            use_8bit=False,
        )
        assert editor.loaded is True
        editor.offload()

    def test_loaded_flag(self, device):
        editor = ImageEditor(
            model_name="timbrooks/instruct-pix2pix",
            device=device,
            dtype="float16",
            use_8bit=False,
        )
        assert editor.loaded is True
        editor.offload()
        assert editor.loaded is False


# ============================================================================
# Offload / Reload
# ============================================================================

class TestEditorOffload:
    @pytest.fixture
    def editor(self, device):
        ed = ImageEditor(
            model_name="timbrooks/instruct-pix2pix",
            device=device,
            dtype="float16",
            use_8bit=False,
        )
        yield ed
        if ed.loaded:
            ed.offload()

    def test_offload_clears_pipe(self, editor):
        editor.offload()
        assert editor.pipe is None
        assert editor.loaded is False

    def test_reload_after_offload(self, editor):
        editor.offload()
        editor.load_to_gpu()
        assert editor.loaded is True
        assert editor.pipe is not None


# ============================================================================
# Edit operation
# ============================================================================

class TestEditorEdit:
    @pytest.fixture(scope="class")
    def editor(self, device):
        ed = ImageEditor(
            model_name="timbrooks/instruct-pix2pix",
            device=device,
            dtype="float16",
            use_8bit=False,
        )
        yield ed
        ed.offload()

    def test_edit_returns_image(self, editor, sample_image):
        result = editor.edit(sample_image, "Make it blue", num_steps=2, seed=42)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_original_size(self, editor, large_image):
        result = editor.edit(large_image, "Change color", num_steps=2, seed=42)
        assert result.size == large_image.size

    def test_edit_deterministic_with_seed(self, editor, sample_image):
        r1 = editor.edit(sample_image, "Make it blue", num_steps=2, seed=123)
        r2 = editor.edit(sample_image, "Make it blue", num_steps=2, seed=123)
        # Same seed should produce identical results
        import numpy as np
        assert np.array_equal(np.array(r1), np.array(r2))

    def test_edit_different_seeds_differ(self, editor, sample_image):
        r1 = editor.edit(sample_image, "Make it blue", num_steps=2, seed=1)
        r2 = editor.edit(sample_image, "Make it blue", num_steps=2, seed=999)
        import numpy as np
        assert not np.array_equal(np.array(r1), np.array(r2))

    def test_edit_small_image(self, editor, small_image):
        """Small images should be handled (resized up to min 256)."""
        result = editor.edit(small_image, "Change color", num_steps=2, seed=42)
        assert isinstance(result, Image.Image)
        assert result.size == small_image.size
