"""GPU tests for editor.py — Flux2 and Qwen-Image-Edit backends.

Requires downloaded models:
- black-forest-labs/FLUX.2-klein-9b-kv
- Qwen/Qwen-Image-Edit
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.editor import ImageEditor

pytestmark = pytest.mark.gpu


@pytest.fixture
def small_image():
    return Image.new("RGB", (512, 512), "blue")


# ============================================================================
# Flux2 backend
# ============================================================================

class TestFlux2Editor:
    @pytest.fixture(scope="class")
    def editor(self, device):
        ed = ImageEditor(
            model_name="black-forest-labs/FLUX.2-klein-9b-kv",
            device=device,
            dtype="bfloat16",
            use_8bit=True,
        )
        yield ed
        if ed.loaded:
            ed.offload()

    def test_init_loaded(self, editor):
        assert editor.loaded is True
        assert editor._is_flux is True

    def test_edit_returns_image(self, editor, small_image):
        result = editor.edit(small_image, "Make it red", num_steps=2, seed=42)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self, editor, small_image):
        result = editor.edit(small_image, "Change color", num_steps=2, seed=42)
        assert result.size == small_image.size

    def test_offload_and_reload(self, editor):
        editor.offload()
        assert editor.loaded is False
        editor.load_to_gpu()
        assert editor.loaded is True


# ============================================================================
# Qwen-Image-Edit backend
# ============================================================================

class TestQwenEditor:
    @pytest.fixture(scope="class")
    def editor(self, device):
        ed = ImageEditor(
            model_name="Qwen/Qwen-Image-Edit",
            device=device,
            dtype="bfloat16",
            use_8bit=True,
        )
        yield ed
        if ed.loaded:
            ed.offload()

    def test_init_loaded(self, editor):
        assert editor.loaded is True
        assert editor._is_qwen is True

    def test_edit_returns_image(self, editor, small_image):
        result = editor.edit(small_image, "Make it green", num_steps=2, seed=42)
        assert isinstance(result, Image.Image)

    def test_offload_and_reload(self, editor):
        editor.offload()
        assert editor.loaded is False
        editor.load_to_gpu()
        assert editor.loaded is True
