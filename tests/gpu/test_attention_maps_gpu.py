"""GPU integration tests for src/models/attention_maps.py.

Tests GradCAM, GradCAM++, ScoreCAM with a real ResNet-50.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from src.models.attention_maps import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    compute_attention_diff,
    get_attention_generator,
    render_diff_heatmap,
)

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resnet():
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights).eval()
    return model


@pytest.fixture(scope="module")
def input_tensor():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.new("RGB", (256, 256), "red")
    return preprocess(img).unsqueeze(0)


# ============================================================================
# Factory
# ============================================================================

class TestGetAttentionGenerator:
    def test_gradcam(self):
        gen = get_attention_generator("gradcam")
        assert isinstance(gen, GradCAM)

    def test_gradcam_plus_plus(self):
        gen = get_attention_generator("gradcam++")
        assert isinstance(gen, GradCAMPlusPlus)

    def test_scorecam(self):
        gen = get_attention_generator("scorecam")
        assert isinstance(gen, ScoreCAM)

    def test_unknown_falls_back_to_scorecam(self):
        gen = get_attention_generator("nonexistent")
        assert isinstance(gen, ScoreCAM)

    def test_case_insensitive(self):
        gen = get_attention_generator("GradCAM")
        assert isinstance(gen, GradCAM)


# ============================================================================
# GradCAM
# ============================================================================

class TestGradCAM:
    def test_output_shape(self, resnet, input_tensor):
        cam = GradCAM().generate(resnet, input_tensor, target_class=0)
        assert cam.ndim == 2
        assert cam.shape[0] > 0 and cam.shape[1] > 0

    def test_output_range(self, resnet, input_tensor):
        cam = GradCAM().generate(resnet, input_tensor, target_class=0)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_different_classes_different_maps(self, resnet, input_tensor):
        cam1 = GradCAM().generate(resnet, input_tensor, target_class=0)
        cam2 = GradCAM().generate(resnet, input_tensor, target_class=100)
        assert not np.array_equal(cam1, cam2)

    def test_returns_numpy(self, resnet, input_tensor):
        cam = GradCAM().generate(resnet, input_tensor, target_class=0)
        assert isinstance(cam, np.ndarray)
        assert cam.dtype in (np.float32, np.float64)

    def test_custom_target_layer(self, resnet, input_tensor):
        cam = GradCAM(target_layer="layer3").generate(resnet, input_tensor, target_class=0)
        assert cam.ndim == 2


# ============================================================================
# GradCAM++
# ============================================================================

class TestGradCAMPlusPlus:
    def test_output_shape(self, resnet, input_tensor):
        cam = GradCAMPlusPlus().generate(resnet, input_tensor, target_class=0)
        assert cam.ndim == 2

    def test_output_range(self, resnet, input_tensor):
        cam = GradCAMPlusPlus().generate(resnet, input_tensor, target_class=0)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_custom_target_layer(self, resnet, input_tensor):
        cam = GradCAMPlusPlus(target_layer="layer3").generate(
            resnet, input_tensor, target_class=0
        )
        assert cam.ndim == 2


# ============================================================================
# ScoreCAM
# ============================================================================

class TestScoreCAM:
    def test_output_shape(self, resnet, input_tensor):
        cam = ScoreCAM(batch_size=16).generate(resnet, input_tensor, target_class=0)
        assert cam.ndim == 2

    def test_output_range(self, resnet, input_tensor):
        cam = ScoreCAM(batch_size=16).generate(resnet, input_tensor, target_class=0)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_gradient_free(self, resnet, input_tensor):
        """ScoreCAM should work even when gradients are disabled."""
        with torch.no_grad():
            cam = ScoreCAM(batch_size=16).generate(resnet, input_tensor, target_class=0)
        assert cam.ndim == 2


# ============================================================================
# Overlay
# ============================================================================

class TestOverlay:
    def test_overlay_returns_image(self, resnet, input_tensor):
        cam = GradCAM().generate(resnet, input_tensor, target_class=0)
        original = Image.new("RGB", (224, 224), "blue")
        overlay = GradCAM().overlay_on_image(cam, original)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == original.size

    def test_overlay_alpha_zero_is_original(self, resnet, input_tensor):
        cam = GradCAM().generate(resnet, input_tensor, target_class=0)
        original = Image.new("RGB", (100, 100), "blue")
        overlay = GradCAM().overlay_on_image(cam, original, alpha=0.0)
        # With alpha=0, result should be close to original
        diff = np.abs(
            np.array(overlay).astype(float) - np.array(original).astype(float)
        ).mean()
        assert diff < 2.0  # Allow small rounding


# ============================================================================
# Attention Diff
# ============================================================================

class TestAttentionDiff:
    def test_diff_same_maps_is_zero(self):
        m = np.random.rand(7, 7).astype(np.float32)
        diff = compute_attention_diff(m, m)
        assert np.allclose(diff, 0.0, atol=1e-6)

    def test_diff_range(self):
        m1 = np.zeros((7, 7), dtype=np.float32)
        m2 = np.ones((7, 7), dtype=np.float32)
        diff = compute_attention_diff(m1, m2)
        assert diff.min() >= -1.0
        assert diff.max() <= 1.0

    def test_diff_shape_mismatch_handled(self):
        m1 = np.random.rand(5, 5).astype(np.float32)
        m2 = np.random.rand(7, 7).astype(np.float32)
        diff = compute_attention_diff(m1, m2)
        assert diff.shape == (7, 7)

    def test_render_diff_heatmap_returns_image(self):
        diff = np.random.rand(7, 7).astype(np.float32) * 2 - 1
        original = Image.new("RGB", (100, 100), "white")
        result = render_diff_heatmap(diff, original)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
