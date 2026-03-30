"""Tests for ViT attention map support."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.attention_maps import (
    GradCAM, GradCAMPlusPlus, ScoreCAM,
    get_attention_generator, reshape_vit_tokens,
)


class TestReshapeVitTokens:
    def test_strips_cls_and_reshapes(self):
        # (B=1, N+1=257, D=768) — 256 patches + CLS token
        tensor = torch.randn(1, 257, 768)
        result = reshape_vit_tokens(tensor)
        assert result.shape == (1, 768, 16, 16)

    def test_single_patch(self):
        # (B=1, N+1=2, D=4) — 1 patch + CLS
        tensor = torch.randn(1, 2, 4)
        result = reshape_vit_tokens(tensor)
        assert result.shape == (1, 4, 1, 1)

    def test_batch_preserved(self):
        tensor = torch.randn(3, 257, 768)
        result = reshape_vit_tokens(tensor)
        assert result.shape[0] == 3

    def test_values_preserved(self):
        tensor = torch.zeros(1, 5, 3)
        tensor[0, 1:, :] = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        result = reshape_vit_tokens(tensor)
        # CLS token (index 0) should be stripped
        assert result.shape == (1, 3, 2, 2)
        # First channel should be [1, 4, 7, 10] reshaped to (2, 2)
        assert result[0, 0, 0, 0].item() == 1.0
        assert result[0, 0, 0, 1].item() == 4.0


class TestFactoryWithArch:
    def test_gradcam_with_vit(self):
        gen = get_attention_generator("gradcam", target_layer="blocks.11", arch="vit")
        assert isinstance(gen, GradCAM)
        assert gen.arch == "vit"
        assert gen.target_layer == "blocks.11"

    def test_gradcampp_with_vit(self):
        gen = get_attention_generator("gradcam++", target_layer="blocks.11", arch="vit")
        assert isinstance(gen, GradCAMPlusPlus)
        assert gen.arch == "vit"

    def test_scorecam_with_cnn(self):
        gen = get_attention_generator("scorecam", target_layer="layer4.2", arch="cnn")
        assert isinstance(gen, ScoreCAM)
        assert gen.arch == "cnn"

    def test_default_arch_is_cnn(self):
        gen = get_attention_generator("gradcam")
        assert gen.arch == "cnn"


class TestGradCAMViTReshape:
    def test_to_spatial_reshapes_3d_vit(self):
        cam = GradCAM(target_layer="blocks.11", arch="vit")
        tensor = torch.randn(1, 257, 768)
        result = cam._to_spatial(tensor)
        assert result.shape == (1, 768, 16, 16)

    def test_to_spatial_passes_through_4d_cnn(self):
        cam = GradCAM(target_layer="layer4.2", arch="cnn")
        tensor = torch.randn(1, 512, 7, 7)
        result = cam._to_spatial(tensor)
        assert result.shape == (1, 512, 7, 7)

    def test_to_spatial_passes_through_4d_even_with_vit(self):
        """If a ViT layer happens to output 4D, don't reshape."""
        cam = GradCAM(target_layer="blocks.11", arch="vit")
        tensor = torch.randn(1, 512, 7, 7)
        result = cam._to_spatial(tensor)
        assert result.shape == (1, 512, 7, 7)
