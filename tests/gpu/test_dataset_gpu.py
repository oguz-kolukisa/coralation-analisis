"""GPU integration tests for src/dataset.py.

Tests ImageNetSampler with actual HuggingFace dataset access.
Requires HF_TOKEN for gated datasets like ImageNet.
"""
from __future__ import annotations

import os

import pytest
from PIL import Image

pytestmark = pytest.mark.gpu


def _get_hf_token():
    return os.environ.get("HF_TOKEN")


def _can_access_imagenet():
    """Check if ImageNet is accessible (requires token + dataset access)."""
    try:
        from datasets import load_dataset
        load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=_get_hf_token())
        return True
    except Exception:
        return False


needs_imagenet = pytest.mark.skipif(
    not _can_access_imagenet(),
    reason="ImageNet dataset not accessible (need HF_TOKEN with dataset access)"
)


# ============================================================================
# Init
# ============================================================================

@needs_imagenet
class TestSamplerInit:
    def test_loads_dataset(self):
        from src.dataset import ImageNetSampler
        sampler = ImageNetSampler(hf_token=_get_hf_token(), max_scan=100)
        assert sampler._ds is not None

    def test_detects_label_field(self):
        from src.dataset import ImageNetSampler
        sampler = ImageNetSampler(hf_token=_get_hf_token(), max_scan=100)
        assert sampler._label_field == "label"

    def test_detects_image_field(self):
        from src.dataset import ImageNetSampler
        sampler = ImageNetSampler(hf_token=_get_hf_token(), max_scan=100)
        assert sampler._image_field == "image"


# ============================================================================
# Label discovery
# ============================================================================

@needs_imagenet
class TestLabelDiscovery:
    @pytest.fixture(scope="class")
    def sampler(self):
        from src.dataset import ImageNetSampler
        return ImageNetSampler(hf_token=_get_hf_token(), max_scan=500)

    def test_get_label_names(self, sampler):
        names = sampler.get_label_names()
        assert isinstance(names, list)
        assert len(names) == 1000

    def test_find_label_index_existing(self, sampler):
        idx = sampler.find_label_index("goldfish")
        assert idx is not None
        assert isinstance(idx, int)

    def test_find_label_index_nonexistent(self, sampler):
        idx = sampler.find_label_index("xyznonexistent123")
        assert idx is None

    def test_find_label_indices_returns_list(self, sampler):
        indices = sampler.find_label_indices("shark")
        assert isinstance(indices, list)
        assert len(indices) > 0


# ============================================================================
# Sampling
# ============================================================================

@needs_imagenet
class TestSampling:
    @pytest.fixture(scope="class")
    def sampler(self):
        from src.dataset import ImageNetSampler
        return ImageNetSampler(hf_token=_get_hf_token(), max_scan=2000, seed=42)

    def test_sample_positive(self, sampler):
        results = sampler.sample_positive("goldfish", n=2)
        assert len(results) > 0
        img, label = results[0]
        assert isinstance(img, Image.Image)
        assert isinstance(label, str)

    def test_sample_positive_caches(self, sampler):
        r1 = sampler.sample_positive("goldfish", n=2)
        r2 = sampler.sample_positive("goldfish", n=2)
        # Second call should use cache (same list contents)
        assert len(r1) == len(r2)
        assert r1[0][1] == r2[0][1]  # Same label

    def test_sample_negative(self, sampler):
        results = sampler.sample_negative("goldfish", n=2)
        assert len(results) > 0
        _, label = results[0]
        assert "goldfish" not in label.lower()

    def test_sample_nonexistent_class(self, sampler):
        results = sampler.sample_positive("xyznonexistent123", n=2)
        assert results == []

    def test_sample_from_classes(self, sampler):
        results = sampler.sample_from_classes(["goldfish"], n_per_class=1)
        assert len(results) > 0

    def test_sample_from_classes_nonexistent(self, sampler):
        results = sampler.sample_from_classes(["xyznonexistent123"], n_per_class=1)
        assert results == []
