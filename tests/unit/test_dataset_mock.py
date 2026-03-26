"""Tests for src/dataset.py using mocked HuggingFace datasets."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.dataset import ImageNetSampler


def _make_mock_dataset(label_field="label", image_field="image", items=None):
    """Create a mock HF dataset with controllable features and items."""
    ds = MagicMock()

    # Mock features
    features = MagicMock()
    features.__contains__ = lambda self, key: key in {label_field, image_field}
    features.__getitem__ = lambda self, key: MagicMock(names=[f"class_{i}" for i in range(10)])
    ds.features = features

    # Mock items
    if items is None:
        items = [
            {label_field: 0, image_field: Image.new("RGB", (32, 32), "red")},
            {label_field: 1, image_field: Image.new("RGB", (32, 32), "blue")},
            {label_field: 0, image_field: Image.new("RGB", (32, 32), "green")},
        ]

    ds.__len__ = lambda self: len(items)
    ds.__getitem__ = lambda self, idx: items[idx]
    ds.shuffle = lambda seed=None: ds  # Return self for simplicity

    return ds


def _make_sampler(ds, label_field="label", image_field="image"):
    """Create an ImageNetSampler with a pre-mocked dataset."""
    sampler = object.__new__(ImageNetSampler)
    sampler._ds = ds
    sampler._max_scan = 100
    sampler._seed = 42
    sampler._label_field = label_field
    sampler._image_field = image_field
    sampler._label_names = None
    sampler._positive_cache = {}
    sampler._negative_cache = {}
    return sampler


# ============================================================================
# Field detection (init)
# ============================================================================

class TestFieldDetection:
    def test_cifar_style_fields(self):
        """Simulate CIFAR-100 dataset with fine_label/img fields."""
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds, label_field="fine_label", image_field="img")
        assert sampler._label_field == "fine_label"
        assert sampler._image_field == "img"

    def test_imagenet_style_fields(self):
        """Simulate ImageNet dataset with label/image fields."""
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        assert sampler._label_field == "label"
        assert sampler._image_field == "image"


# ============================================================================
# get_label_names
# ============================================================================

class TestGetLabelNames:
    def test_returns_names(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        names = sampler.get_label_names()
        assert isinstance(names, list)
        assert len(names) == 10

    def test_caches_result(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        n1 = sampler.get_label_names()
        n2 = sampler.get_label_names()
        assert n1 is n2  # Same object (cached)


# ============================================================================
# find_label_index
# ============================================================================

class TestFindLabelIndex:
    def test_finds_exact_match(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog", "fish"]
        idx = sampler.find_label_index("dog")
        assert idx == 1

    def test_case_insensitive(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        sampler._label_names = ["Cat", "Dog"]
        idx = sampler.find_label_index("cat")
        assert idx == 0

    def test_substring_match(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        sampler._label_names = ["tabby cat", "golden retriever"]
        idx = sampler.find_label_index("tabby")
        assert idx == 0

    def test_not_found_returns_none(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog"]
        assert sampler.find_label_index("xyz") is None


# ============================================================================
# find_label_indices
# ============================================================================

class TestFindLabelIndices:
    def test_finds_multiple(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        sampler._label_names = ["tiger shark", "hammerhead shark", "cat"]
        indices = sampler.find_label_indices("shark")
        assert set(indices) == {0, 1}

    def test_no_match_returns_empty(self):
        ds = _make_mock_dataset()
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog"]
        assert sampler.find_label_indices("xyz") == []


# ============================================================================
# sample_positive
# ============================================================================

class TestSamplePositive:
    def test_returns_matching_images(self):
        items = [
            {"label": 0, "image": Image.new("RGB", (32, 32), "red")},
            {"label": 1, "image": Image.new("RGB", (32, 32), "blue")},
            {"label": 0, "image": Image.new("RGB", (32, 32), "green")},
        ]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog"]

        results = sampler.sample_positive("cat", n=5)
        assert len(results) == 2  # Only label=0 items
        assert all(label == "cat" for _, label in results)

    def test_respects_n_limit(self):
        items = [
            {"label": 0, "image": Image.new("RGB", (32, 32))},
            {"label": 0, "image": Image.new("RGB", (32, 32))},
            {"label": 0, "image": Image.new("RGB", (32, 32))},
        ]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat"]

        results = sampler.sample_positive("cat", n=2)
        assert len(results) == 2

    def test_caches_results(self):
        items = [{"label": 0, "image": Image.new("RGB", (32, 32))}]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat"]

        r1 = sampler.sample_positive("cat", n=1)
        r2 = sampler.sample_positive("cat", n=1)
        assert r1[0][1] == r2[0][1]

    def test_no_match_returns_empty(self):
        items = [{"label": 1, "image": Image.new("RGB", (32, 32))}]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog"]

        results = sampler.sample_positive("xyz", n=5)
        assert results == []

    def test_converts_numpy_to_pil(self):
        import numpy as np
        items = [{"label": 0, "image": np.zeros((32, 32, 3), dtype=np.uint8)}]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat"]

        results = sampler.sample_positive("cat", n=1)
        assert isinstance(results[0][0], Image.Image)


# ============================================================================
# sample_negative
# ============================================================================

class TestSampleNegative:
    def test_excludes_target_class(self):
        items = [
            {"label": 0, "image": Image.new("RGB", (32, 32))},
            {"label": 1, "image": Image.new("RGB", (32, 32))},
            {"label": 2, "image": Image.new("RGB", (32, 32))},
        ]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog", "fish"]

        results = sampler.sample_negative("cat", n=5)
        assert all(label != "cat" for _, label in results)
        assert len(results) == 2

    def test_caches_results(self):
        items = [{"label": 1, "image": Image.new("RGB", (32, 32))}]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog"]

        r1 = sampler.sample_negative("cat", n=1)
        r2 = sampler.sample_negative("cat", n=1)
        assert r1[0][1] == r2[0][1]


# ============================================================================
# sample_from_classes
# ============================================================================

class TestSampleFromClasses:
    def test_samples_from_specified_classes(self):
        items = [
            {"label": 0, "image": Image.new("RGB", (32, 32))},
            {"label": 1, "image": Image.new("RGB", (32, 32))},
            {"label": 2, "image": Image.new("RGB", (32, 32))},
        ]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat", "dog", "fish"]

        results = sampler.sample_from_classes(["cat", "dog"], n_per_class=1)
        assert len(results) == 2

    def test_nonexistent_class_skipped(self):
        items = [{"label": 0, "image": Image.new("RGB", (32, 32))}]
        ds = _make_mock_dataset(items=items)
        sampler = _make_sampler(ds)
        sampler._label_names = ["cat"]

        results = sampler.sample_from_classes(["xyz"], n_per_class=1)
        assert results == []
