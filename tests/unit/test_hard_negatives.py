"""Tests for src/analysis/hard_negatives.py — HardNegativeMiner."""
from __future__ import annotations

from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.analysis.hard_negatives import HardNegative, HardNegativeMiner


# ============================================================================
# Helpers
# ============================================================================

class FakeResult(NamedTuple):
    label_name: str = "dog"
    confidence: float = 0.9


def _make_images(labels_confs: list[tuple[str, float]]):
    """Create an iterator of (image, label) pairs with matching classifier responses."""
    images = []
    for label, _ in labels_confs:
        images.append((Image.new("RGB", (32, 32), "red"), label))
    return images


def _make_classifier(labels_confs: list[tuple[str, float]]):
    """Create a mock classifier that returns confidence based on the call order."""
    clf = MagicMock()
    confs = [c for _, c in labels_confs]
    clf.get_class_confidence.side_effect = confs
    clf.predict.return_value = FakeResult()
    return clf


# ============================================================================
# HardNegative dataclass
# ============================================================================

class TestHardNegativeDataclass:
    def test_creates_instance(self):
        img = Image.new("RGB", (32, 32))
        hn = HardNegative(img, "dog", 0.15, "cat", 0.85)
        assert hn.true_label == "dog"
        assert hn.target_class_confidence == 0.15


# ============================================================================
# mine()
# ============================================================================

class TestMine:
    def test_finds_hard_negatives_in_range(self):
        # 3 images: dog@0.10, cat@0.50 (too high), bird@0.20
        images = _make_images([("dog", 0.10), ("cat", 0.50), ("bird", 0.20)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.10, 0.50, 0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10)
        assert len(results) == 2  # dog@0.10 and bird@0.20

    def test_skips_target_class(self):
        images = _make_images([("target", 0.30), ("dog", 0.20)])
        clf = MagicMock()
        # Only dog's confidence is checked (target is skipped)
        clf.get_class_confidence.side_effect = [0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10)
        assert len(results) == 1
        assert results[0].true_label == "dog"

    def test_case_insensitive_skip(self):
        images = _make_images([("Target", 0.30), ("dog", 0.20)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10)
        assert len(results) == 1

    def test_respects_n_samples_limit(self):
        images = _make_images([("a", 0.10), ("b", 0.20), ("c", 0.15)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.10, 0.20, 0.15]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=2)
        assert len(results) == 2

    def test_respects_max_scan_limit(self):
        images = _make_images([("a", 0.10), ("b", 0.20), ("c", 0.15)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.10]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10, max_scan=1)
        assert len(results) == 1

    def test_sorted_by_confidence_desc(self):
        images = _make_images([("a", 0.10), ("b", 0.35), ("c", 0.20)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.10, 0.35, 0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10)
        confs = [r.target_class_confidence for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_excludes_below_min(self):
        images = _make_images([("a", 0.01)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.01]

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10)
        assert len(results) == 0

    def test_excludes_above_max(self):
        images = _make_images([("a", 0.60)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.60]

        miner = HardNegativeMiner(clf, min_confidence=0.05, max_confidence=0.40)
        results = miner.mine(iter(images), "target", n_samples=10)
        assert len(results) == 0

    def test_empty_iterator(self):
        clf = MagicMock()
        miner = HardNegativeMiner(clf)
        results = miner.mine(iter([]), "target", n_samples=10)
        assert results == []


# ============================================================================
# categorize_negatives()
# ============================================================================

class TestCategorizeNegatives:
    def test_buckets_correctly(self):
        images = _make_images([
            ("a", 0.35),  # very_hard
            ("b", 0.20),  # hard
            ("c", 0.10),  # medium
            ("d", 0.02),  # easy
        ])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.35, 0.20, 0.10, 0.02]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target")
        assert len(cats["very_hard"]) == 1
        assert len(cats["hard"]) == 1
        assert len(cats["medium"]) == 1
        assert len(cats["easy"]) == 1

    def test_skips_target_class(self):
        images = _make_images([("target", 0.90), ("dog", 0.20)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target")
        total = sum(len(v) for v in cats.values())
        assert total == 1

    def test_respects_max_scan(self):
        images = _make_images([("a", 0.10), ("b", 0.20), ("c", 0.30)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.10, 0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target", max_scan=2)
        total = sum(len(v) for v in cats.values())
        assert total == 2

    def test_sorted_within_categories(self):
        images = _make_images([("a", 0.16), ("b", 0.25), ("c", 0.20)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.16, 0.25, 0.20]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target")
        hard = cats["hard"]
        confs = [h.target_class_confidence for h in hard]
        assert confs == sorted(confs, reverse=True)

    def test_empty_iterator(self):
        clf = MagicMock()
        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter([]), "target")
        assert all(len(v) == 0 for v in cats.values())

    def test_boundary_030(self):
        """0.30 exactly should go to very_hard."""
        images = _make_images([("a", 0.30)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.30]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target")
        assert len(cats["very_hard"]) == 1

    def test_boundary_015(self):
        """0.15 exactly should go to hard."""
        images = _make_images([("a", 0.15)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.15]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target")
        assert len(cats["hard"]) == 1

    def test_boundary_005(self):
        """0.05 exactly should go to medium."""
        images = _make_images([("a", 0.05)])
        clf = MagicMock()
        clf.get_class_confidence.side_effect = [0.05]
        clf.predict.return_value = FakeResult()

        miner = HardNegativeMiner(clf)
        cats = miner.categorize_negatives(iter(images), "target")
        assert len(cats["medium"]) == 1
