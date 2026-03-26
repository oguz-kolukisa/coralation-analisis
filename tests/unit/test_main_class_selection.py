"""Tests for class selection logic in main.py."""
from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from main import _select_n_classes, resolve_classes
from src.config import Config, load_classes_from_file, _DEFAULT_CLASS_FILE


ALL_NAMES = [f"class_{i}" for i in range(100)]
IMAGENET100_NAMES = load_classes_from_file(_DEFAULT_CLASS_FILE)


def _make_pipeline(all_names=ALL_NAMES):
    pipeline = MagicMock()
    pipeline.models.sampler.return_value.get_label_names.return_value = all_names
    return pipeline


class TestResolveClasses:
    def test_explicit_class_names(self):
        args = Namespace(class_names=["cat", "dog"], all=False, classes=20, random_classes=False)
        result = resolve_classes(args, _make_pipeline(), Config())
        assert result == ["cat", "dog"]

    def test_all_flag_json_source(self):
        args = Namespace(class_names=None, all=True, classes=20, random_classes=False)
        cfg = Config(class_source="json")
        result = resolve_classes(args, _make_pipeline(), cfg)
        assert result == IMAGENET100_NAMES

    def test_default_takes_first_n_json_source(self):
        args = Namespace(class_names=None, all=False, classes=5, random_classes=False)
        cfg = Config(class_source="json")
        result = resolve_classes(args, _make_pipeline(), cfg)
        assert result == IMAGENET100_NAMES[:5]

    def test_all_flag_dataset_source(self):
        args = Namespace(class_names=None, all=True, classes=20, random_classes=False)
        cfg = Config(class_source="dataset")
        result = resolve_classes(args, _make_pipeline(), cfg)
        assert result == ALL_NAMES

    def test_default_takes_first_n_dataset_source(self):
        args = Namespace(class_names=None, all=False, classes=5, random_classes=False)
        cfg = Config(class_source="dataset")
        result = resolve_classes(args, _make_pipeline(), cfg)
        assert result == ALL_NAMES[:5]


class TestSelectNClasses:
    def test_sequential_returns_first_n(self):
        cfg = Config(random_classes=False)
        result = _select_n_classes(ALL_NAMES, 10, cfg)
        assert result == ALL_NAMES[:10]

    def test_random_returns_correct_count(self):
        cfg = Config(random_classes=True, random_seed=42)
        result = _select_n_classes(ALL_NAMES, 10, cfg)
        assert len(result) == 10

    def test_random_differs_from_sequential(self):
        cfg = Config(random_classes=True, random_seed=42)
        result = _select_n_classes(ALL_NAMES, 10, cfg)
        assert result != ALL_NAMES[:10]

    def test_random_seed_is_deterministic(self):
        cfg = Config(random_classes=True, random_seed=99)
        r1 = _select_n_classes(ALL_NAMES, 10, cfg)
        r2 = _select_n_classes(ALL_NAMES, 10, cfg)
        assert r1 == r2

    def test_n_larger_than_list(self):
        cfg = Config(random_classes=True, random_seed=1)
        result = _select_n_classes(ALL_NAMES, 200, cfg)
        assert len(result) == len(ALL_NAMES)
