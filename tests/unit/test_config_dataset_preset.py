"""Tests for src/config.py dataset preset logic."""
from __future__ import annotations

import pytest

from src.config import Config, _DATASET_PRESETS, apply_dataset_preset, get_config


class TestDatasetPresets:
    def test_imagenet_preset_has_expected_fields(self):
        preset = _DATASET_PRESETS["imagenet"]
        assert preset["hf_dataset"] == "ILSVRC/imagenet-1k"
        assert preset["hf_dataset_split"] == "validation"
        assert preset["class_source"] == "json"

    def test_cub_preset_has_expected_fields(self):
        preset = _DATASET_PRESETS["cub"]
        assert preset["hf_dataset"] == "bentrevett/caltech-ucsd-birds-200-2011"
        assert preset["hf_dataset_split"] == "test"
        assert preset["class_source"] == "dataset"


class TestApplyDatasetPreset:
    def test_default_dataset_is_imagenet(self):
        merged = apply_dataset_preset({})
        assert merged["hf_dataset"] == "ILSVRC/imagenet-1k"

    def test_cub_dataset_applies_preset(self):
        merged = apply_dataset_preset({"dataset": "cub"})
        assert merged["hf_dataset_split"] == "test"
        assert merged["hf_dataset"].endswith("caltech-ucsd-birds-200-2011")

    def test_user_override_wins(self):
        merged = apply_dataset_preset({
            "dataset": "cub",
            "hf_dataset_split": "train",   # user explicitly overrides
        })
        assert merged["hf_dataset_split"] == "train"

    def test_unknown_dataset_treated_as_no_preset(self):
        merged = apply_dataset_preset({"dataset": "mystery"})
        # No preset applied; user fields only
        assert "hf_dataset" not in merged or merged.get("hf_dataset") != "ILSVRC/imagenet-1k"


class TestGetConfig:
    def test_imagenet_is_default(self):
        cfg = get_config()
        assert cfg.dataset == "imagenet"
        assert cfg.hf_dataset == "ILSVRC/imagenet-1k"
        assert cfg.class_source == "json"

    def test_cub_wires_test_split(self):
        cfg = get_config(dataset="cub")
        assert cfg.dataset == "cub"
        assert cfg.hf_dataset_split == "test"
        assert cfg.class_source == "dataset"

    def test_cub_still_accepts_custom_samples(self):
        cfg = get_config(dataset="cub", samples_per_class=2)
        assert cfg.samples_per_class == 2


class TestConfigDatasetField:
    def test_default_is_imagenet(self):
        assert Config().dataset == "imagenet"

    def test_accepts_cub_value(self):
        cfg = Config(dataset="cub")
        assert cfg.dataset == "cub"
