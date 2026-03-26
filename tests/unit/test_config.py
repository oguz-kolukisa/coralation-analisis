"""Tests for Config and related utilities."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config, get_config, load_hf_token


# =========================================================================
# Config defaults
# =========================================================================

class TestConfigDefaults:
    def test_default_classifier(self):
        cfg = Config()
        assert cfg.classifier_model == "resnet50"

    def test_default_device(self):
        cfg = Config()
        assert cfg.device == "cuda"

    def test_default_samples(self):
        cfg = Config()
        assert cfg.samples_per_class == 100
        assert cfg.top_negative_classes == 5
        assert cfg.negative_samples_per_class == 5
        assert cfg.inspect_samples == 10

    def test_default_thresholds(self):
        cfg = Config()
        assert cfg.confidence_delta_threshold == 0.15
        assert cfg.statistical_alpha == 0.05
        assert cfg.min_effect_size == 0.5
        assert cfg.dedup_similarity_threshold == 0.70

    def test_default_output_dir(self):
        cfg = Config()
        assert cfg.output_dir == Path("output")

    def test_default_low_vram(self):
        cfg = Config()
        assert cfg.low_vram is True


# =========================================================================
# Config overrides
# =========================================================================

class TestConfigOverrides:
    def test_override_device(self):
        cfg = Config(device="cpu")
        assert cfg.device == "cpu"

    def test_override_samples(self):
        cfg = Config(samples_per_class=10, top_negative_classes=3, negative_samples_per_class=8)
        assert cfg.samples_per_class == 10
        assert cfg.top_negative_classes == 3
        assert cfg.negative_samples_per_class == 8

    def test_override_output_dir(self):
        cfg = Config(output_dir=Path("/tmp/custom"))
        assert cfg.output_dir == Path("/tmp/custom")

    def test_override_generations(self):
        cfg = Config(generations_per_edit=5)
        assert cfg.generations_per_edit == 5


# =========================================================================
# Config roundtrip
# =========================================================================

class TestConfigRoundtrip:
    def test_model_dump_returns_dict(self):
        cfg = Config()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "classifier_model" in d

    def test_model_validate_roundtrip(self):
        cfg = Config(samples_per_class=7)
        d = cfg.model_dump()
        restored = Config.model_validate(d)
        assert restored.samples_per_class == 7
        assert restored.classifier_model == cfg.classifier_model


# =========================================================================
# get_config factory
# =========================================================================

class TestGetConfig:
    def test_returns_config(self):
        cfg = get_config()
        assert isinstance(cfg, Config)

    def test_applies_overrides(self):
        cfg = get_config(samples_per_class=20)
        assert cfg.samples_per_class == 20

    def test_multiple_overrides(self):
        cfg = get_config(device="cpu", low_vram=False, samples_per_class=1)
        assert cfg.device == "cpu"
        assert cfg.low_vram is False
        assert cfg.samples_per_class == 1


# =========================================================================
# load_hf_token
# =========================================================================

class TestLoadHfToken:
    def test_reads_valid_token(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text("hf_abc123\n")
        assert load_hf_token(token_file) == "hf_abc123"

    def test_ignores_comments(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text("# This is a comment\nhf_real_token\n")
        assert load_hf_token(token_file) == "hf_real_token"

    def test_ignores_blank_lines(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text("\n\n\nhf_token123\n")
        assert load_hf_token(token_file) == "hf_token123"

    def test_ignores_placeholder(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text("hf_REPLACE_WITH_YOUR_TOKEN\n")
        with patch.dict(os.environ, {}, clear=True):
            assert load_hf_token(token_file) is None

    def test_only_comments_returns_none(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text("# comment 1\n# comment 2\n")
        with patch.dict(os.environ, {}, clear=True):
            assert load_hf_token(token_file) is None

    def test_missing_file_falls_back_to_env(self, tmp_path):
        token_file = tmp_path / "nonexistent"
        with patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
            assert load_hf_token(token_file) == "hf_from_env"

    def test_missing_file_no_env_returns_none(self, tmp_path):
        token_file = tmp_path / "nonexistent"
        with patch.dict(os.environ, {}, clear=True):
            assert load_hf_token(token_file) is None


# =========================================================================
# random_classes config field
# =========================================================================

class TestRandomClassesConfig:
    def test_default_is_false(self):
        cfg = Config()
        assert cfg.random_classes is False

    def test_override_to_true(self):
        cfg = Config(random_classes=True)
        assert cfg.random_classes is True
