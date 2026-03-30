"""Tests for multi-model pipeline and comparison report."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import Template

from src.config import Config
from src.pipeline import (
    ClassAnalysisResult, EditResult, GenerationResult, MultiModelResult,
)
from src.reporter import (
    Reporter, _COMPARISON_HTML_TEMPLATE, _COMPARISON_MD_TEMPLATE,
)
from tests.conftest import make_edit_result, make_generation


# =========================================================================
# MultiModelResult
# =========================================================================

class TestMultiModelResult:
    def test_to_dict(self):
        r1 = ClassAnalysisResult(class_name="cat")
        r1.finalize()
        r2 = ClassAnalysisResult(class_name="cat")
        r2.finalize()
        mmr = MultiModelResult(class_name="cat", per_model={"resnet50": r1, "dinov2": r2})
        d = mmr.to_dict()
        assert d["class_name"] == "cat"
        assert "resnet50" in d["models"]
        assert "dinov2" in d["models"]

    def test_from_dict(self):
        d = {
            "class_name": "dog",
            "models": {
                "resnet50": ClassAnalysisResult(class_name="dog").to_dict(),
            },
        }
        mmr = MultiModelResult.from_dict(d)
        assert mmr.class_name == "dog"
        assert "resnet50" in mmr.per_model

    def test_empty_per_model(self):
        mmr = MultiModelResult(class_name="fish")
        d = mmr.to_dict()
        assert d["models"] == {}


# =========================================================================
# Config multi-model properties
# =========================================================================

class TestConfigMultiModel:
    def test_is_multi_model_true_default(self):
        cfg = Config(device="cpu")
        assert cfg.is_multi_model is True

    def test_is_multi_model_false_single(self):
        cfg = Config(device="cpu", classifier_models=["resnet50"])
        assert cfg.is_multi_model is False

    def test_is_multi_model_true_with_two(self):
        cfg = Config(device="cpu", classifier_models=["resnet50", "dinov2_vitb14_lc"])
        assert cfg.is_multi_model is True

    def test_images_dir(self):
        cfg = Config(device="cpu", output_dir=Path("/tmp/test"))
        assert cfg.images_dir == Path("/tmp/test/images")

    def test_checkpoints_dir(self):
        cfg = Config(device="cpu", output_dir=Path("/tmp/test"))
        assert cfg.checkpoints_dir == Path("/tmp/test/checkpoints")

    def test_reports_dir(self):
        cfg = Config(device="cpu", output_dir=Path("/tmp/test"))
        assert cfg.reports_dir == Path("/tmp/test/reports")

    def test_model_checkpoint_dir(self):
        cfg = Config(device="cpu", output_dir=Path("/tmp/test"))
        assert cfg.model_checkpoint_dir("resnet50") == Path("/tmp/test/checkpoints/resnet50")


# =========================================================================
# Comparison report rendering
# =========================================================================

class TestComparisonReport:
    @pytest.fixture
    def comparison_data(self):
        return {
            "model_names": ["resnet50", "dinov2"],
            "per_model": {
                "resnet50": {"classes": 2, "total_edits": 10, "total_confirmed": 5, "total_spurious": 2},
                "dinov2": {"classes": 2, "total_edits": 10, "total_confirmed": 3, "total_spurious": 1},
            },
            "per_class": [
                {
                    "class_name": "cat",
                    "models": {
                        "resnet50": {"confirmed": 3, "spurious": 1, "risk": "HIGH", "spurious_features": ["background"]},
                        "dinov2": {"confirmed": 2, "spurious": 0, "risk": "MEDIUM", "spurious_features": []},
                    },
                },
                {
                    "class_name": "dog",
                    "models": {
                        "resnet50": {"confirmed": 2, "spurious": 1, "risk": "MEDIUM", "spurious_features": ["grass"]},
                        "dinov2": {"confirmed": 1, "spurious": 1, "risk": "LOW", "spurious_features": ["grass"]},
                    },
                },
            ],
        }

    def test_html_renders(self, comparison_data):
        html = Template(_COMPARISON_HTML_TEMPLATE).render(
            models=["resnet50", "dinov2"], comparison=comparison_data, config={},
        )
        assert "Cross-Model Comparison Report" in html
        assert "resnet50" in html
        assert "dinov2" in html
        assert "cat" in html
        assert "dog" in html

    def test_md_renders(self, comparison_data):
        md = Template(_COMPARISON_MD_TEMPLATE).render(
            models=["resnet50", "dinov2"], comparison=comparison_data, config={},
        )
        assert "Cross-Model Comparison Report" in md
        assert "resnet50" in md

    def test_html_shows_spurious_counts(self, comparison_data):
        html = Template(_COMPARISON_HTML_TEMPLATE).render(
            models=["resnet50", "dinov2"], comparison=comparison_data, config={},
        )
        assert "background" in html
        assert "grass" in html

    def test_reporter_build_comparison_data(self, tmp_path):
        reporter = Reporter(tmp_path, config={})
        model_dicts = {
            "resnet50": [
                ClassAnalysisResult(class_name="cat").to_dict(),
            ],
            "dinov2": [
                ClassAnalysisResult(class_name="cat").to_dict(),
            ],
        }
        data = reporter._build_comparison_data(model_dicts, ["resnet50", "dinov2"])
        assert len(data["per_class"]) == 1
        assert data["per_class"][0]["class_name"] == "cat"
        assert "resnet50" in data["per_model"]

    def test_is_spurious_positive_contextual(self):
        e = {"feature_type": "contextual", "target_type": "positive", "mean_delta": -0.10}
        assert Reporter._is_spurious(e) is True

    def test_is_spurious_negative_addition(self):
        e = {"feature_type": "", "target_type": "negative", "mean_delta": 0.10}
        assert Reporter._is_spurious(e) is True

    def test_not_spurious_intrinsic(self):
        e = {"feature_type": "intrinsic", "target_type": "positive", "mean_delta": -0.20}
        assert Reporter._is_spurious(e) is False

    def test_is_spurious_empty_feature_type_context_addition(self):
        """edit_type fallback when feature_type is empty."""
        e = {"feature_type": "", "target_type": "positive",
             "edit_type": "context_addition", "mean_delta": -0.10}
        assert Reporter._is_spurious(e) is True

    def test_is_spurious_empty_feature_type_background_change(self):
        e = {"feature_type": "", "target_type": "positive",
             "edit_type": "background_change", "mean_delta": -0.10}
        assert Reporter._is_spurious(e) is True

    def test_not_spurious_empty_feature_type_modification(self):
        """Non-contextual edit_type should not be spurious even if feature_type empty."""
        e = {"feature_type": "", "target_type": "positive",
             "edit_type": "modification", "mean_delta": -0.10}
        assert Reporter._is_spurious(e) is False

    def test_is_spurious_state_dependent(self):
        """State-dependent features with negative delta are spurious."""
        e = {"feature_type": "state_dependent", "target_type": "positive",
             "mean_delta": -0.20}
        assert Reporter._is_spurious(e) is True

    def test_not_spurious_state_dependent_small_delta(self):
        e = {"feature_type": "state_dependent", "target_type": "positive",
             "mean_delta": -0.03}
        assert Reporter._is_spurious(e) is False

    def test_is_spurious_contextual_boost_positive(self):
        """Adding context to positive image that boosts confidence = spurious."""
        e = {"feature_type": "contextual", "target_type": "positive",
             "mean_delta": 0.15}
        assert Reporter._is_spurious(e) is True

    def test_not_spurious_contextual_small_boost(self):
        """Small contextual boost on positive is not spurious."""
        e = {"feature_type": "contextual", "target_type": "positive",
             "mean_delta": 0.05}
        assert Reporter._is_spurious(e) is False

    def test_not_spurious_body_part_mislabeled_contextual(self):
        """VLM mislabeled body part removal as contextual — not spurious."""
        e = {"feature_type": "contextual", "target_type": "positive",
             "instruction": "Remove the combs and wattles completely",
             "mean_delta": -0.50}
        assert Reporter._is_spurious(e) is False

    def test_is_spurious_background_removal(self):
        """Removing background that drops confidence IS spurious."""
        e = {"feature_type": "contextual", "target_type": "positive",
             "instruction": "Remove the boat entirely",
             "mean_delta": -0.50}
        assert Reporter._is_spurious(e) is True

    def test_is_spurious_context_removal(self):
        """Actual context removal that drops confidence IS spurious."""
        e = {"feature_type": "contextual", "target_type": "positive",
             "edit_type": "context_removal", "mean_delta": -0.30}
        assert Reporter._is_spurious(e) is True
