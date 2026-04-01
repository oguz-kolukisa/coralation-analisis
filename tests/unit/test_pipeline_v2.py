"""Tests for pipeline_v2 auto-verdict logic."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline_v2 import (
    EditPlan, EditResultV2, ImageRecord, ModelMeasurement,
    PipelineV2, _obvious_verdict,
)


def _make_edit_result(feature_type, edit_type, target, deltas):
    """Build an EditResultV2 with per-model measurements."""
    plan = EditPlan(
        feature_name="test_feature", feature_type=feature_type,
        edit_type=edit_type, edit_instruction="test", target=target,
    )
    record = MagicMock(spec=ImageRecord)
    er = EditResultV2(edit=plan, image_record=record, edited_image_path="/tmp/x.jpg")
    for model, d in deltas.items():
        er.per_model[model] = ModelMeasurement(
            original_confidence=0.9, edited_confidence=0.9 + d, delta=d,
        )
    return er


# =========================================================================
# _obvious_verdict (free function)
# =========================================================================

class TestObviousVerdict:
    def test_target_removal_big_drop_is_essential(self):
        v = _obvious_verdict("target", "feature_removal", "positive", -0.30)
        assert v["verdict"] == "essential"

    def test_target_removal_small_drop_is_none(self):
        assert _obvious_verdict("target", "feature_removal", "positive", -0.05) is None

    def test_env_removal_big_drop_is_spurious(self):
        v = _obvious_verdict("environmental", "environment_removal", "positive", -0.20)
        assert v["verdict"] == "spurious"

    def test_env_change_big_drop_is_spurious(self):
        v = _obvious_verdict("environmental", "environment_change", "positive", -0.15)
        assert v["verdict"] == "spurious"

    def test_negative_big_increase_is_spurious(self):
        v = _obvious_verdict("negative", "negative_modify", "negative", 0.20)
        assert v["verdict"] == "spurious"

    def test_state_change_target_big_drop_is_state_bias(self):
        v = _obvious_verdict("target", "state_change", "positive", -0.25)
        assert v["verdict"] == "state_bias"

    def test_ambiguous_returns_none(self):
        assert _obvious_verdict("target", "environment_change", "positive", -0.20) is None


# =========================================================================
# _auto_verdict (per-model split)
# =========================================================================

def _make_pipeline_stub():
    """Build a minimal PipelineV2 stub with config for _auto_verdict."""
    from src.config import Config
    stub = MagicMock(spec=PipelineV2)
    stub.cfg = Config()
    stub._auto_verdict = PipelineV2._auto_verdict.__get__(stub, PipelineV2)
    return stub


class TestAutoVerdict:
    def test_all_obvious_returns_verdicts_no_ambiguous(self):
        pipe = _make_pipeline_stub()
        er = _make_edit_result("target", "feature_removal", "positive",
                               {"resnet": -0.30, "dino": -0.50})
        verdicts, ambiguous = pipe._auto_verdict(er)
        assert len(verdicts) == 2
        assert verdicts["resnet"]["verdict"] == "essential"
        assert verdicts["dino"]["verdict"] == "essential"
        assert ambiguous == []

    def test_mixed_returns_obvious_plus_ambiguous(self):
        pipe = _make_pipeline_stub()
        er = _make_edit_result("target", "feature_removal", "positive",
                               {"resnet": -0.50, "dino": -0.05})
        verdicts, ambiguous = pipe._auto_verdict(er)
        assert verdicts["resnet"]["verdict"] == "essential"
        assert "dino" in ambiguous
        assert "dino" not in verdicts

    def test_all_ambiguous_returns_empty_verdicts(self):
        pipe = _make_pipeline_stub()
        er = _make_edit_result("target", "feature_removal", "positive",
                               {"resnet": -0.05, "dino": -0.03})
        verdicts, ambiguous = pipe._auto_verdict(er)
        assert verdicts == {}
        assert set(ambiguous) == {"resnet", "dino"}

    def test_single_model_obvious(self):
        pipe = _make_pipeline_stub()
        er = _make_edit_result("environmental", "environment_removal", "positive",
                               {"resnet": -0.20})
        verdicts, ambiguous = pipe._auto_verdict(er)
        assert verdicts["resnet"]["verdict"] == "spurious"
        assert ambiguous == []

    def test_single_model_ambiguous(self):
        pipe = _make_pipeline_stub()
        er = _make_edit_result("environmental", "environment_removal", "positive",
                               {"resnet": -0.05})
        verdicts, ambiguous = pipe._auto_verdict(er)
        assert verdicts == {}
        assert ambiguous == ["resnet"]


# =========================================================================
# _fill_missing_verdicts
# =========================================================================

class TestFillMissingVerdicts:
    def test_fills_missing_models(self):
        er = _make_edit_result("target", "feature_removal", "positive",
                               {"resnet": -0.30, "dino": -0.50})
        er.verdict = {"resnet": {"verdict": "essential", "reasoning": "ok"}}
        PipelineV2._fill_missing_verdicts(er)
        assert "dino" in er.verdict
        assert er.verdict["dino"]["verdict"] == "unknown"

    def test_does_not_overwrite_existing(self):
        er = _make_edit_result("target", "feature_removal", "positive",
                               {"resnet": -0.30})
        er.verdict = {"resnet": {"verdict": "essential", "reasoning": "ok"}}
        PipelineV2._fill_missing_verdicts(er)
        assert er.verdict["resnet"]["verdict"] == "essential"

    def test_no_op_when_all_present(self):
        er = _make_edit_result("target", "feature_removal", "positive",
                               {"resnet": -0.30})
        er.verdict = {"resnet": {"verdict": "spurious", "reasoning": "test"}}
        PipelineV2._fill_missing_verdicts(er)
        assert len(er.verdict) == 1
