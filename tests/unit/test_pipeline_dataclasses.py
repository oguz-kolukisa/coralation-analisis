"""Tests for pipeline dataclasses: GenerationResult, EditResult, ClassAnalysisResult."""
from __future__ import annotations

import pytest

from src.pipeline import (
    ClassAnalysisResult, DiscoveredFeatures, EditResult, GenerationResult,
)
from tests.conftest import make_edit_result, make_generation


# =========================================================================
# GenerationResult.from_dict
# =========================================================================

class TestGenerationResultFromDict:
    def test_valid_full_dict(self):
        data = {"seed": 42, "edited_confidence": 0.75, "delta": -0.1, "edited_image_path": "a.jpg"}
        g = GenerationResult.from_dict(data)
        assert g.seed == 42
        assert g.delta == -0.1

    def test_extra_fields_ignored(self):
        data = {"seed": 1, "edited_confidence": 0.5, "delta": 0.0, "edited_image_path": "b.jpg", "bogus": 999}
        g = GenerationResult.from_dict(data)
        assert not hasattr(g, "bogus")

    def test_optional_fields_default(self):
        data = {"seed": 0, "edited_confidence": 0.9, "delta": -0.05, "edited_image_path": "c.jpg"}
        g = GenerationResult.from_dict(data)
        assert g.edit_verified is True
        assert g.verification_confidence == 1.0


# =========================================================================
# EditResult.from_dict
# =========================================================================

class TestEditResultFromDict:
    def test_roundtrip(self):
        original = make_edit_result(confirmed=True, delta=-0.3)
        from dataclasses import asdict
        data = asdict(original)
        restored = EditResult.from_dict(data)
        assert restored.instruction == original.instruction
        assert restored.confirmed == original.confirmed
        assert len(restored.generations) == len(original.generations)

    def test_empty_generations(self):
        data = {
            "instruction": "Remove X", "hypothesis": "H", "edit_type": "feature_removal",
            "target_type": "positive", "priority": 3, "original_confidence": 0.9,
            "original_image_path": "x.jpg", "generations": [],
        }
        er = EditResult.from_dict(data)
        assert er.generations == []

    def test_missing_generations_key(self):
        data = {
            "instruction": "Remove X", "hypothesis": "H", "edit_type": "feature_removal",
            "target_type": "positive", "priority": 3, "original_confidence": 0.9,
            "original_image_path": "x.jpg",
        }
        er = EditResult.from_dict(data)
        assert er.generations == []

    def test_nested_generations_converted(self):
        data = {
            "instruction": "Edit", "hypothesis": "H", "edit_type": "feature_removal",
            "target_type": "positive", "priority": 3, "original_confidence": 0.9,
            "original_image_path": "x.jpg",
            "generations": [
                {"seed": 1, "edited_confidence": 0.7, "delta": -0.2, "edited_image_path": "e1.jpg"},
                {"seed": 2, "edited_confidence": 0.6, "delta": -0.3, "edited_image_path": "e2.jpg"},
            ],
        }
        er = EditResult.from_dict(data)
        assert len(er.generations) == 2
        assert isinstance(er.generations[0], GenerationResult)
        assert er.generations[1].seed == 2


# =========================================================================
# EditResult properties
# =========================================================================

class TestEditResultProperties:
    def test_edited_confidence_property(self):
        er = make_edit_result(delta=-0.2)
        assert er.edited_confidence == er.mean_edited_confidence

    def test_delta_property(self):
        er = make_edit_result(delta=-0.15)
        assert er.delta == er.mean_delta

    def test_edited_image_path_with_generations(self):
        er = make_edit_result()
        assert er.edited_image_path == er.generations[0].edited_image_path

    def test_edited_image_path_empty_generations(self):
        er = EditResult(
            instruction="X", hypothesis="H", edit_type="t", target_type="p",
            priority=1, original_confidence=0.9, original_image_path="o.jpg",
            generations=[],
        )
        assert er.edited_image_path == ""


# =========================================================================
# ClassAnalysisResult
# =========================================================================

class TestClassAnalysisResultFinalize:
    def test_confirmed_hypotheses_populated(self, sample_class_result):
        sample_class_result.finalize()
        assert len(sample_class_result.confirmed_hypotheses) == 2

    def test_iterations_set(self, sample_class_result):
        sample_class_result.finalize()
        assert sample_class_result.iterations_completed == 1

    def test_key_features_from_intrinsic_high_attention(self, sample_class_result):
        sample_class_result.finalize()
        assert "ears" in sample_class_result.key_features
        assert "whiskers" in sample_class_result.key_features
        assert "background" not in sample_class_result.key_features

    def test_model_focus_from_gradcam_summary(self, sample_class_result):
        sample_class_result.finalize()
        assert sample_class_result.model_focus == "Focus on head region"

    def test_deduplicate_spurious_vs_essential(self, sample_class_result):
        sample_class_result.spurious_features = ["ears", "background_noise"]
        sample_class_result.finalize()
        assert "ears" not in sample_class_result.spurious_features
        assert "background_noise" in sample_class_result.spurious_features

    def test_no_key_features_when_already_set(self, sample_class_result):
        sample_class_result.key_features = ["custom_feature"]
        sample_class_result.finalize()
        assert sample_class_result.key_features == ["custom_feature"]

    def test_no_model_focus_when_already_set(self, sample_class_result):
        sample_class_result.model_focus = "Already set"
        sample_class_result.finalize()
        assert sample_class_result.model_focus == "Already set"

    def test_empty_edit_results(self):
        result = ClassAnalysisResult(class_name="test")
        result.finalize()
        assert result.confirmed_hypotheses == []
        assert result.iterations_completed == 1


class TestClassAnalysisResultToDict:
    def test_summary_fields(self, sample_class_result):
        sample_class_result.finalize()
        d = sample_class_result.to_dict()
        assert "summary" in d
        assert d["summary"]["total_edits"] == 3
        assert d["summary"]["confirmed_count"] == 2
        assert d["summary"]["confirmation_rate"] == round(2 / 3, 2)

    def test_zero_edits_no_division_error(self):
        result = ClassAnalysisResult(class_name="empty")
        d = result.to_dict()
        assert d["summary"]["confirmation_rate"] == 0.0
        assert d["summary"]["total_edits"] == 0

    def test_class_name_preserved(self, sample_class_result):
        d = sample_class_result.to_dict()
        assert d["class_name"] == "tabby cat"


class TestClassAnalysisResultFromDict:
    def test_roundtrip(self, sample_class_result):
        d = sample_class_result.to_dict()
        restored = ClassAnalysisResult.from_dict(dict(d))
        assert restored.class_name == sample_class_result.class_name
        assert len(restored.edit_results) == len(sample_class_result.edit_results)

    def test_confirmed_hypotheses_recalculated(self, sample_class_result):
        d = sample_class_result.to_dict()
        restored = ClassAnalysisResult.from_dict(dict(d))
        confirmed = [e for e in restored.edit_results if e.confirmed]
        assert len(restored.confirmed_hypotheses) == len(confirmed)

    def test_extra_keys_ignored(self):
        data = {"class_name": "test", "summary": {}, "nonexistent_field": True}
        result = ClassAnalysisResult.from_dict(data)
        assert result.class_name == "test"


class TestApplyDiscoveredFeatures:
    def test_fields_assigned(self):
        result = ClassAnalysisResult(class_name="test")
        discovered = DiscoveredFeatures(
            detected=[{"name": "fur"}], essential=["fur"], gradcam_summary="focus on fur",
        )
        result.apply_discovered_features(discovered)
        assert result.detected_features == [{"name": "fur"}]
        assert result.essential_features == ["fur"]
        assert result.gradcam_summary == "focus on fur"


class TestApplyFinalAnalysis:
    def test_none_is_noop(self):
        result = ClassAnalysisResult(class_name="test")
        result.apply_final_analysis(None)
        assert result.robustness_score == 5
        assert result.final_summary == ""

    def test_fields_absorbed(self):
        result = ClassAnalysisResult(class_name="test")

        class FakeFinal:
            feature_importance = [{"name": "ears", "impact": "high"}]
            robustness_score = 3
            risk_level = "HIGH"
            vulnerabilities = ["background bias"]
            recommendations = ["use diverse backgrounds"]
            summary = "Model relies on background"
            confirmed_shortcuts = [{"feature": "grass"}, {"feature": "sky"}]

        result.apply_final_analysis(FakeFinal())
        assert result.robustness_score == 3
        assert result.risk_level == "HIGH"
        assert result.final_summary == "Model relies on background"
        assert "grass" in result.confirmed_shortcuts
        assert "sky" in result.spurious_features

    def test_shortcuts_not_duplicated_in_spurious(self):
        result = ClassAnalysisResult(class_name="test")
        result.spurious_features = ["grass"]

        class FakeFinal:
            feature_importance = []
            robustness_score = 5
            risk_level = "LOW"
            vulnerabilities = []
            recommendations = []
            summary = ""
            confirmed_shortcuts = [{"feature": "grass"}]

        result.apply_final_analysis(FakeFinal())
        assert result.spurious_features.count("grass") == 1

    def test_string_shortcuts_handled(self):
        result = ClassAnalysisResult(class_name="test")

        class FakeFinal:
            feature_importance = []
            robustness_score = 5
            risk_level = "LOW"
            vulnerabilities = []
            recommendations = []
            summary = ""
            confirmed_shortcuts = ["plain_string"]

        result.apply_final_analysis(FakeFinal())
        assert "plain_string" in result.confirmed_shortcuts
