"""Tests for AnalysisPipeline helper functions."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import Config
from src.pipeline import (
    AnalysisPipeline, DiscoveredFeatures, EditInput, EditResult,
    GenerationResult,
)
from tests.conftest import make_edit_result, make_generation, make_image, make_instruction


@pytest.fixture
def pipeline(config):
    with patch("src.pipeline.ModelManager"):
        return AnalysisPipeline(config)


# =========================================================================
# _expected_direction
# =========================================================================

class TestExpectedDirection:
    def test_positive_removal_expects_negative(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._expected_direction(instr) == "negative"

    def test_negative_target_expects_positive(self, pipeline):
        instr = make_instruction(target="negative", edit_type="feature_removal")
        assert pipeline._expected_direction(instr) == "positive"

    def test_positive_addition_expects_positive(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_addition")
        assert pipeline._expected_direction(instr) == "positive"

    def test_unknown_type_expects_any(self, pipeline):
        instr = make_instruction(target="positive", edit_type="background_change")
        assert pipeline._expected_direction(instr) == "any"


# =========================================================================
# _validate_direction
# =========================================================================

class TestValidateDirection:
    def test_removal_negative_delta_valid(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._validate_direction(instr, -0.20) is True

    def test_removal_positive_delta_invalid(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._validate_direction(instr, 0.20) is False

    def test_removal_small_delta_invalid(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._validate_direction(instr, -0.05) is False

    def test_addition_positive_delta_valid(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_addition")
        assert pipeline._validate_direction(instr, 0.20) is True

    def test_negative_target_positive_delta_valid(self, pipeline):
        instr = make_instruction(target="negative", edit_type="feature_addition")
        assert pipeline._validate_direction(instr, 0.20) is True

    def test_negative_target_negative_delta_invalid(self, pipeline):
        instr = make_instruction(target="negative", edit_type="feature_addition")
        assert pipeline._validate_direction(instr, -0.20) is False

    def test_zero_delta_invalid(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._validate_direction(instr, 0.0) is False


# =========================================================================
# _deduplicate_inputs
# =========================================================================

class TestDeduplicateInputs:
    def _make_input(self, edit_text):
        img = make_image()
        instr = make_instruction(edit=edit_text)
        return EditInput(img, instr, 0.9)

    def test_empty_list(self, pipeline):
        assert pipeline._deduplicate_inputs([]) == []

    def test_identical_edits_deduplicated(self, pipeline):
        inputs = [self._make_input("Remove the ears"), self._make_input("Remove the ears")]
        result = pipeline._deduplicate_inputs(inputs)
        assert len(result) == 1

    def test_similar_edits_deduplicated(self, pipeline):
        inputs = [self._make_input("Remove the ears"), self._make_input("Remove ears")]
        result = pipeline._deduplicate_inputs(inputs)
        assert len(result) == 1

    def test_different_edits_kept(self, pipeline):
        inputs = [self._make_input("Remove the ears"), self._make_input("Replace background with white")]
        result = pipeline._deduplicate_inputs(inputs)
        assert len(result) == 2

    def test_case_insensitive(self, pipeline):
        inputs = [self._make_input("REMOVE EARS"), self._make_input("remove ears")]
        result = pipeline._deduplicate_inputs(inputs)
        assert len(result) == 1

    def test_punctuation_normalized(self, pipeline):
        inputs = [self._make_input("Remove the ears!"), self._make_input("Remove the ears")]
        result = pipeline._deduplicate_inputs(inputs)
        assert len(result) == 1


# =========================================================================
# _deduplicate_features_by_name
# =========================================================================

class TestDeduplicateFeaturesByName:
    def test_empty_list(self, pipeline):
        assert pipeline._deduplicate_features_by_name([]) == []

    def test_case_insensitive_dedup(self, pipeline):
        features = [
            {"name": "Ears", "category": "part"},
            {"name": "ears", "category": "part2"},
        ]
        result = pipeline._deduplicate_features_by_name(features)
        assert len(result) == 1
        assert result[0]["name"] == "Ears"

    def test_unique_features_preserved(self, pipeline):
        features = [
            {"name": "ears", "category": "part"},
            {"name": "tail", "category": "part"},
            {"name": "background", "category": "context"},
        ]
        result = pipeline._deduplicate_features_by_name(features)
        assert len(result) == 3

    def test_preserves_all_fields(self, pipeline):
        features = [{"name": "ears", "extra_field": "value"}]
        result = pipeline._deduplicate_features_by_name(features)
        assert result[0]["extra_field"] == "value"


# =========================================================================
# _match_feature_type
# =========================================================================

class TestMatchFeatureType:
    def test_exact_match(self, pipeline):
        types = {"ears": "intrinsic", "background": "contextual"}
        assert pipeline._match_feature_type("ears", types) == "intrinsic"

    def test_substring_match(self, pipeline):
        types = {"ears": "intrinsic"}
        assert pipeline._match_feature_type("ear", types) == "intrinsic"

    def test_reverse_substring_match(self, pipeline):
        types = {"fur": "intrinsic"}
        assert pipeline._match_feature_type("furry", types) == "intrinsic"

    def test_no_match_returns_unknown(self, pipeline):
        types = {"ears": "intrinsic"}
        assert pipeline._match_feature_type("sky", types) == "unknown"

    def test_empty_types_returns_unknown(self, pipeline):
        assert pipeline._match_feature_type("anything", {}) == "unknown"


# =========================================================================
# _threshold_sig
# =========================================================================

class TestThresholdSig:
    def test_confirmed_when_majority_and_not_failed(self, pipeline):
        result = pipeline._threshold_sig([-0.3, -0.25, -0.2], False, 3)
        assert result["confirmed"] is True

    def test_not_confirmed_when_likely_failed(self, pipeline):
        result = pipeline._threshold_sig([-0.3, -0.25], True, 2)
        assert result["confirmed"] is False

    def test_not_confirmed_when_minority(self, pipeline):
        result = pipeline._threshold_sig([-0.3, 0.1, 0.2], False, 1)
        assert result["confirmed"] is False

    def test_practically_significant_large_delta(self, pipeline):
        result = pipeline._threshold_sig([-0.5, -0.4], False, 2)
        assert result["practically_significant"] is True

    def test_not_practically_significant_small_delta(self, pipeline):
        result = pipeline._threshold_sig([-0.01, 0.01], False, 0)
        assert result["practically_significant"] is False

    def test_returns_expected_keys(self, pipeline):
        result = pipeline._threshold_sig([-0.1], False, 1)
        expected_keys = {"confirmed", "confirmation_count", "p_value", "cohens_d",
                         "effect_size", "statistically_significant", "practically_significant"}
        assert set(result.keys()) == expected_keys

    def test_static_values(self, pipeline):
        result = pipeline._threshold_sig([-0.1], False, 1)
        assert result["p_value"] == 1.0
        assert result["cohens_d"] == 0.0
        assert result["statistically_significant"] is False


# =========================================================================
# _compute_significance (dispatches to _statistical_sig or _threshold_sig)
# =========================================================================

class TestComputeSignificance:
    def test_uses_statistical_when_enabled_and_enough_samples(self, pipeline):
        instr = make_instruction()
        with patch.object(pipeline, "_statistical_sig", return_value={"confirmed": True}) as mock:
            pipeline._compute_significance([-0.3, -0.2, -0.1], instr, False, 3)
            mock.assert_called_once()

    def test_uses_threshold_when_disabled(self, pipeline):
        pipeline.cfg.use_statistical_validation = False
        instr = make_instruction()
        with patch.object(pipeline, "_threshold_sig", return_value={"confirmed": False}) as mock:
            pipeline._compute_significance([-0.3, -0.2], instr, False, 2)
            mock.assert_called_once()

    def test_uses_threshold_when_too_few_samples(self, pipeline):
        instr = make_instruction()
        with patch.object(pipeline, "_threshold_sig", return_value={"confirmed": False}) as mock:
            pipeline._compute_significance([-0.3], instr, False, 1)
            mock.assert_called_once()


# =========================================================================
# _build_edit_result
# =========================================================================

class TestBuildEditResult:
    def test_mean_delta_calculated(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.3), make_generation(delta=-0.1)]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        assert result.mean_delta == round((-0.3 + -0.1) / 2, 4)

    def test_std_delta_single_generation(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.2)]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        assert result.std_delta == 0.0

    def test_std_delta_multiple_generations(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.3), make_generation(delta=-0.1)]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        expected_std = round(float(np.std([-0.3, -0.1], ddof=1)), 4)
        assert result.std_delta == expected_std

    def test_likely_failed_near_zero_delta(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=0.005), make_generation(delta=-0.005)]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        assert result.likely_failed is True

    def test_not_likely_failed_large_delta(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.3)]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        assert result.likely_failed is False

    def test_instruction_fields_copied(self, pipeline):
        instr = make_instruction(edit="Remove the ears")
        gens = [make_generation()]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        assert result.instruction == "Remove the ears"
        assert result.edit_type == "feature_removal"
        assert result.original_confidence == 0.9

    def test_min_max_delta(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.5), make_generation(delta=-0.1), make_generation(delta=-0.3)]
        result = pipeline._build_edit_result(instr, 0.9, "orig.jpg", gens)
        assert result.min_delta == -0.5
        assert result.max_delta == -0.1


# =========================================================================
# _accumulate_discovery
# =========================================================================

class TestAccumulateDiscovery:
    def _make_discovery(self, summary="Focus on head", features=None, intrinsic=None):
        d = MagicMock()
        d.gradcam_summary = summary
        d.features = features or []
        d.intrinsic_features = intrinsic or []
        return d

    def test_first_summary_no_separator(self, pipeline):
        result = DiscoveredFeatures()
        discovery = self._make_discovery(summary="Head focus")
        pipeline._accumulate_discovery(discovery, 0, result)
        assert result.gradcam_summary == "Head focus"

    def test_subsequent_summary_has_separator(self, pipeline):
        result = DiscoveredFeatures(gradcam_summary="First")
        discovery = self._make_discovery(summary="Second")
        pipeline._accumulate_discovery(discovery, 1, result)
        assert result.gradcam_summary == "First | Second"

    def test_features_appended_with_index(self, pipeline):
        result = DiscoveredFeatures()
        feat = MagicMock(name="ears", category="part", feature_type="intrinsic", gradcam_attention="high")
        feat.name = "ears"
        feat.category = "part"
        feat.feature_type = "intrinsic"
        feat.gradcam_attention = "high"
        discovery = self._make_discovery(features=[feat])
        pipeline._accumulate_discovery(discovery, 3, result)
        assert len(result.detected) == 1
        assert result.detected[0]["source_image"] == 3
        assert result.detected[0]["name"] == "ears"

    def test_intrinsic_features_no_duplicates(self, pipeline):
        result = DiscoveredFeatures()
        result.essential = ["ears"]
        discovery = self._make_discovery(intrinsic=["ears", "whiskers"])
        pipeline._accumulate_discovery(discovery, 0, result)
        assert result.essential == ["ears", "whiskers"]


# =========================================================================
# _base_record
# =========================================================================

class TestBaseRecord:
    def test_all_fields_present(self, pipeline):
        from tests.conftest import FakePrediction
        pred = FakePrediction()
        record = pipeline._base_record("/path/img.jpg", "tabby cat", pred, 0.85)
        assert record["image_path"] == "/path/img.jpg"
        assert record["true_label"] == "tabby cat"
        assert record["predicted_label"] == "tabby cat"
        assert record["predicted_confidence"] == 0.85
        assert record["class_confidence"] == 0.85
        assert record["top_k"] == [("tabby cat", 0.85), ("tiger cat", 0.10)]

    def test_different_pred_and_true(self, pipeline):
        from tests.conftest import FakePrediction
        pred = FakePrediction(label_name="tiger cat", confidence=0.7)
        record = pipeline._base_record("/p.jpg", "tabby cat", pred, 0.3)
        assert record["predicted_label"] == "tiger cat"
        assert record["true_label"] == "tabby cat"
