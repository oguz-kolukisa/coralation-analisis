"""Tests for VLM parsing and helper methods (no GPU needed).

Tests _parse_*, _repair_json, _fallback_classify, select_confusing_classes,
and other methods that can be tested by mocking _run().
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.vlm import (
    DetectedFeature,
    EditInstruction,
    FeatureDiscovery,
    FeatureEditPlan,
    FinalAnalysis,
    QwenVLAnalyzer,
    VLMAnalysis,
)


def _make_analyzer():
    """Create a QwenVLAnalyzer without loading the model."""
    obj = object.__new__(QwenVLAnalyzer)
    obj.loaded = False
    return obj


# ============================================================================
# _repair_json — thorough edge cases
# ============================================================================

class TestRepairJsonEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_nested_trailing_commas(self, analyzer):
        bad = '{"a": [1, 2,], "b": {"c": 3,},}'
        result = json.loads(analyzer._repair_json(bad))
        assert result["a"] == [1, 2]
        assert result["b"]["c"] == 3

    def test_multiple_unquoted_keys(self, analyzer):
        bad = '{name: "test", value: 42}'
        result = json.loads(analyzer._repair_json(bad))
        assert result["name"] == "test"

    def test_truncated_object_in_array(self, analyzer):
        bad = '{"items": [1, 2, {"name": "hello"'
        repaired = analyzer._repair_json(bad)
        result = json.loads(repaired)
        assert "items" in result

    def test_truncated_array(self, analyzer):
        bad = '{"list": [1, 2, 3'
        repaired = analyzer._repair_json(bad)
        result = json.loads(repaired)
        assert "list" in result

    def test_control_characters_removed(self, analyzer):
        bad = '{"a": "hello\x00world"}'
        repaired = analyzer._repair_json(bad)
        result = json.loads(repaired)
        assert "hello" in result["a"]

    def test_deeply_nested_truncation(self, analyzer):
        bad = '{"a": {"b": [1, {"c": '
        repaired = analyzer._repair_json(bad)
        # Should not raise
        json.loads(repaired)

    def test_already_valid_complex_json(self, analyzer):
        good = json.dumps({"a": [1, 2], "b": {"c": True}, "d": None})
        assert json.loads(analyzer._repair_json(good)) == json.loads(good)


# ============================================================================
# _parse_feature_edits
# ============================================================================

class TestParseFeatureEdits:
    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_parses_feature_edits(self, analyzer):
        raw = json.dumps({
            "feature_edits": [
                {"feature_name": "ears", "edit_instruction": "Remove ears",
                 "edit_type": "removal", "expected_impact": "high",
                 "hypothesis": "ears are key"},
            ]
        })
        result = analyzer._parse_feature_edits(raw)
        assert len(result) == 1
        assert isinstance(result[0], FeatureEditPlan)
        assert result[0].feature_name == "ears"

    def test_parses_compound_edits(self, analyzer):
        raw = json.dumps({
            "feature_edits": [],
            "compound_edits": [
                {"features": ["ears", "eyes"], "edit_instruction": "Remove both",
                 "hypothesis": "test"},
            ],
        })
        result = analyzer._parse_feature_edits(raw)
        assert len(result) == 1
        assert result[0].feature_name == "ears+eyes"
        assert result[0].edit_type == "compound"

    def test_no_json_returns_empty(self, analyzer):
        assert analyzer._parse_feature_edits("no json here") == []

    def test_malformed_json_repaired(self, analyzer):
        raw = '{"feature_edits": [{"feature_name": "bg", "edit_instruction": "remove", "edit_type": "removal", "expected_impact": "high", "hypothesis": "test",}]}'
        result = analyzer._parse_feature_edits(raw)
        assert len(result) == 1

    def test_missing_fields_use_defaults(self, analyzer):
        raw = json.dumps({"feature_edits": [{}]})
        result = analyzer._parse_feature_edits(raw)
        assert len(result) == 1
        assert result[0].feature_name == ""
        assert result[0].edit_type == "removal"
        assert result[0].expected_impact == "medium"


# ============================================================================
# _parse_feature_discovery
# ============================================================================

class TestParseFeatureDiscoveryEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_missing_detected_features_key(self, analyzer):
        raw = json.dumps({"gradcam_summary": "focus on head"})
        result = analyzer._parse_feature_discovery(raw, "cat")
        assert result.gradcam_summary == "focus on head"
        assert len(result.features) == 0

    def test_feature_missing_fields_uses_defaults(self, analyzer):
        raw = json.dumps({
            "detected_features": [{"name": "ears"}],
        })
        result = analyzer._parse_feature_discovery(raw, "cat")
        assert result.features[0].category == "unknown"
        assert result.features[0].feature_type == "intrinsic"
        assert result.features[0].gradcam_attention == "medium"

    def test_multiple_features_parsed(self, analyzer):
        raw = json.dumps({
            "detected_features": [
                {"name": "ears", "category": "part", "feature_type": "intrinsic",
                 "location": "top", "gradcam_attention": "high", "reasoning": "pointy"},
                {"name": "bg", "category": "context", "feature_type": "contextual",
                 "location": "all", "gradcam_attention": "low", "reasoning": "green"},
            ],
            "intrinsic_features": ["ears"],
            "contextual_features": ["bg"],
        })
        result = analyzer._parse_feature_discovery(raw, "cat")
        assert len(result.features) == 2
        assert result.intrinsic_features == ["ears"]
        assert result.contextual_features == ["bg"]


# ============================================================================
# _parse_final_analysis — edge cases
# ============================================================================

class TestParseFinalAnalysisEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_partial_fields(self, analyzer):
        raw = json.dumps({"robustness_score": 3, "risk_level": "HIGH"})
        result = analyzer._parse_final_analysis(raw, "cat")
        assert result.robustness_score == 3
        assert result.risk_level == "HIGH"
        assert result.feature_importance == []

    def test_completely_invalid_json(self, analyzer):
        result = analyzer._parse_final_analysis("not json at all !!!", "cat")
        assert isinstance(result, FinalAnalysis)
        assert result.robustness_score == 5  # default


# ============================================================================
# _parse_analysis — edge cases
# ============================================================================

class TestParseAnalysisEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_instructions_sorted_by_priority(self, analyzer):
        raw = json.dumps({
            "edit_instructions": [
                {"edit": "a", "hypothesis": "h", "type": "t", "priority": 1},
                {"edit": "b", "hypothesis": "h", "type": "t", "priority": 5},
                {"edit": "c", "hypothesis": "h", "type": "t", "priority": 3},
            ],
        })
        result = analyzer._parse_analysis(raw, "cat", "positive")
        priorities = [i.priority for i in result.edit_instructions]
        assert priorities == [5, 3, 1]

    def test_image_index_set_from_enumerate(self, analyzer):
        raw = json.dumps({
            "edit_instructions": [
                {"edit": "a", "hypothesis": "h", "type": "t", "priority": 3},
                {"edit": "b", "hypothesis": "h", "type": "t", "priority": 3},
            ],
        })
        result = analyzer._parse_analysis(raw, "cat", "positive")
        indices = {i.image_index for i in result.edit_instructions}
        assert indices == {0, 1}

    def test_json_decode_error_returns_empty_analysis(self, analyzer):
        result = analyzer._parse_analysis("{invalid json!!!}", "cat", "positive")
        assert isinstance(result, VLMAnalysis)
        assert result.edit_instructions == []


# ============================================================================
# _parse_iterative_analysis — edge cases
# ============================================================================

class TestParseIterativeEdgeCases:
    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_all_targets_are_positive(self, analyzer):
        raw = json.dumps({
            "edit_instructions": [
                {"edit": "a", "hypothesis": "h", "type": "t", "priority": 3},
            ],
        })
        result = analyzer._parse_iterative_analysis(raw, "cat")
        assert result.edit_instructions[0].target == "positive"

    def test_json_error_returns_empty(self, analyzer):
        result = analyzer._parse_iterative_analysis("{bad}", "cat")
        assert result.edit_instructions == []


# ============================================================================
# FeatureDiscovery properties
# ============================================================================

class TestFeatureDiscoveryProperties:
    def test_potential_shortcuts_returns_contextual(self):
        fd = FeatureDiscovery(class_name="cat", contextual_features=["bg", "lighting"])
        assert fd.potential_shortcuts == ["bg", "lighting"]

    def test_robust_features_returns_intrinsic(self):
        fd = FeatureDiscovery(class_name="cat", intrinsic_features=["ears", "fur"])
        assert fd.robust_features == ["ears", "fur"]


# ============================================================================
# Methods that call _run() — mock _run
# ============================================================================

class TestSelectConfusingClasses:
    def test_returns_valid_classes(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "confusing_classes": ["dog", "wolf"],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.select_confusing_classes(
            "cat", ["dog", "wolf", "fish", "bird"], num_classes=2
        )
        assert result == ["dog", "wolf"]

    def test_filters_invalid_classes(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "confusing_classes": ["dog", "nonexistent"],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.select_confusing_classes(
            "cat", ["dog", "wolf"], num_classes=5
        )
        assert result == ["dog"]

    def test_fallback_on_failure(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(side_effect=Exception("VLM failed"))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.select_confusing_classes(
            "cat", ["dog", "wolf", "fish"], num_classes=2
        )
        assert len(result) == 2
        assert all(c in ["dog", "wolf", "fish"] for c in result)

    def test_limits_class_list_for_long_lists(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value='{"confusing_classes": ["c_0"]}')
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        classes = [f"c_{i}" for i in range(500)]
        result = analyzer.select_confusing_classes("cat", classes, num_classes=1)
        assert len(result) <= 1


class TestGenerateKnowledgeBasedFeatures:
    def test_returns_features(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "knowledge_based_features": [
                {"feature": "grass", "category": "environment"},
            ],
            "potential_edit_instructions": [],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.generate_knowledge_based_features("cat")
        assert len(result["knowledge_based_features"]) == 1

    def test_handles_failure(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(side_effect=Exception("fail"))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.generate_knowledge_based_features("cat")
        assert result["knowledge_based_features"] == []

    def test_with_class_names_context(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "knowledge_based_features": [],
            "potential_edit_instructions": [],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.generate_knowledge_based_features("cat", all_class_names=["dog", "cat"])
        assert result["target_class"] == "cat"

    def test_no_json_in_response(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value="no json here")
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        result = analyzer.generate_knowledge_based_features("cat")
        assert result["knowledge_based_features"] == []


class TestDiscoverFeaturesMocked:
    def test_calls_run_and_parses(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "detected_features": [
                {"name": "ears", "category": "part", "feature_type": "intrinsic",
                 "location": "top", "gradcam_attention": "high", "reasoning": "pointy"},
            ],
            "gradcam_summary": "head focus",
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        gc = Image.new("RGB", (64, 64))
        result = analyzer.discover_features(img, gc, "cat", 0.9)
        assert isinstance(result, FeatureDiscovery)
        assert len(result.features) == 1


class TestGenerateFeatureEditsMocked:
    def test_calls_run_and_parses(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "feature_edits": [
                {"feature_name": "bg", "edit_instruction": "Remove bg",
                 "edit_type": "removal", "expected_impact": "high", "hypothesis": "h"},
            ],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        features = [DetectedFeature("bg", "ctx", "contextual", "all", "high", "green")]
        result = analyzer.generate_feature_edits(img, features, "cat")
        assert len(result) == 1


class TestFinalAnalysisMocked:
    def test_calls_run_and_parses(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "feature_importance": [{"feature": "bg", "impact": "-0.3"}],
            "confirmed_shortcuts": [],
            "legitimate_features": ["ears"],
            "robustness_score": 8,
            "risk_level": "LOW",
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        result = analyzer.final_analysis(img, "cat", [{"feature": "bg", "delta": -0.3}])
        assert result.robustness_score == 8


class TestAnalyzePositiveMocked:
    def test_with_gradcam(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "key_features": ["ears"],
            "edit_instructions": [
                {"edit": "Remove ears", "hypothesis": "h", "type": "feature_removal", "priority": 5},
            ],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        gc = Image.new("RGB", (64, 64))
        result = analyzer.analyze_positive(img, gc, "cat", 0.9)
        assert result.edit_instructions[0].target == "positive"

    def test_without_gradcam(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "edit_instructions": [],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        result = analyzer.analyze_positive(img, None, "cat", 0.9)
        assert isinstance(result, VLMAnalysis)


class TestAnalyzeNegativeMocked:
    def test_returns_negative_target(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "common_mistake_triggers": ["grass"],
            "edit_instructions": [
                {"edit": "Add grass", "hypothesis": "h", "type": "feature_addition", "priority": 4},
            ],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        result = analyzer.analyze_negative(img, "cat", "dog", 0.3)
        assert result.edit_instructions[0].target == "negative"


class TestAnalyzeIterativeMocked:
    def test_returns_insights(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "insights": ["bg matters"],
            "confirmed_shortcuts": ["bg"],
            "needs_more_testing": ["texture"],
            "edit_instructions": [
                {"edit": "Remove bg", "hypothesis": "h", "type": "background_change", "priority": 5},
            ],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        edited = [Image.new("RGB", (64, 64))]
        prev = [{"edit": "test", "original_confidence": 0.9, "edited_confidence": 0.7,
                 "delta": -0.2, "confirmed": True}]
        result = analyzer.analyze_iterative(img, edited, "cat", prev)
        assert result.insights == ["bg matters"]

    def test_limits_edited_images_to_four(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value='{"edit_instructions": []}')
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        img = Image.new("RGB", (64, 64))
        edited = [Image.new("RGB", (64, 64)) for _ in range(10)]
        analyzer.analyze_iterative(img, edited, "cat", [])
        # Check the messages sent to _run
        call_args = analyzer._run.call_args[0][0]
        # Count image items in content
        image_count = sum(1 for item in call_args[0]["content"] if item.get("type") == "image")
        assert image_count == 5  # 1 original + 4 edited (capped)


class TestVerifyEditMocked:
    def test_returns_parsed_result(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "edit_applied": True,
            "confidence": 0.9,
            "description": "Color changed",
            "issues": [],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        orig = Image.new("RGB", (64, 64))
        edited = Image.new("RGB", (64, 64), "green")
        result = analyzer.verify_edit(orig, edited, "Change to green")
        assert result["edit_applied"] is True
        assert result["confidence"] == 0.9

    def test_failure_returns_default(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(side_effect=Exception("fail"))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        orig = Image.new("RGB", (64, 64))
        edited = Image.new("RGB", (64, 64))
        result = analyzer.verify_edit(orig, edited, "test")
        assert result["edit_applied"] is True
        assert result["confidence"] == 0.5
        assert "Could not verify" in result["issues"][0]


class TestClassifyFeaturesMocked:
    def test_maps_classifications_back(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "classifications": [
                {"index": 1, "feature_name": "ears", "feature_type": "intrinsic", "reasoning": "part of cat"},
                {"index": 2, "feature_name": "background", "feature_type": "contextual", "reasoning": "env"},
            ],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        features = [
            {"instruction": "Remove ears", "hypothesis": "h"},
            {"instruction": "Remove bg", "hypothesis": "h"},
        ]
        result = analyzer.classify_features("cat", features)
        assert result[0]["feature_type"] == "intrinsic"
        assert result[1]["feature_type"] == "contextual"

    def test_empty_features_returns_empty(self):
        analyzer = _make_analyzer()
        result = analyzer.classify_features("cat", [])
        assert result == []

    def test_fallback_on_failure(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(side_effect=Exception("fail"))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)
        analyzer._fallback_classify = QwenVLAnalyzer._fallback_classify.__get__(analyzer)

        features = [{"instruction": "Remove ears", "hypothesis": "h"}]
        result = analyzer.classify_features("cat", features)
        # Fallback now uses keyword-based classification
        assert result[0]["feature_type"] == "intrinsic"

    def test_out_of_range_index_ignored(self):
        analyzer = _make_analyzer()
        analyzer._run = MagicMock(return_value=json.dumps({
            "classifications": [
                {"index": 99, "feature_name": "x", "feature_type": "contextual"},
            ],
        }))
        analyzer._repair_json = QwenVLAnalyzer._repair_json.__get__(analyzer)

        features = [{"instruction": "test", "hypothesis": "h"}]
        result = analyzer.classify_features("cat", features)
        # Index 99 is out of range; _fill_unclassified applies keyword fallback
        assert result[0]["feature_type"] == "intrinsic"
