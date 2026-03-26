"""GPU integration tests for src/vlm.py.

Tests QwenVLAnalyzer's parsing logic (no model needed) and
full inference (requires GPU + model weights).
"""
from __future__ import annotations

import json

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

pytestmark = pytest.mark.gpu


# ============================================================================
# JSON Repair (no GPU needed, but lives in VLM module)
# ============================================================================

class TestRepairJson:
    """Test _repair_json via a lightweight analyzer stub."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer without loading model (just for _repair_json)."""
        # _repair_json is a regular method, we can test it by constructing
        # the object partially. We use __new__ to skip __init__.
        obj = object.__new__(QwenVLAnalyzer)
        return obj

    def test_trailing_comma(self, analyzer):
        bad = '{"a": 1, "b": 2,}'
        result = json.loads(analyzer._repair_json(bad))
        assert result == {"a": 1, "b": 2}

    def test_code_fence_removal(self, analyzer):
        bad = '```json\n{"a": 1}\n```'
        result = json.loads(analyzer._repair_json(bad))
        assert result == {"a": 1}

    def test_unquoted_keys(self, analyzer):
        bad = '{key: "value"}'
        result = json.loads(analyzer._repair_json(bad))
        assert result == {"key": "value"}

    def test_truncated_json_closes_brackets(self, analyzer):
        bad = '{"list": [1, 2'
        repaired = analyzer._repair_json(bad)
        result = json.loads(repaired)
        assert "list" in result

    def test_valid_json_unchanged(self, analyzer):
        good = '{"a": 1}'
        assert json.loads(analyzer._repair_json(good)) == {"a": 1}

    def test_empty_json_object(self, analyzer):
        result = json.loads(analyzer._repair_json("{}"))
        assert result == {}


# ============================================================================
# Parse methods (no GPU needed)
# ============================================================================

class TestParseFeatureDiscovery:
    @pytest.fixture
    def analyzer(self):
        obj = object.__new__(QwenVLAnalyzer)
        return obj

    def test_parses_valid_response(self, analyzer):
        raw = json.dumps({
            "detected_features": [
                {
                    "name": "ears",
                    "category": "object_part",
                    "feature_type": "intrinsic",
                    "location": "top",
                    "gradcam_attention": "high",
                    "reasoning": "pointy cat ears",
                }
            ],
            "gradcam_summary": "Focus on head",
            "intrinsic_features": ["ears"],
            "contextual_features": ["background"],
        })
        result = analyzer._parse_feature_discovery(raw, "tabby cat")
        assert isinstance(result, FeatureDiscovery)
        assert len(result.features) == 1
        assert result.features[0].name == "ears"
        assert result.gradcam_summary == "Focus on head"

    def test_handles_no_json(self, analyzer):
        result = analyzer._parse_feature_discovery("no json here", "cat")
        assert isinstance(result, FeatureDiscovery)
        assert len(result.features) == 0

    def test_handles_malformed_json(self, analyzer):
        result = analyzer._parse_feature_discovery("{ broken json", "cat")
        assert isinstance(result, FeatureDiscovery)


class TestParseAnalysis:
    @pytest.fixture
    def analyzer(self):
        obj = object.__new__(QwenVLAnalyzer)
        return obj

    def test_parses_edit_instructions(self, analyzer):
        raw = json.dumps({
            "key_features": ["ears", "fur"],
            "essential_features": ["ears"],
            "spurious_features": ["background"],
            "model_focus": "head region",
            "edit_instructions": [
                {"edit": "Remove ears", "hypothesis": "test", "type": "feature_removal", "priority": 5},
                {"edit": "Change bg", "hypothesis": "test", "type": "background_change", "priority": 2},
            ],
        })
        result = analyzer._parse_analysis(raw, "cat", "positive")
        assert isinstance(result, VLMAnalysis)
        assert len(result.edit_instructions) == 2
        # Should be sorted by priority descending
        assert result.edit_instructions[0].priority >= result.edit_instructions[1].priority

    def test_handles_empty_response(self, analyzer):
        result = analyzer._parse_analysis("", "cat", "positive")
        assert isinstance(result, VLMAnalysis)
        assert result.edit_instructions == []

    def test_sets_default_target(self, analyzer):
        raw = json.dumps({
            "edit_instructions": [
                {"edit": "test", "hypothesis": "h", "type": "t", "priority": 1},
            ],
        })
        result = analyzer._parse_analysis(raw, "cat", "negative")
        assert result.edit_instructions[0].target == "negative"


class TestParseFinalAnalysis:
    @pytest.fixture
    def analyzer(self):
        obj = object.__new__(QwenVLAnalyzer)
        return obj

    def test_parses_all_fields(self, analyzer):
        raw = json.dumps({
            "feature_importance": [{"feature": "ears", "impact": "-0.3"}],
            "confirmed_shortcuts": [{"feature": "bg", "evidence": "contextual"}],
            "legitimate_features": ["ears"],
            "robustness_score": 7,
            "risk_level": "LOW",
            "vulnerabilities": ["background bias"],
            "recommendations": ["diversify training"],
            "summary": "Model is mostly robust.",
        })
        result = analyzer._parse_final_analysis(raw, "cat")
        assert isinstance(result, FinalAnalysis)
        assert result.robustness_score == 7
        assert result.risk_level == "LOW"
        assert len(result.confirmed_shortcuts) == 1

    def test_defaults_on_missing_fields(self, analyzer):
        raw = json.dumps({})
        result = analyzer._parse_final_analysis(raw, "cat")
        assert result.robustness_score == 5
        assert result.risk_level == "MEDIUM"


class TestParseIterativeAnalysis:
    @pytest.fixture
    def analyzer(self):
        obj = object.__new__(QwenVLAnalyzer)
        return obj

    def test_parses_insights(self, analyzer):
        raw = json.dumps({
            "insights": ["background matters"],
            "confirmed_shortcuts": ["green bg"],
            "needs_more_testing": ["texture"],
            "edit_instructions": [
                {"edit": "Remove bg", "hypothesis": "h", "type": "background_change", "priority": 4},
            ],
        })
        result = analyzer._parse_iterative_analysis(raw, "cat")
        assert result.insights == ["background matters"]
        assert result.confirmed_shortcuts == ["green bg"]
        assert len(result.edit_instructions) == 1
        assert result.edit_instructions[0].target == "positive"


class TestFallbackClassify:
    def test_returns_empty_string(self):
        obj = object.__new__(QwenVLAnalyzer)
        assert obj._fallback_classify("Remove the background") == ""

    def test_returns_empty_for_any_input(self):
        obj = object.__new__(QwenVLAnalyzer)
        assert obj._fallback_classify("") == ""


# ============================================================================
# Full inference tests (require GPU + model)
# ============================================================================

class TestVLMInference:
    """These tests load the actual Qwen VLM and run inference."""

    @pytest.fixture(scope="class")
    def vlm(self, device):
        return QwenVLAnalyzer(device=device)

    def test_loaded(self, vlm):
        assert vlm.loaded is True

    def test_discover_features(self, vlm, sample_image):
        gradcam = Image.new("RGB", (224, 224), "yellow")
        result = vlm.discover_features(sample_image, gradcam, "test_class", 0.9)
        assert isinstance(result, FeatureDiscovery)

    def test_verify_edit(self, vlm, sample_image):
        edited = Image.new("RGB", (224, 224), "green")
        result = vlm.verify_edit(sample_image, edited, "Change color to green")
        assert "edit_applied" in result
        assert isinstance(result["edit_applied"], bool)

    def test_offload_and_reload(self, vlm):
        vlm.offload()
        assert vlm.loaded is False
        vlm.load_to_gpu()
        assert vlm.loaded is True
