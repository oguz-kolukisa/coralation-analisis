"""Tests for VLM JSON repair fallback paths (the except JSONDecodeError branches)."""
from __future__ import annotations

import json

import pytest

from src.vlm import QwenVLAnalyzer, FeatureDiscovery, FinalAnalysis, VLMAnalysis


def _make_analyzer():
    obj = object.__new__(QwenVLAnalyzer)
    return obj


class TestParseFeatureDiscoveryRepairFallback:
    def test_malformed_json_repaired(self):
        a = _make_analyzer()
        # Trailing comma triggers JSONDecodeError, then _repair_json fixes it
        raw = '{"gradcam_summary": "head focus", "detected_features": [{"name": "ears", "category": "part", "feature_type": "intrinsic", "location": "top", "gradcam_attention": "high", "reasoning": "test",}],}'
        result = a._parse_feature_discovery(raw, "cat")
        assert result.gradcam_summary == "head focus"
        assert len(result.features) == 1


class TestParseFeatureEditsRepairFallback:
    def test_malformed_json_repaired(self):
        a = _make_analyzer()
        raw = '{"feature_edits": [{"feature_name": "bg", "edit_instruction": "remove", "edit_type": "removal", "expected_impact": "high", "hypothesis": "test",},],}'
        result = a._parse_feature_edits(raw)
        assert len(result) == 1


class TestParseFinalAnalysisRepairFallback:
    def test_malformed_json_repaired(self):
        a = _make_analyzer()
        raw = '{"robustness_score": 8, "risk_level": "LOW", "feature_importance": [], "confirmed_shortcuts": [], "legitimate_features": [],}'
        result = a._parse_final_analysis(raw, "cat")
        assert result.robustness_score == 8


class TestParseAnalysisRepairFallback:
    def test_malformed_json_repaired(self):
        a = _make_analyzer()
        raw = '{"key_features": ["ears",], "edit_instructions": [{"edit": "test", "hypothesis": "h", "type": "t", "priority": 3,},],}'
        result = a._parse_analysis(raw, "cat", "positive")
        assert len(result.edit_instructions) == 1


class TestParseIterativeRepairFallback:
    def test_malformed_json_repaired(self):
        a = _make_analyzer()
        raw = '{"insights": ["bg matters",], "edit_instructions": [{"edit": "test", "hypothesis": "h", "type": "t", "priority": 3,},],}'
        result = a._parse_iterative_analysis(raw, "cat")
        assert len(result.edit_instructions) == 1


class TestGenerateKnowledgeRepairFallback:
    def test_malformed_json_repaired(self):
        a = _make_analyzer()
        a._run = lambda *args, **kwargs: '{"knowledge_based_features": [{"feature": "grass",},],}'
        a._repair_json = QwenVLAnalyzer._repair_json.__get__(a)

        result = a.generate_knowledge_based_features("cat")
        assert len(result["knowledge_based_features"]) == 1


class TestClassifyFeaturesJsonFallback:
    def test_fallback_json_search(self):
        """When the classifications-specific regex fails, falls back to any JSON."""
        a = _make_analyzer()
        # Return JSON that doesn't match the classifications-specific pattern
        a._run = lambda *args, **kwargs: 'Here is result: {"results": [{"index": 1, "feature_name": "bg", "feature_type": "contextual"}]}'
        a._repair_json = QwenVLAnalyzer._repair_json.__get__(a)

        features = [{"instruction": "Remove bg", "hypothesis": "h"}]
        result = a.classify_features("cat", features)
        # Won't match index pattern since key is "results" not "classifications"
        assert "feature_type" not in result[0]
