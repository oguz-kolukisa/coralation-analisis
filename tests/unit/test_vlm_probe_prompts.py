"""Tests for QwenVLAnalyzer.generate_probe_prompts and its helpers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.vlm import (
    ProbeFeatures,
    ProbePromptSet,
    QwenVLAnalyzer,
    _clean_prompt_list,
)


def _make_analyzer(run_returns=""):
    analyzer = object.__new__(QwenVLAnalyzer)
    analyzer._run = MagicMock(return_value=run_returns)
    return analyzer


class TestCleanPromptList:
    def test_returns_empty_on_non_list(self):
        assert _clean_prompt_list(None, 3) == []
        assert _clean_prompt_list("a string", 3) == []
        assert _clean_prompt_list({"a": 1}, 3) == []

    def test_strips_whitespace(self):
        assert _clean_prompt_list(["  hi  ", "\tok\n"], 5) == ["hi", "ok"]

    def test_drops_non_strings(self):
        assert _clean_prompt_list([1, "a", None, "b"], 5) == ["a", "b"]

    def test_drops_empty_strings(self):
        assert _clean_prompt_list(["", "   ", "real"], 5) == ["real"]

    def test_truncates_to_n(self):
        items = ["a", "b", "c", "d", "e"]
        assert _clean_prompt_list(items, 2) == ["a", "b"]

    def test_empty_list_returns_empty(self):
        assert _clean_prompt_list([], 3) == []


class TestBuildProbePrompt:
    def test_prompt_includes_class_name(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("goldfish", ["water"], ["orange scales"])
        prompt = analyzer._build_probe_prompt(feats, 3)
        assert "goldfish" in prompt

    def test_prompt_includes_bias_features(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("goldfish", ["aquarium glass", "water"], [])
        prompt = analyzer._build_probe_prompt(feats, 3)
        assert "aquarium glass" in prompt
        assert "water" in prompt

    def test_prompt_includes_real_features(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("goldfish", [], ["orange body", "dorsal fin"])
        prompt = analyzer._build_probe_prompt(feats, 3)
        assert "orange body" in prompt

    def test_prompt_mentions_n_per_variant(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("cat", ["couch"], [])
        prompt = analyzer._build_probe_prompt(feats, 7)
        assert "7" in prompt

    def test_prompt_empty_bias_shows_none_marker(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("tench", [], [])
        prompt = analyzer._build_probe_prompt(feats, 3)
        assert "none" in prompt.lower()

    def test_prompt_forbids_class_in_bias_stripped(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("parrot", ["tree"], [])
        prompt = analyzer._build_probe_prompt(feats, 3)
        assert "Do NOT use the word 'parrot'" in prompt

    def test_prompt_lists_confusing_classes(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("goldfish", ["water"], [], ["koi", "carp"])
        prompt = analyzer._build_probe_prompt(feats, 2)
        assert "koi" in prompt and "carp" in prompt

    def test_prompt_describes_adversarial_variant(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("tench", ["pond"], [], ["goldfish"])
        prompt = analyzer._build_probe_prompt(feats, 2)
        assert "adversarial" in prompt
        assert "DESIGNED TO TRICK" in prompt

    def test_round_robin_mode_shows_groups(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("tench", ["water", "rocks", "plants"], [], [])
        groups = [["water", "plants"], ["rocks"]]
        prompt = analyzer._build_probe_prompt(
            feats, 2, mode="round_robin", bias_groups=groups,
        )
        assert "Group 0:" in prompt
        assert "Group 1:" in prompt
        assert "water, plants" in prompt
        assert "DIFFERENT group" in prompt

    def test_vlm_discretion_mode_lists_features_flat(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("tench", ["water", "rocks"], [], [])
        prompt = analyzer._build_probe_prompt(feats, 2, mode="vlm_discretion")
        assert "Group 0" not in prompt
        assert "water, rocks" in prompt

    def test_round_robin_with_empty_features_still_renders(self):
        analyzer = _make_analyzer()
        feats = ProbeFeatures("tench", [], [], [])
        prompt = analyzer._build_probe_prompt(
            feats, 3, mode="round_robin", bias_groups=[[], [], []],
        )
        assert "(none discovered)" in prompt


class TestParseProbePrompts:
    def test_parses_valid_json(self):
        raw = json.dumps({
            "bias_heavy": ["a", "b"],
            "bias_stripped": ["c", "d"],
            "real_feature_only": ["e", "f"],
            "adversarial": ["g", "h"],
        })
        analyzer = _make_analyzer()
        result = analyzer._parse_probe_prompts(raw, 3, "tench")
        assert result.bias_heavy == ["a", "b"]
        assert result.bias_stripped == ["c", "d"]
        assert result.real_feature_only == ["e", "f"]
        assert result.adversarial == ["g", "h"]
        assert result.class_name == "tench"

    def test_truncates_to_n(self):
        raw = json.dumps({
            "bias_heavy": ["a", "b", "c", "d"],
            "bias_stripped": ["x", "y"],
            "real_feature_only": ["r"],
        })
        analyzer = _make_analyzer()
        result = analyzer._parse_probe_prompts(raw, 2, "tench")
        assert result.bias_heavy == ["a", "b"]

    def test_handles_missing_variant_key(self):
        raw = json.dumps({"bias_heavy": ["a"]})
        analyzer = _make_analyzer()
        result = analyzer._parse_probe_prompts(raw, 3, "tench")
        assert result.bias_heavy == ["a"]
        assert result.bias_stripped == []
        assert result.real_feature_only == []

    def test_invalid_json_triggers_fallback(self):
        analyzer = _make_analyzer()
        result = analyzer._parse_probe_prompts("not json", 3, "goldfish")
        assert len(result.bias_heavy) == 3
        assert len(result.bias_stripped) == 3
        assert len(result.real_feature_only) == 3
        # fallback control prompt mentions the class
        assert any("goldfish" in p for p in result.real_feature_only)

    def test_json_wrapped_in_text_is_extracted(self):
        raw = 'Sure thing!\n{"bias_heavy": ["x"], "bias_stripped": [], "real_feature_only": []}\nGoodbye'
        analyzer = _make_analyzer()
        result = analyzer._parse_probe_prompts(raw, 3, "tench")
        assert result.bias_heavy == ["x"]


class TestFallbackPromptSet:
    def test_returns_n_prompts_per_variant(self):
        analyzer = _make_analyzer()
        result = analyzer._fallback_prompt_set("hen", 4)
        assert len(result.bias_heavy) == 4
        assert len(result.bias_stripped) == 4
        assert len(result.real_feature_only) == 4
        assert len(result.adversarial) == 4

    def test_bias_stripped_does_not_contain_class_name(self):
        analyzer = _make_analyzer()
        result = analyzer._fallback_prompt_set("hen", 3)
        for p in result.bias_stripped:
            assert "hen" not in p.lower()

    def test_other_variants_contain_class_name(self):
        analyzer = _make_analyzer()
        result = analyzer._fallback_prompt_set("ostrich", 2)
        for p in result.bias_heavy:
            assert "ostrich" in p
        for p in result.real_feature_only:
            assert "ostrich" in p


class TestGenerateProbePrompts:
    def test_happy_path_returns_promptset(self):
        raw = json.dumps({
            "bias_heavy": ["fish in water tank"],
            "bias_stripped": ["clear water and plants"],
            "real_feature_only": ["orange fish on white bg"],
        })
        analyzer = _make_analyzer(run_returns=raw)
        feats = ProbeFeatures("goldfish", ["water"], ["orange body"])
        result = analyzer.generate_probe_prompts(feats, 1)
        assert isinstance(result, ProbePromptSet)
        assert result.bias_heavy == ["fish in water tank"]

    def test_vlm_exception_triggers_fallback(self):
        analyzer = _make_analyzer()
        analyzer._run.side_effect = RuntimeError("cuda oom")
        feats = ProbeFeatures("tench", ["pond"], [])
        result = analyzer.generate_probe_prompts(feats, 3)
        assert len(result.bias_heavy) == 3
        assert any("tench" in p for p in result.real_feature_only)

    def test_invalid_vlm_output_triggers_fallback(self):
        analyzer = _make_analyzer(run_returns="random chatter no JSON")
        feats = ProbeFeatures("stingray", ["sea"], [])
        result = analyzer.generate_probe_prompts(feats, 2)
        assert len(result.bias_heavy) == 2
