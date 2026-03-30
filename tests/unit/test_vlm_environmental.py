"""Tests for VLM environmental pattern analysis."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.vlm import (
    EnvironmentalPattern,
    _ENVIRONMENTAL_ANALYSIS_PROMPT,
    _NEGATIVE_ANALYSIS_PROMPT,
)


# =========================================================================
# EnvironmentalPattern dataclass
# =========================================================================

class TestEnvironmentalPattern:
    def test_create_pattern(self):
        p = EnvironmentalPattern(
            pattern="blue water background",
            category="background",
            frequency="7 out of 8",
            removal_edit="Replace background with plain white studio backdrop",
            hypothesis="Model may rely on blue water context",
        )
        assert p.pattern == "blue water background"
        assert p.category == "background"

    def test_default_fields_required(self):
        with pytest.raises(TypeError):
            EnvironmentalPattern()


# =========================================================================
# _ENVIRONMENTAL_ANALYSIS_PROMPT format
# =========================================================================

class TestEnvironmentalPrompt:
    def test_prompt_accepts_format_args(self):
        result = _ENVIRONMENTAL_ANALYSIS_PROMPT.format(
            class_name="tench", num_images=8,
        )
        assert "tench" in result
        assert "8" in result

    def test_prompt_never_reference_class_name(self):
        prompt = _ENVIRONMENTAL_ANALYSIS_PROMPT.lower()
        assert "never reference" in prompt

    def test_prompt_requests_frequency(self):
        prompt = _ENVIRONMENTAL_ANALYSIS_PROMPT.lower()
        assert "frequency" in prompt


# =========================================================================
# _NEGATIVE_ANALYSIS_PROMPT allows correlate_addition
# =========================================================================

class TestNegativePromptCorrelates:
    def test_prompt_allows_correlate_addition(self):
        prompt = _NEGATIVE_ANALYSIS_PROMPT.lower()
        assert "correlate_addition" in prompt

    def test_prompt_forbids_morphological(self):
        prompt = _NEGATIVE_ANALYSIS_PROMPT.lower()
        assert "body parts" in prompt and "forbidden" in prompt

    def test_prompt_allows_colors_textures(self):
        prompt = _NEGATIVE_ANALYSIS_PROMPT.lower()
        assert "colors" in prompt
        assert "textures" in prompt

    def test_prompt_requests_mix(self):
        prompt = _NEGATIVE_ANALYSIS_PROMPT.lower()
        assert "at least half" in prompt


# =========================================================================
# _parse_environmental_patterns
# =========================================================================

class TestParseEnvironmentalPatterns:
    @pytest.fixture
    def analyzer(self):
        with patch("src.vlm.QwenVLAnalyzer.__init__", return_value=None):
            from src.vlm import QwenVLAnalyzer
            obj = QwenVLAnalyzer.__new__(QwenVLAnalyzer)
            obj.model = MagicMock()
            obj.processor = MagicMock()
            obj.loaded = False
            return obj

    def test_valid_json(self, analyzer):
        raw = '''```json
{
  "environmental_patterns": [
    {
      "pattern": "murky green water",
      "category": "background",
      "frequency": "6 out of 8",
      "removal_edit": "Replace background with plain white studio backdrop",
      "hypothesis": "Model relies on water color"
    }
  ]
}
```'''
        result = analyzer._parse_environmental_patterns(raw)
        assert len(result) == 1
        assert isinstance(result[0], EnvironmentalPattern)
        assert result[0].pattern == "murky green water"
        assert result[0].category == "background"

    def test_empty_patterns(self, analyzer):
        raw = '{"environmental_patterns": []}'
        result = analyzer._parse_environmental_patterns(raw)
        assert result == []

    def test_no_json(self, analyzer):
        raw = "I could not analyze these images."
        result = analyzer._parse_environmental_patterns(raw)
        assert result == []

    def test_missing_key(self, analyzer):
        raw = '{"something_else": []}'
        result = analyzer._parse_environmental_patterns(raw)
        assert result == []

    def test_multiple_patterns(self, analyzer):
        raw = '''{
  "environmental_patterns": [
    {"pattern": "blue water", "category": "background", "frequency": "7/8",
     "removal_edit": "Replace with white", "hypothesis": "water bias"},
    {"pattern": "fishing net", "category": "co_object", "frequency": "5/8",
     "removal_edit": "Remove net", "hypothesis": "net bias"}
  ]
}'''
        result = analyzer._parse_environmental_patterns(raw)
        assert len(result) == 2
        assert result[1].pattern == "fishing net"

    def test_missing_fields_default_empty(self, analyzer):
        raw = '{"environmental_patterns": [{"pattern": "blue sky"}]}'
        result = analyzer._parse_environmental_patterns(raw)
        assert len(result) == 1
        assert result[0].category == ""
        assert result[0].removal_edit == ""
