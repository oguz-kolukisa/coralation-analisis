"""Tests verifying VLM prompts contain anti-tautology constraints."""
from src.vlm import (
    _FEATURE_EDIT_PROMPT,
    _KNOWLEDGE_BASED_FEATURES_PROMPT,
    _NEGATIVE_ANALYSIS_PROMPT,
    _ANALYSIS_PROMPT,
    _ITERATIVE_REFINEMENT_PROMPT,
)


class TestPromptAntiTautology:
    def test_feature_edit_prompt(self):
        prompt = _FEATURE_EDIT_PROMPT.lower()
        assert "never reference the class name" in prompt

    def test_knowledge_prompt(self):
        prompt = _KNOWLEDGE_BASED_FEATURES_PROMPT.lower()
        assert "never reference" in prompt or "never mention" in prompt

    def test_negative_prompt(self):
        prompt = _NEGATIVE_ANALYSIS_PROMPT.lower()
        assert "never reference" in prompt or "never mention" in prompt

    def test_analysis_prompt(self):
        prompt = _ANALYSIS_PROMPT.lower()
        assert "never reference the class name" in prompt

    def test_iterative_prompt(self):
        prompt = _ITERATIVE_REFINEMENT_PROMPT.lower()
        assert "never reference" in prompt
