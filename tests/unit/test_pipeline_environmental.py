"""Tests for pipeline environmental pattern integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.pipeline import (
    AnalysisPipeline, ClassAnalysisResult, EditInput, ImageSet,
)
from src.vlm import EditInstruction, EnvironmentalPattern
from tests.conftest import make_image, FakePrediction


def _make_pattern(pattern="blue water", edit="Replace with white"):
    return EnvironmentalPattern(
        pattern=pattern, category="background",
        frequency="6/8", removal_edit=edit,
        hypothesis="Model relies on this",
    )


@pytest.fixture
def pipeline(config):
    with patch("src.pipeline.ModelManager"):
        return AnalysisPipeline(config)


@pytest.fixture
def images_with_inspect():
    img_set = ImageSet()
    for i in range(4):
        img = make_image(color="blue")
        pred = FakePrediction()
        img_set.annotated_inspect.append((img, pred, "tench"))
    return img_set


@pytest.fixture
def baseline_results():
    return [{"class_confidence": 0.85} for _ in range(4)]


# =========================================================================
# _pattern_to_instruction
# =========================================================================

class TestPatternToInstruction:
    def test_creates_edit_instruction(self, pipeline):
        pattern = _make_pattern(edit="Replace background with neutral gray")
        instr = pipeline._pattern_to_instruction(pattern, image_index=2)
        assert isinstance(instr, EditInstruction)
        assert instr.edit == "Replace background with neutral gray"
        assert instr.type == "environment_removal"
        assert instr.target == "positive"
        assert instr.image_index == 2

    def test_uses_hypothesis(self, pipeline):
        pattern = _make_pattern()
        instr = pipeline._pattern_to_instruction(pattern, image_index=0)
        assert instr.hypothesis == "Model relies on this"


# =========================================================================
# _convert_patterns_to_edits
# =========================================================================

class TestConvertPatternsToEdits:
    def test_empty_patterns(self, pipeline, images_with_inspect, baseline_results):
        result = pipeline._convert_patterns_to_edits([], images_with_inspect, baseline_results)
        assert result == []

    def test_creates_edits_for_each_pattern_and_image(
        self, pipeline, images_with_inspect, baseline_results,
    ):
        patterns = [_make_pattern("blue water"), _make_pattern("fishing net")]
        result = pipeline._convert_patterns_to_edits(patterns, images_with_inspect, baseline_results)
        # 2 patterns × 2 images (first 2 inspect images) = 4
        assert len(result) == 4
        assert all(isinstance(e, EditInput) for e in result)

    def test_uses_baseline_confidence(
        self, pipeline, images_with_inspect, baseline_results,
    ):
        baseline_results[0]["class_confidence"] = 0.92
        patterns = [_make_pattern()]
        result = pipeline._convert_patterns_to_edits(patterns, images_with_inspect, baseline_results)
        assert result[0].original_confidence == 0.92


# =========================================================================
# _analyze_environmental_patterns
# =========================================================================

class TestAnalyzeEnvironmentalPatterns:
    def test_too_few_images_returns_empty(self, pipeline):
        img_set = ImageSet()
        img_set.annotated_inspect = [
            (make_image(), FakePrediction(), "tench"),
            (make_image(), FakePrediction(), "tench"),
        ]
        result = pipeline._analyze_environmental_patterns("tench", img_set)
        assert result == []

    def test_calls_vlm_with_images(self, pipeline, images_with_inspect):
        mock_vlm = MagicMock()
        mock_vlm.analyze_environmental_patterns.return_value = [_make_pattern()]
        pipeline.models.vlm.return_value = mock_vlm

        result = pipeline._analyze_environmental_patterns("tench", images_with_inspect)
        assert len(result) == 1
        mock_vlm.analyze_environmental_patterns.assert_called_once()
        call_args = mock_vlm.analyze_environmental_patterns.call_args
        assert len(call_args[0][0]) == 4  # 4 images passed


# =========================================================================
# _generate_environmental_edits
# =========================================================================

class TestGenerateEnvironmentalEdits:
    def test_stores_patterns_on_result(self, pipeline, images_with_inspect, baseline_results):
        mock_vlm = MagicMock()
        mock_vlm.analyze_environmental_patterns.return_value = [_make_pattern()]
        pipeline.models.vlm.return_value = mock_vlm

        result = ClassAnalysisResult(class_name="tench")
        result.baseline_results = baseline_results
        edits = pipeline._generate_environmental_edits("tench", images_with_inspect, result)

        assert len(result.environmental_patterns) == 1
        assert result.environmental_patterns[0]["pattern"] == "blue water"
        assert len(edits) > 0

    def test_no_patterns_returns_empty(self, pipeline, images_with_inspect, baseline_results):
        mock_vlm = MagicMock()
        mock_vlm.analyze_environmental_patterns.return_value = []
        pipeline.models.vlm.return_value = mock_vlm

        result = ClassAnalysisResult(class_name="tench")
        result.baseline_results = baseline_results
        edits = pipeline._generate_environmental_edits("tench", images_with_inspect, result)

        assert edits == []
        assert result.environmental_patterns == []


# =========================================================================
# Config fields
# =========================================================================

class TestEnvironmentalConfig:
    def test_default_min_frequency(self):
        cfg = Config()
        assert cfg.min_pattern_frequency == 3

    def test_override_min_frequency(self):
        cfg = Config(min_pattern_frequency=5)
        assert cfg.min_pattern_frequency == 5
