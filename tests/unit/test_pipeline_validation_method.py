"""Tests for validation_method field in significance computation."""
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.pipeline import AnalysisPipeline
from src.vlm import EditInstruction


@pytest.fixture
def pipeline():
    cfg = Config()
    with patch("src.pipeline.ModelManager"):
        p = AnalysisPipeline.__new__(AnalysisPipeline)
        p.cfg = cfg
        p.stat_validator = MagicMock()
        return p


class TestValidationMethod:
    def test_threshold_method_with_single_generation(self, pipeline):
        instr = EditInstruction(
            edit="Remove ears", hypothesis="h",
            type="feature_removal", target="positive",
            priority=3, image_index=0, source_class="",
        )
        result = pipeline._compute_significance(
            [-0.2], instr, False, 1,
        )
        assert result["validation_method"] == "threshold"

    def test_statistical_method_with_multiple_generations(self, pipeline):
        mock_stat = MagicMock()
        mock_stat.confirmed = True
        mock_stat.p_value = 0.01
        mock_stat.cohens_d = 1.5
        mock_stat.effect_size_interpretation = "large"
        mock_stat.statistically_significant = True
        mock_stat.practically_significant = True
        pipeline.stat_validator.validate.return_value = mock_stat
        instr = EditInstruction(
            edit="Remove ears", hypothesis="h",
            type="feature_removal", target="positive",
            priority=3, image_index=0, source_class="",
        )
        result = pipeline._compute_significance(
            [-0.2, -0.3, -0.1], instr, False, 3,
        )
        assert result["validation_method"] == "statistical"
