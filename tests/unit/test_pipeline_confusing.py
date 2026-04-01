"""Tests for confusing class selection in the pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.pipeline import AnalysisPipeline


@pytest.fixture
def pipeline(config):
    with patch("src.pipeline.ModelManager"):
        return AnalysisPipeline(config)


class TestValidateClassNames:
    def test_keeps_valid_classes(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.is_valid_class.return_value = True
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._validate_class_names(["tabby cat", "tiger cat"])
        assert result == ["tabby cat", "tiger cat"]

    def test_drops_invalid_classes(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.is_valid_class.side_effect = lambda n: n != "fire truck"
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._validate_class_names(["fire engine", "fire truck"])
        assert result == ["fire engine"]

    def test_empty_input(self, pipeline):
        mock_clf = MagicMock()
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._validate_class_names([])
        assert result == []

    def test_all_invalid_returns_empty(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.is_valid_class.return_value = False
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._validate_class_names(["lighthouse", "fire truck"])
        assert result == []



