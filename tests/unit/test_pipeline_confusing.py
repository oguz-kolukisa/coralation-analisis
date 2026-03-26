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


class TestClassifierConfusingClasses:
    def test_returns_sorted_by_count(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = MagicMock(
            top_k=[("tiger cat", 0.3), ("Persian cat", 0.2)]
        )
        pipeline.models.classifier.return_value = mock_clf

        images = [(MagicMock(), "img.jpg")] * 3
        result = pipeline._classifier_confusing_classes("tabby cat", images)
        assert result[0] == "tiger cat"

    def test_excludes_target_class(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = MagicMock(
            top_k=[("tabby cat", 0.9), ("tiger cat", 0.1)]
        )
        pipeline.models.classifier.return_value = mock_clf

        images = [(MagicMock(), "img.jpg")]
        result = pipeline._classifier_confusing_classes("tabby cat", images)
        assert "tabby cat" not in result

    def test_empty_when_no_predictions(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = MagicMock(top_k=[])
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._classifier_confusing_classes("tabby cat", [])
        assert result == []


class TestVlmConfusingClasses:
    def test_returns_validated_candidates(self, pipeline):
        mock_vlm = MagicMock()
        mock_vlm.select_confusing_classes.return_value = ["fire truck", "fire engine"]
        pipeline.models.vlm.return_value = mock_vlm

        mock_sampler = MagicMock()
        mock_sampler.get_label_names.return_value = ["fire engine", "school bus"]
        pipeline.models.sampler.return_value = mock_sampler

        mock_clf = MagicMock()
        mock_clf.is_valid_class.side_effect = lambda n: n != "fire truck"
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._vlm_confusing_classes("school bus")
        assert result == ["fire engine"]

    def test_returns_empty_on_exception(self, pipeline):
        mock_vlm = MagicMock()
        mock_vlm.select_confusing_classes.side_effect = RuntimeError("OOM")
        pipeline.models.vlm.return_value = mock_vlm

        mock_sampler = MagicMock()
        mock_sampler.get_label_names.return_value = ["fire engine"]
        pipeline.models.sampler.return_value = mock_sampler

        result = pipeline._vlm_confusing_classes("tabby cat")
        assert result == []

    def test_returns_empty_when_all_invalid(self, pipeline):
        mock_vlm = MagicMock()
        mock_vlm.select_confusing_classes.return_value = ["invalid_class"]
        pipeline.models.vlm.return_value = mock_vlm

        mock_sampler = MagicMock()
        mock_sampler.get_label_names.return_value = ["fire engine"]
        pipeline.models.sampler.return_value = mock_sampler

        mock_clf = MagicMock()
        mock_clf.is_valid_class.return_value = False
        pipeline.models.classifier.return_value = mock_clf

        result = pipeline._vlm_confusing_classes("tabby cat")
        assert result == []


class TestMergeConfusingClasses:
    def test_deduplicates_case_insensitive(self, pipeline):
        result = pipeline._merge_confusing_classes(
            ["Tiger Cat", "persian cat"],
            ["tiger cat", "Siamese cat"],
        )
        assert result == ["Tiger Cat", "persian cat", "Siamese cat"]

    def test_classifier_first_priority(self, pipeline):
        result = pipeline._merge_confusing_classes(
            ["tiger cat"],
            ["persian cat", "tiger cat"],
        )
        assert result == ["tiger cat", "persian cat"]

    def test_empty_inputs(self, pipeline):
        assert pipeline._merge_confusing_classes([], []) == []

    def test_one_empty(self, pipeline):
        result = pipeline._merge_confusing_classes([], ["tiger cat"])
        assert result == ["tiger cat"]


class TestFindConfusingClasses:
    def test_combines_both_strategies(self, pipeline):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = MagicMock(
            top_k=[("tiger cat", 0.3)]
        )
        mock_clf.is_valid_class.return_value = True
        pipeline.models.classifier.return_value = mock_clf

        mock_vlm = MagicMock()
        mock_vlm.select_confusing_classes.return_value = ["Persian cat"]
        pipeline.models.vlm.return_value = mock_vlm

        mock_sampler = MagicMock()
        mock_sampler.get_label_names.return_value = ["tiger cat", "Persian cat"]
        pipeline.models.sampler.return_value = mock_sampler

        images = [(MagicMock(), "img.jpg")]
        result = pipeline._find_confusing_classes("tabby cat", images)
        assert "tiger cat" in result
        assert "Persian cat" in result

    def test_respects_top_negative_classes_limit(self, pipeline):
        pipeline.cfg.top_negative_classes = 1

        mock_clf = MagicMock()
        mock_clf.predict.return_value = MagicMock(
            top_k=[("tiger cat", 0.3), ("Persian cat", 0.2)]
        )
        mock_clf.is_valid_class.return_value = True
        pipeline.models.classifier.return_value = mock_clf

        mock_vlm = MagicMock()
        mock_vlm.select_confusing_classes.return_value = ["Siamese cat"]
        pipeline.models.vlm.return_value = mock_vlm

        mock_sampler = MagicMock()
        mock_sampler.get_label_names.return_value = ["tiger cat", "Siamese cat"]
        pipeline.models.sampler.return_value = mock_sampler

        images = [(MagicMock(), "img.jpg")]
        result = pipeline._find_confusing_classes("tabby cat", images)
        assert len(result) == 1
