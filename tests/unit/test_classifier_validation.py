"""Tests for ImageNetClassifier.is_valid_class method."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestIsValidClass:
    @pytest.fixture
    def classifier(self):
        with patch("src.classifier.models.resnet50") as mock_resnet, \
             patch("src.classifier.ResNet50_Weights") as mock_weights, \
             patch("src.classifier.get_attention_generator"):
            mock_weights.IMAGENET1K_V1.meta = {
                "categories": ["tabby cat", "tiger cat", "fire engine", "school bus"]
            }
            mock_model = MagicMock()
            mock_resnet.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            from src.classifier import ImageNetClassifier
            clf = ImageNetClassifier(device="cpu")
            return clf

    def test_exact_match_valid(self, classifier):
        assert classifier.is_valid_class("tabby cat") is True

    def test_case_insensitive_valid(self, classifier):
        assert classifier.is_valid_class("Tabby Cat") is True

    def test_substring_match_valid(self, classifier):
        assert classifier.is_valid_class("fire") is True

    def test_no_match_invalid(self, classifier):
        assert classifier.is_valid_class("fire truck") is False

    def test_completely_unknown_invalid(self, classifier):
        assert classifier.is_valid_class("lighthouse") is False

    def test_empty_string_invalid(self, classifier):
        assert classifier.is_valid_class("") is True  # empty matches all via substring
