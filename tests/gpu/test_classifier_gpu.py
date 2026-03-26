"""GPU integration tests for src/classifier.py.

Tests ResNet-50 classification, attention maps, and label resolution
using actual model weights on GPU/CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.classifier import ImageNetClassifier, ClassifierResult

pytestmark = pytest.mark.gpu


# ============================================================================
# Initialization
# ============================================================================

class TestClassifierInit:
    def test_loads_on_cpu(self):
        clf = ImageNetClassifier(device="cpu", attention_method="gradcam")
        assert clf.loaded is True
        assert clf.model is not None

    def test_labels_populated(self):
        clf = ImageNetClassifier(device="cpu", attention_method="gradcam")
        assert len(clf.labels) == 1000

    def test_invalid_attention_method_falls_back(self):
        clf = ImageNetClassifier(device="cpu", attention_method="nonexistent")
        assert clf._attention_generator is not None


# ============================================================================
# Offload / Reload
# ============================================================================

class TestClassifierOffload:
    def test_offload_sets_loaded_false(self):
        clf = ImageNetClassifier(device="cpu", attention_method="gradcam")
        clf.offload()
        assert clf.loaded is False
        assert clf.model is None

    def test_reload_after_offload(self):
        clf = ImageNetClassifier(device="cpu", attention_method="gradcam")
        clf.offload()
        clf.load_to_gpu()
        assert clf.loaded is True
        assert clf.model is not None

    def test_reload_is_idempotent(self):
        clf = ImageNetClassifier(device="cpu", attention_method="gradcam")
        clf.load_to_gpu()
        assert clf.loaded is True


# ============================================================================
# Prediction
# ============================================================================

class TestPredict:
    @pytest.fixture(scope="class")
    def classifier(self):
        return ImageNetClassifier(device="cpu", attention_method="gradcam")

    def test_returns_classifier_result(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=False)
        assert isinstance(result, ClassifierResult)

    def test_top_k_has_correct_length(self, classifier, sample_image):
        result = classifier.predict(sample_image, top_k=3, compute_gradcam=False)
        assert len(result.top_k) == 3

    def test_confidence_between_0_and_1(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=False)
        assert 0.0 <= result.confidence <= 1.0

    def test_label_name_is_string(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=False)
        assert isinstance(result.label_name, str)
        assert len(result.label_name) > 0

    def test_gradcam_image_when_requested(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=True)
        assert result.gradcam_image is not None
        assert isinstance(result.gradcam_image, Image.Image)

    def test_no_gradcam_when_disabled(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=False)
        assert result.gradcam_image is None

    def test_attention_map_shape(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=True)
        assert result.attention_map is not None
        assert isinstance(result.attention_map, np.ndarray)
        assert result.attention_map.ndim == 2

    def test_attention_map_range(self, classifier, sample_image):
        result = classifier.predict(sample_image, compute_gradcam=True)
        assert result.attention_map.min() >= 0.0
        assert result.attention_map.max() <= 1.0

    def test_target_class_name_changes_gradcam(self, classifier, sample_image):
        r1 = classifier.predict(sample_image, target_class_name="goldfish", compute_gradcam=True)
        r2 = classifier.predict(sample_image, target_class_name="tabby", compute_gradcam=True)
        # Attention maps should differ for different target classes
        assert not np.array_equal(r1.attention_map, r2.attention_map)

    def test_predict_with_small_image(self, classifier, small_image):
        result = classifier.predict(small_image, compute_gradcam=False)
        assert isinstance(result, ClassifierResult)

    def test_predict_with_large_image(self, classifier, large_image):
        result = classifier.predict(large_image, compute_gradcam=False)
        assert isinstance(result, ClassifierResult)


# ============================================================================
# Confidence API
# ============================================================================

class TestConfidence:
    @pytest.fixture(scope="class")
    def classifier(self):
        return ImageNetClassifier(device="cpu", attention_method="gradcam")

    def test_get_class_confidence_returns_float(self, classifier, sample_image):
        conf = classifier.get_class_confidence(sample_image, "goldfish")
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_predict_with_gradcam_returns_tuple(self, classifier, sample_image):
        conf, result = classifier.predict_with_gradcam(sample_image, "goldfish")
        assert isinstance(conf, float)
        assert isinstance(result, ClassifierResult)


# ============================================================================
# Label Resolution
# ============================================================================

class TestLabelResolution:
    @pytest.fixture(scope="class")
    def classifier(self):
        return ImageNetClassifier(device="cpu", attention_method="gradcam")

    def test_valid_class(self, classifier):
        assert classifier.is_valid_class("goldfish") is True

    def test_invalid_class(self, classifier):
        assert classifier.is_valid_class("xyznonexistent123") is False

    def test_substring_match(self, classifier):
        assert classifier.is_valid_class("tabby") is True

    def test_resolve_imagenet_labels_returns_list(self, classifier):
        indices = classifier.resolve_imagenet_labels("shark")
        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_resolve_imagenet_labels_no_match(self, classifier):
        indices = classifier.resolve_imagenet_labels("xyznonexistent123")
        assert indices == []

    def test_resolve_label_raises_for_unknown(self, classifier):
        with pytest.raises(ValueError, match="not found"):
            classifier._resolve_label("xyznonexistent123")
