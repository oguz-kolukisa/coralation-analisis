"""Unit tests for src/hf_classifier.py — the HuggingFace classification wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from src.hf_classifier import (
    HFClassifier,
    _HF_REGISTRY,
    available_hf_classifiers,
    is_hf_classifier,
)


def _stub_clf(monkeypatch, n_classes=3):
    """Build an HFClassifier bypassing model download."""
    labels = [f"Bird_{i}" for i in range(n_classes)]
    monkeypatch.setattr(HFClassifier, "_load_model", lambda self: None)
    monkeypatch.setattr(HFClassifier, "_extract_labels", lambda self: labels)
    clf = HFClassifier(model_name="swin_cub", device="cpu")
    clf.model = MagicMock()
    # Processor output must support .to(device) like a BatchFeature.
    batch = MagicMock()
    batch.to.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    clf.processor = MagicMock(return_value=batch)
    return clf


class TestRegistry:
    def test_swin_cub_is_registered(self):
        assert "swin_cub" in available_hf_classifiers()
        assert _HF_REGISTRY["swin_cub"]["hf_id"].endswith("cub-200-bird-classifier-swin")

    def test_is_hf_classifier_true(self):
        assert is_hf_classifier("swin_cub") is True

    def test_is_hf_classifier_false(self):
        assert is_hf_classifier("resnet50") is False
        assert is_hf_classifier("clip_vitb32") is False


class TestHFClassifierInit:
    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError):
            HFClassifier(model_name="not_a_real_hf_shortcut")


class TestExtractLabels:
    def test_orders_by_index(self, monkeypatch):
        # Stub loader + labels so __init__ doesn't touch the network, then
        # exercise the real _extract_labels by swapping the model afterwards.
        monkeypatch.setattr(HFClassifier, "_load_model", lambda self: None)
        monkeypatch.setattr(HFClassifier, "_extract_labels", lambda self: [])
        clf = HFClassifier(model_name="swin_cub", device="cpu")
        # Call the real method with a stubbed model
        clf.model = MagicMock()
        clf.model.config.id2label = {0: "Alpha", 1: "Beta", 2: "Gamma"}
        monkeypatch.undo()
        assert clf._extract_labels() == ["Alpha", "Beta", "Gamma"]


class TestPredict:
    def _with_softmax_output(self, clf, probs_row):
        logits = torch.log(torch.tensor([probs_row]) + 1e-12)
        out = MagicMock()
        out.logits = logits
        clf.model.return_value = out

    def test_returns_argmax_label(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=3)
        self._with_softmax_output(clf, [0.1, 0.7, 0.2])
        result = clf.predict(Image.new("RGB", (32, 32)), top_k=3)
        assert result.label_idx == 1
        assert result.label_name == "Bird_1"

    def test_returns_none_gradcam(self, monkeypatch):
        clf = _stub_clf(monkeypatch)
        self._with_softmax_output(clf, [0.1, 0.7, 0.2])
        result = clf.predict(Image.new("RGB", (32, 32)), top_k=1)
        assert result.gradcam_image is None
        assert result.attention_map is None

    def test_topk_limited_to_available(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=2)
        self._with_softmax_output(clf, [0.3, 0.7])
        result = clf.predict(Image.new("RGB", (32, 32)), top_k=5)
        assert len(result.top_k) == 2


class TestGetClassConfidence:
    def test_returns_probability_for_class(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=3)
        out = MagicMock()
        out.logits = torch.log(torch.tensor([[0.5, 0.1, 0.4]]) + 1e-12)
        clf.model.return_value = out
        conf = clf.get_class_confidence(Image.new("RGB", (32, 32)), "Bird_0")
        assert conf == pytest.approx(0.5, abs=1e-3)

    def test_unknown_class_raises(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=3)
        out = MagicMock()
        out.logits = torch.log(torch.tensor([[0.5, 0.1, 0.4]]) + 1e-12)
        clf.model.return_value = out
        with pytest.raises(ValueError):
            clf.get_class_confidence(Image.new("RGB", (32, 32)), "Unknown_xyz")


class TestOffload:
    def test_drops_model_and_processor(self, monkeypatch):
        clf = _stub_clf(monkeypatch)
        clf.offload()
        assert clf.model is None
        assert clf.processor is None
        assert clf.loaded is False


class TestLabelResolution:
    def test_is_valid_class_true_false(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=3)
        assert clf.is_valid_class("Bird_0") is True
        assert clf.is_valid_class("Flamingo_unknown") is False

    def test_resolve_imagenet_labels_matches_substring(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=3)
        assert clf.resolve_imagenet_labels("Bird") == [0, 1, 2]

    def test__resolve_label_returns_index(self, monkeypatch):
        clf = _stub_clf(monkeypatch, n_classes=3)
        assert clf._resolve_label("Bird_2") == 2


class TestFactoryDispatch:
    def test_build_classifier_routes_swin_cub_to_hf(self, monkeypatch):
        import src.classifier as classifier_mod
        from src.hf_classifier import HFClassifier as RealHF
        called = {}

        class Stub:
            def __init__(self, **kwargs):
                called.update(kwargs)
        monkeypatch.setattr("src.hf_classifier.HFClassifier", Stub)
        result = classifier_mod.build_classifier("swin_cub", device="cpu")
        assert isinstance(result, Stub)
        assert called["model_name"] == "swin_cub"
        assert called["device"] == "cpu"
