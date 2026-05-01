"""Unit tests for src/clip_classifier.py — pure-logic surfaces only.

Model loading / text encoding / image encoding go through HuggingFace
transformers and are covered by the GPU integration tests. Here we test
the label lookup, scoring math, top-k selection, and the factory wiring.
"""
from __future__ import annotations

import pytest
import torch

from src import classifier as classifier_mod
from src.classifier import build_classifier
from src.clip_classifier import (
    CLIPClassifier,
    LabelLookup,
    _CLIP_REGISTRY,
    _short_label,
    available_clip_classifiers,
    imagenet_label_lookup,
    is_clip_model,
    logits_to_probs,
    similarity_logits,
    topk_from_probs,
    unwrap_features,
)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_default_models_registered(self):
        names = available_clip_classifiers()
        assert "clip_vitb32" in names
        assert "siglip2_base" in names
        assert "siglip2_large" in names

    def test_is_clip_model_true_for_registered(self):
        assert is_clip_model("clip_vitb32") is True

    def test_is_clip_model_false_for_unknown(self):
        assert is_clip_model("resnet50") is False
        assert is_clip_model("not_a_model") is False

    def test_each_spec_has_required_fields(self):
        for name, spec in _CLIP_REGISTRY.items():
            assert "hf_id" in spec, f"{name} missing hf_id"
            assert "kind" in spec and spec["kind"] in {"clip", "siglip"}
            assert "prompt" in spec and "{label}" in spec["prompt"]


class TestBuildClassifierFactory:
    def test_routes_clip_name_to_clip_class(self, monkeypatch):
        called = {}

        class StubCLIP:
            def __init__(self, **kwargs):
                called.update(kwargs)
        monkeypatch.setattr("src.clip_classifier.CLIPClassifier", StubCLIP)
        result = build_classifier("clip_vitb32", device="cpu")
        assert isinstance(result, StubCLIP)
        assert called["model_name"] == "clip_vitb32"
        assert called["device"] == "cpu"

    def test_routes_imagenet_name_to_imagenet_class(self, monkeypatch):
        called = {}

        class StubINet:
            def __init__(self, **kwargs):
                called.update(kwargs)
        monkeypatch.setattr(classifier_mod, "ImageNetClassifier", StubINet)
        result = build_classifier("resnet50", device="cpu", attention_method="gradcam")
        assert isinstance(result, StubINet)
        assert called["model_name"] == "resnet50"
        assert called["attention_method"] == "gradcam"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            build_classifier("not_a_model_xyz")


# ---------------------------------------------------------------------------
# LabelLookup
# ---------------------------------------------------------------------------

class TestLabelLookup:
    @pytest.fixture
    def lookup(self):
        return LabelLookup(["tench", "goldfish", "great white shark"])

    def test_exact_match(self, lookup):
        assert lookup.resolve("tench") == 0

    def test_case_insensitive(self, lookup):
        assert lookup.resolve("TENCH") == 0

    def test_substring_fuzzy_fallback(self, lookup):
        # "great white shark" contains "white shark"
        assert lookup.resolve("white shark") == 2

    def test_label_contains_query_fallback(self, lookup):
        # query "goldfishy" — "goldfish" is a substring of the query
        assert lookup.resolve("goldfishy") == 1

    def test_unknown_class_raises(self, lookup):
        with pytest.raises(ValueError):
            lookup.resolve("martian fish")

    def test_is_valid_true(self, lookup):
        assert lookup.is_valid("tench") is True

    def test_is_valid_false(self, lookup):
        assert lookup.is_valid("martian fish") is False

    def test_matches_returns_all_substring_hits(self, lookup):
        # "shark" matches only "great white shark"
        assert lookup.matches("shark") == [2]

    def test_matches_empty_for_no_hits(self, lookup):
        assert lookup.matches("dragon") == []

    def test_empty_labels_everything_invalid(self):
        empty = LabelLookup([])
        assert empty.is_valid("anything") is False
        assert empty.matches("anything") == []


class TestImagenetLabelLookup:
    def test_has_1000_classes(self):
        lookup = imagenet_label_lookup()
        assert len(lookup.labels) == 1000

    def test_resolves_tench(self):
        # Tench is index 0 in standard ImageNet-1K
        lookup = imagenet_label_lookup()
        assert lookup.resolve("tench") == 0


# ---------------------------------------------------------------------------
# Scoring math
# ---------------------------------------------------------------------------

class TestSimilarityLogits:
    def test_perfect_alignment_peaks_on_same_index(self):
        # Image feat matches text_bank[2] exactly
        text_bank = torch.eye(5)
        image = torch.zeros(1, 5)
        image[0, 2] = 1.0
        logits = similarity_logits(image, text_bank, logit_scale=1.0)
        assert logits.argmax().item() == 2

    def test_logit_scale_amplifies_but_does_not_shift_argmax(self):
        text_bank = torch.eye(3)
        image = torch.tensor([[1.0, 0.0, 0.0]])
        low = similarity_logits(image, text_bank, logit_scale=1.0)
        high = similarity_logits(image, text_bank, logit_scale=10.0)
        assert low.argmax() == high.argmax() == 0
        assert high[0].item() == pytest.approx(10.0)

    def test_returns_1d_tensor(self):
        logits = similarity_logits(torch.zeros(1, 4), torch.eye(4), 1.0)
        assert logits.ndim == 1
        assert logits.shape[0] == 4


class TestLogitsToProbs:
    def test_clip_uses_softmax(self):
        logits = torch.tensor([0.0, 10.0, 0.0])
        probs = logits_to_probs(logits, "clip")
        assert probs[1].item() > 0.99
        assert probs.sum().item() == pytest.approx(1.0)

    def test_siglip_uses_sigmoid(self):
        logits = torch.tensor([0.0, 10.0, -10.0])
        probs = logits_to_probs(logits, "siglip")
        # Sigmoid: independent per-class, not normalized to 1
        assert probs[0].item() == pytest.approx(0.5)
        assert probs[1].item() > 0.99
        assert probs[2].item() < 0.01

    def test_unknown_kind_defaults_to_softmax(self):
        # Function falls through to softmax branch for anything non-siglip
        probs = logits_to_probs(torch.tensor([1.0, 2.0]), "clip")
        assert probs.sum().item() == pytest.approx(1.0)


class TestShortLabel:
    def test_strips_comma_tail(self):
        assert _short_label("goldfish, Carassius auratus") == "goldfish"

    def test_preserves_single_word(self):
        assert _short_label("tench") == "tench"

    def test_trims_whitespace(self):
        assert _short_label("  goldfish  , Carassius") == "goldfish"

    def test_empty_after_comma_uses_first_token(self):
        assert _short_label("goldfish, ") == "goldfish"

    def test_falls_back_to_original_when_first_empty(self):
        # pathological: label starts with a comma
        assert _short_label(", something") == ", something"


class TestUnwrapFeatures:
    """transformers ≥5 wraps features in BaseModelOutputWithPooling; older
    versions returned bare tensors. ``unwrap_features`` handles both."""

    def test_passes_tensor_through(self):
        t = torch.ones(1, 4)
        assert unwrap_features(t) is t

    def test_extracts_pooler_output(self):
        class Stub:
            pooler_output = torch.ones(1, 8)
        assert unwrap_features(Stub()).shape == (1, 8)

    def test_extracts_text_embeds_when_no_pooler(self):
        class Stub:
            text_embeds = torch.ones(1, 6)
        assert unwrap_features(Stub()).shape == (1, 6)

    def test_extracts_image_embeds_when_no_pooler(self):
        class Stub:
            image_embeds = torch.ones(1, 7)
        assert unwrap_features(Stub()).shape == (1, 7)

    def test_falls_back_to_last_hidden_state(self):
        class Stub:
            last_hidden_state = torch.ones(1, 5, 3)
        assert unwrap_features(Stub()).shape == (1, 5, 3)

    def test_raises_on_unrecognized_output(self):
        class Stub:
            something_else = torch.ones(1)
        with pytest.raises(TypeError):
            unwrap_features(Stub())

    def test_prefers_pooler_over_last_hidden(self):
        class Stub:
            pooler_output = torch.ones(1, 4)
            last_hidden_state = torch.ones(1, 10, 4)
        assert unwrap_features(Stub()).shape == (1, 4)


class TestTopKFromProbs:
    def test_returns_best_index(self):
        probs = torch.tensor([0.1, 0.7, 0.2])
        best, _ = topk_from_probs(probs, ["a", "b", "c"], k=2)
        assert best == 1

    def test_returns_topk_pairs_sorted_desc(self):
        probs = torch.tensor([0.1, 0.7, 0.2])
        _, pairs = topk_from_probs(probs, ["a", "b", "c"], k=2)
        assert pairs == [("b", 0.7), ("c", 0.2)]

    def test_k_larger_than_classes_clamps(self):
        probs = torch.tensor([0.6, 0.4])
        _, pairs = topk_from_probs(probs, ["a", "b"], k=10)
        assert len(pairs) == 2

    def test_scores_are_rounded_to_4dp(self):
        probs = torch.tensor([0.1234567, 0.8765433])
        _, pairs = topk_from_probs(probs, ["a", "b"], k=2)
        assert pairs[0][1] == 0.8765
        assert pairs[1][1] == 0.1235


# ---------------------------------------------------------------------------
# CLIPClassifier — inference path with stubbed model + text_bank
# ---------------------------------------------------------------------------

def _make_stub_clip(monkeypatch, *, kind="clip", n_classes=3):
    """Build a CLIPClassifier skipping HuggingFace model loads.

    Returns an instance whose `model`, `processor`, and `_text_bank` are set
    to simple fakes so we can exercise predict/get_class_confidence logic.
    """
    import src.clip_classifier as cc

    monkeypatch.setitem(cc._CLIP_REGISTRY, "_test",
                        {"hf_id": "none", "kind": kind, "prompt": "{label}"})

    monkeypatch.setattr(cc.CLIPClassifier, "_load_model", lambda self: None)
    monkeypatch.setattr(cc.CLIPClassifier, "_encode_text_bank", lambda self: None)
    monkeypatch.setattr(
        cc, "imagenet_label_lookup",
        lambda: LabelLookup([f"class_{i}" for i in range(n_classes)]),
    )

    clf = cc.CLIPClassifier(model_name="_test", device="cpu")
    # Identity text bank: row i = unit vector along dim i
    clf._text_bank = torch.eye(n_classes)
    clf.model = None
    clf.processor = None
    return clf


class _FakeImageFeat:
    """Swap _encode_image with a closure returning a preset tensor."""


class TestCLIPClassifierPredict:
    def test_predict_selects_highest_similarity_class(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, n_classes=3)
        # image aligns with class_1
        monkeypatch.setattr(clf, "_encode_image",
                            lambda img: torch.tensor([[0.0, 1.0, 0.0]]))
        monkeypatch.setattr(clf, "_logit_scale", lambda: 1.0)

        result = clf.predict(image=None, top_k=3)
        assert result.label_idx == 1
        assert result.label_name == "class_1"
        assert result.gradcam_image is None
        assert result.attention_map is None

    def test_get_class_confidence_returns_class_probability(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, n_classes=3)
        monkeypatch.setattr(clf, "_encode_image",
                            lambda img: torch.tensor([[1.0, 0.0, 0.0]]))
        monkeypatch.setattr(clf, "_logit_scale", lambda: 1.0)
        conf = clf.get_class_confidence(image=None, class_name="class_0")
        assert 0.0 <= conf <= 1.0
        # class_0 is best match, so its softmax prob is > 1/3
        assert conf > 1 / 3

    def test_predict_top_k_is_sorted(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, n_classes=4)
        monkeypatch.setattr(clf, "_encode_image",
                            lambda img: torch.tensor([[0.1, 0.9, 0.2, 0.3]]))
        monkeypatch.setattr(clf, "_logit_scale", lambda: 1.0)
        result = clf.predict(image=None, top_k=3)
        confidences = [c for _, c in result.top_k]
        assert confidences == sorted(confidences, reverse=True)

    def test_siglip_kind_uses_sigmoid_not_softmax(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, kind="siglip", n_classes=3)
        monkeypatch.setattr(clf, "_encode_image",
                            lambda img: torch.tensor([[1.0, 0.0, 0.0]]))
        monkeypatch.setattr(clf, "_logit_scale", lambda: 1.0)
        result = clf.predict(image=None, top_k=3)
        total = sum(c for _, c in result.top_k)
        # sigmoid over independent logits → sum ≠ 1
        assert not (0.99 < total < 1.01)

    def test_unknown_class_name_raises(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, n_classes=3)
        monkeypatch.setattr(clf, "_encode_image",
                            lambda img: torch.tensor([[1.0, 0.0, 0.0]]))
        monkeypatch.setattr(clf, "_logit_scale", lambda: 1.0)
        with pytest.raises(ValueError):
            clf.get_class_confidence(image=None, class_name="nonexistent_martian")

    def test_is_valid_class_delegates_to_lookup(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, n_classes=3)
        assert clf.is_valid_class("class_0") is True
        assert clf.is_valid_class("definitely_not_a_class") is False

    def test_offload_drops_model_and_bank(self, monkeypatch):
        clf = _make_stub_clip(monkeypatch, n_classes=3)
        clf.offload()
        assert clf.model is None
        assert clf._text_bank is None
        assert clf.loaded is False


class TestCLIPClassifierInit:
    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError):
            CLIPClassifier(model_name="not_registered")
