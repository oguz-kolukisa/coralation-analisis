"""Tests for src/probe_evaluator.py."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.probe_evaluator import (
    ClassMetrics,
    ModelMetrics,
    VariantStats,
    _accumulate_totals,
    _aggregate_model_metrics,
    _any_variant_missing,
    _class_has_no_bias,
    _classify_one,
    _classify_per_variant,
    _collect_images,
    _existing_images_for_class,
    _evaluate_one_model,
    _label_matches,
    _mean,
    _metrics_to_dict,
    _safe_ratio,
    _score_one_class,
    _stats_for_variant,
    _VARIANTS,
    evaluate_probes,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classifier_returning(label="tench", confidence=0.8):
    """Build a mock classifier that always returns the same prediction."""
    clf = MagicMock()
    pred = SimpleNamespace(label_name=label, confidence=confidence,
                            top_k=[(label, confidence)],
                            gradcam_image=None, attention_map=None,
                            label_idx=0)
    clf.predict.return_value = pred
    clf.get_class_confidence.return_value = confidence
    return clf


def _make_image_record(path: Path, variant="bias_heavy", target="tench"):
    Image.new("RGB", (32, 32), "red").save(path)
    return {"path": path, "variant": variant, "target_class": target}


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

class TestSafeRatio:
    def test_normal_ratio(self):
        assert _safe_ratio(2, 4) == 0.5

    def test_zero_denominator(self):
        assert _safe_ratio(5, 0) == 0.0


class TestMean:
    def test_normal(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert _mean([]) == 0.0

    def test_negative_values(self):
        assert _mean([-1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Variant classification
# ---------------------------------------------------------------------------

class TestLabelMatches:
    def test_exact(self):
        assert _label_matches("goldfish", "goldfish") is True

    def test_case_insensitive(self):
        assert _label_matches("Goldfish", "GOLDFISH") is True

    def test_short_label_matches_long_target(self):
        # classifier returned 'goldfish', target is the full ImageNet label
        assert _label_matches("goldfish", "goldfish, Carassius auratus") is True

    def test_long_label_matches_short_target(self):
        assert _label_matches("goldfish, Carassius auratus", "goldfish") is True

    def test_unrelated_labels(self):
        assert _label_matches("tench", "goldfish") is False

    def test_whitespace_tolerated(self):
        assert _label_matches("  goldfish  ", "goldfish") is True


class TestClassifyOne:
    def test_returns_hit_and_conf_on_success(self, tmp_path):
        clf = _classifier_returning(label="tench", confidence=0.82)
        rec = _make_image_record(tmp_path / "a.jpg")
        hit, conf = _classify_one(rec, "tench", clf)
        assert hit is True
        assert conf == pytest.approx(0.82)

    def test_returns_none_conf_on_classifier_error(self, tmp_path):
        clf = MagicMock()
        clf.predict.side_effect = ValueError("label not in set")
        rec = _make_image_record(tmp_path / "a.jpg")
        hit, conf = _classify_one(rec, "tench", clf)
        assert hit is False
        assert conf is None

    def test_returns_none_on_missing_image_file(self, tmp_path):
        clf = _classifier_returning()
        rec = {"path": tmp_path / "nonexistent.jpg", "variant": "bias_heavy"}
        _hit, conf = _classify_one(rec, "tench", clf)
        assert conf is None


class TestStatsForVariant:
    def test_returns_none_on_empty(self):
        clf = _classifier_returning()
        assert _stats_for_variant([], "tench", clf) is None

    def test_returns_none_when_all_records_fail(self, tmp_path):
        clf = MagicMock()
        clf.predict.side_effect = ValueError("label missing")
        recs = [_make_image_record(tmp_path / f"{i}.jpg") for i in range(3)]
        assert _stats_for_variant(recs, "tench", clf) is None

    def test_skips_failing_records_and_averages_the_rest(self, tmp_path):
        clf = MagicMock()
        # First call fails, rest succeed with conf=0.5, all hits
        pred_ok = SimpleNamespace(label_name="tench", confidence=0.5,
                                   top_k=[("tench", 0.5)],
                                   gradcam_image=None, attention_map=None,
                                   label_idx=0)
        clf.predict.side_effect = [ValueError("boom"), pred_ok, pred_ok]
        clf.get_class_confidence.side_effect = [0.5, 0.5]
        recs = [_make_image_record(tmp_path / f"{i}.jpg") for i in range(3)]
        stats = _stats_for_variant(recs, "tench", clf)
        assert stats.n == 2
        assert stats.top1_accuracy == 1.0
        assert stats.mean_conf == pytest.approx(0.5)

    def test_computes_top1_accuracy(self, tmp_path):
        clf = _classifier_returning(label="tench")
        recs = [_make_image_record(tmp_path / f"{i}.jpg") for i in range(4)]
        stats = _stats_for_variant(recs, "tench", clf)
        assert stats.top1_accuracy == 1.0
        assert stats.n == 4

    def test_top1_accuracy_zero_when_label_mismatches(self, tmp_path):
        clf = _classifier_returning(label="goldfish")
        recs = [_make_image_record(tmp_path / f"{i}.jpg") for i in range(2)]
        stats = _stats_for_variant(recs, "tench", clf)
        assert stats.top1_accuracy == 0.0

    def test_mean_conf_uses_get_class_confidence(self, tmp_path):
        clf = _classifier_returning(confidence=0.42)
        recs = [_make_image_record(tmp_path / f"{i}.jpg") for i in range(3)]
        stats = _stats_for_variant(recs, "tench", clf)
        assert stats.mean_conf == pytest.approx(0.42)

    def test_label_match_is_case_insensitive(self, tmp_path):
        clf = _classifier_returning(label="TENCH")
        rec = _make_image_record(tmp_path / "0.jpg")
        stats = _stats_for_variant([rec], "tench", clf)
        assert stats.top1_accuracy == 1.0

    def test_short_predicted_label_matches_long_target(self, tmp_path):
        # Classifier returns 'goldfish'; caller passes 'goldfish, Carassius auratus'.
        clf = _classifier_returning(label="goldfish")
        rec = _make_image_record(tmp_path / "0.jpg")
        stats = _stats_for_variant([rec], "goldfish, Carassius auratus", clf)
        assert stats.top1_accuracy == 1.0


class TestClassifyPerVariant:
    def test_groups_by_variant(self, tmp_path):
        clf = _classifier_returning()
        recs = [
            _make_image_record(tmp_path / "a.jpg", variant="bias_heavy"),
            _make_image_record(tmp_path / "b.jpg", variant="bias_stripped"),
            _make_image_record(tmp_path / "c.jpg", variant="real_feature_only"),
            _make_image_record(tmp_path / "d.jpg", variant="adversarial"),
        ]
        per = _classify_per_variant(recs, "tench", clf)
        assert all(per[v] is not None and per[v].n == 1 for v in _VARIANTS)

    def test_unknown_variant_dropped(self, tmp_path):
        clf = _classifier_returning()
        recs = [_make_image_record(tmp_path / "a.jpg", variant="weird")]
        per = _classify_per_variant(recs, "tench", clf)
        assert all(per[v] is None for v in _VARIANTS)


# ---------------------------------------------------------------------------
# Per-class scoring
# ---------------------------------------------------------------------------

class TestScoreOneClass:
    def test_skips_when_no_bias_features(self, tmp_path):
        clf = _classifier_returning()
        cm = _score_one_class("tench", [], {"bias_features": []}, clf)
        assert cm.skipped is True
        assert cm.skip_reason == "no_bias_features"

    def test_skips_when_missing_variant(self, tmp_path):
        clf = _classifier_returning()
        # only bias_heavy present
        recs = [_make_image_record(tmp_path / "a.jpg", variant="bias_heavy")]
        cm = _score_one_class("tench", recs, {"bias_features": ["water"]}, clf)
        assert cm.skipped is True
        assert cm.skip_reason == "missing_variant"
        assert cm.bias_heavy is not None  # partial data preserved

    def test_computes_bias_lift(self, tmp_path):
        # bias_heavy → conf 0.9; real_feature_only → conf 0.1
        clf_lookup = {"bias_heavy": 0.9, "bias_stripped": 0.5,
                       "real_feature_only": 0.1, "adversarial": 0.2}
        clf = MagicMock()
        clf.predict.side_effect = lambda image, target_class_name, top_k, compute_gradcam: \
            SimpleNamespace(label_name="tench", confidence=0.8,
                             top_k=[("tench", 0.8)], gradcam_image=None,
                             attention_map=None, label_idx=0)
        # confidence depends on which image — track via side effect
        call_idx = {"i": 0}
        recs = []
        order = []
        for v in _VARIANTS:
            for i in range(2):
                recs.append(_make_image_record(tmp_path / f"{v}_{i}.jpg", variant=v))
                order.append(v)

        def conf_side_effect(img, name):
            v = order[call_idx["i"]]
            call_idx["i"] += 1
            return clf_lookup[v]
        clf.get_class_confidence.side_effect = conf_side_effect
        cm = _score_one_class("tench", recs, {"bias_features": ["water"]}, clf)
        assert not cm.skipped
        assert cm.bias_lift == pytest.approx(0.8, abs=1e-6)

    def test_full_path_returns_all_variants(self, tmp_path):
        clf = _classifier_returning()
        recs = [_make_image_record(tmp_path / f"{v}.jpg", variant=v) for v in _VARIANTS]
        cm = _score_one_class("tench", recs, {"bias_features": ["water"]}, clf)
        assert not cm.skipped
        for v in _VARIANTS:
            assert getattr(cm, v) is not None
        # adversarial is included
        assert cm.adversarial is not None


class TestAnyVariantMissing:
    def test_all_present(self):
        per = {v: VariantStats(n=1) for v in _VARIANTS}
        assert _any_variant_missing(per) is False

    def test_one_missing(self):
        per = {v: VariantStats(n=1) for v in _VARIANTS}
        per["bias_stripped"] = None
        assert _any_variant_missing(per) is True


class TestClassHasNoBias:
    def test_empty_bias(self):
        assert _class_has_no_bias({"bias_features": []}) is True

    def test_missing_key(self):
        assert _class_has_no_bias({}) is True

    def test_has_bias(self):
        assert _class_has_no_bias({"bias_features": ["x"]}) is False


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

class TestExistingImagesForClass:
    def test_drops_missing_files(self, tmp_path):
        entry = {"images": [
            {"image_path": "exists.jpg", "variant": "bias_heavy"},
            {"image_path": "missing.jpg", "variant": "bias_heavy"},
        ]}
        Image.new("RGB", (8, 8)).save(tmp_path / "exists.jpg")
        recs = _existing_images_for_class("tench", entry, tmp_path)
        assert len(recs) == 1
        assert recs[0]["target_class"] == "tench"

    def test_empty_when_no_images(self, tmp_path):
        recs = _existing_images_for_class("tench", {"images": []}, tmp_path)
        assert recs == []


class TestCollectImages:
    def test_groups_by_class(self, tmp_path):
        Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
        manifest = {"classes": {
            "tench": {"images": [{"image_path": "a.jpg", "variant": "bias_heavy"}]},
            "hen": {"images": []},
        }}
        out = _collect_images(manifest, tmp_path)
        assert set(out.keys()) == {"tench", "hen"}
        assert len(out["tench"]) == 1
        assert out["hen"] == []


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _make_class_metrics(class_name="tench", n=2,
                        bh_acc=1.0, bh_conf=0.9,
                        bs_acc=0.5, bs_conf=0.6,
                        rf_acc=0.0, rf_conf=0.1,
                        adv_acc=0.25, adv_conf=0.3,
                        skipped=False):
    if skipped:
        return ClassMetrics(class_name=class_name, skipped=True,
                             skip_reason="no_bias_features")
    bh = VariantStats(mean_conf=bh_conf, top1_accuracy=bh_acc, n=n)
    bs = VariantStats(mean_conf=bs_conf, top1_accuracy=bs_acc, n=n)
    rf = VariantStats(mean_conf=rf_conf, top1_accuracy=rf_acc, n=n)
    adv = VariantStats(mean_conf=adv_conf, top1_accuracy=adv_acc, n=n)
    return ClassMetrics(class_name=class_name, bias_heavy=bh,
                         bias_stripped=bs, real_feature_only=rf,
                         adversarial=adv,
                         bias_lift=bh_conf - rf_conf)


class TestAccumulateTotals:
    def test_sums_across_classes(self):
        per_class = {
            "a": _make_class_metrics(n=2, bh_acc=1.0, bh_conf=0.9),
            "b": _make_class_metrics(n=4, bh_acc=0.5, bh_conf=0.7),
        }
        totals = _accumulate_totals(per_class)
        # Combined bias_heavy: n=6, hits=(2*1.0 + 4*0.5)=4, conf=(2*0.9 + 4*0.7)=4.6
        assert totals["bias_heavy"]["n"] == 6
        assert totals["bias_heavy"]["hits"] == 4
        assert totals["bias_heavy"]["conf"] == pytest.approx(4.6)

    def test_skips_skipped_classes(self):
        per_class = {
            "a": _make_class_metrics(n=2, bh_conf=0.5),
            "b": _make_class_metrics(skipped=True),
        }
        totals = _accumulate_totals(per_class)
        assert totals["bias_heavy"]["n"] == 2


class TestAggregateModelMetrics:
    def test_top1_accuracy_overall_combines_all_variants(self):
        per_class = {"a": _make_class_metrics(
            n=2, bh_acc=1.0, bs_acc=0.0, rf_acc=0.5, adv_acc=0.0)}
        m = _aggregate_model_metrics("clf", per_class)
        # Overall: 4 variants × 2 images = 8; hits = 2 + 0 + 1 + 0 = 3 → 0.375
        assert m.top1_accuracy_overall == pytest.approx(0.375)

    def test_adversarial_top1_tracked_separately(self):
        per_class = {"a": _make_class_metrics(adv_acc=0.5, n=4)}
        m = _aggregate_model_metrics("clf", per_class)
        assert m.top1_accuracy_adversarial == pytest.approx(0.5)

    def test_bias_lift_score_means_over_scored_classes(self):
        per_class = {
            "a": _make_class_metrics(bh_conf=0.9, rf_conf=0.1),  # lift 0.8
            "b": _make_class_metrics(bh_conf=0.5, rf_conf=0.5),  # lift 0.0
        }
        m = _aggregate_model_metrics("clf", per_class)
        assert m.bias_lift_score == pytest.approx(0.4)

    def test_bias_lift_score_excludes_skipped(self):
        per_class = {
            "a": _make_class_metrics(bh_conf=1.0, rf_conf=0.0),
            "b": _make_class_metrics(skipped=True),
        }
        m = _aggregate_model_metrics("clf", per_class)
        assert m.bias_lift_score == pytest.approx(1.0)

    def test_zero_classes_yields_zero_metrics(self):
        m = _aggregate_model_metrics("clf", {})
        assert m.top1_accuracy_overall == 0.0
        assert m.bias_lift_score == 0.0

    def test_counts_classes(self):
        per_class = {
            "a": _make_class_metrics(),
            "b": _make_class_metrics(skipped=True),
            "c": _make_class_metrics(skipped=True),
        }
        m = _aggregate_model_metrics("clf", per_class)
        assert m.n_classes_scored == 1
        assert m.n_classes_skipped == 2


# ---------------------------------------------------------------------------
# Orchestrator + report
# ---------------------------------------------------------------------------

class TestEvaluateProbes:
    def test_returns_one_metrics_per_model(self, tmp_path):
        for f in ("a", "b", "c", "d"):
            Image.new("RGB", (8, 8)).save(tmp_path / f"{f}.jpg")
        manifest = {"classes": {"tench": {
            "bias_features": ["water"],
            "images": [
                {"image_path": "a.jpg", "variant": "bias_heavy"},
                {"image_path": "b.jpg", "variant": "bias_stripped"},
                {"image_path": "c.jpg", "variant": "real_feature_only"},
                {"image_path": "d.jpg", "variant": "adversarial"},
            ],
        }}}
        models = MagicMock()
        models.classifier.return_value = _classifier_returning(label="tench", confidence=0.7)
        out = evaluate_probes(manifest, ["clf1", "clf2"], models, manifest_dir=tmp_path)
        assert set(out.keys()) == {"clf1", "clf2"}
        assert out["clf1"].n_classes_scored == 1


class TestWriteReport:
    def test_includes_all_metrics(self, tmp_path):
        per_class = {"tench": _make_class_metrics()}
        m = _aggregate_model_metrics("clf", per_class)
        path = tmp_path / "report.json"
        write_report({"clf": m}, path, manifest_path="m.json")
        loaded = json.loads(path.read_text())
        assert loaded["models_evaluated"] == ["clf"]
        assert loaded["source_manifest"] == "m.json"
        clf_scores = loaded["scores"]["clf"]
        for key in ("top1_accuracy_overall", "top1_accuracy_bias_stripped",
                     "top1_accuracy_bias_heavy", "top1_accuracy_adversarial",
                     "bias_lift_score"):
            assert key in clf_scores

    def test_metrics_to_dict_json_serializable(self):
        per_class = {"tench": _make_class_metrics()}
        m = _aggregate_model_metrics("clf", per_class)
        d = _metrics_to_dict(m)
        json.dumps(d)  # must not raise
