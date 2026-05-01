"""Tests for src/probe_reporter.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.probe_reporter import (
    _VARIANTS,
    _class_block,
    _feature_list,
    _grid_header_row,
    _grid_variant_row,
    _group_images_by_variant,
    _html_header,
    _image_cell,
    _image_grid,
    _lift_css,
    _per_class_model_table,
    _per_class_row,
    _per_class_sections,
    _ranking_row,
    _ranking_table,
    _relative_images_base,
    _sorted_models,
    _variant_cells,
    render_probe_report,
    write_probe_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _evaluation(models=("resnet50", "clip_vitb32")):
    scores = {}
    for i, name in enumerate(models):
        scores[name] = {
            "model_name": name,
            "top1_accuracy_overall": 0.9 - 0.1 * i,
            "top1_accuracy_bias_stripped": 0.0,
            "top1_accuracy_bias_heavy": 0.8 - 0.1 * i,
            "bias_lift_score": 0.15 if i == 0 else -0.05,
            "n_classes_scored": 1,
            "n_classes_skipped": 0,
            "mean_conf_bias_heavy": 0.7,
            "mean_conf_bias_stripped": 0.01,
            "mean_conf_real_feature_only": 0.5,
            "per_class": {"tench": {
                "class_name": "tench",
                "bias_heavy": {"top1_accuracy": 0.7, "mean_conf": 0.8, "n": 3},
                "bias_stripped": {"top1_accuracy": 0.0, "mean_conf": 0.02, "n": 3},
                "real_feature_only": {"top1_accuracy": 0.3, "mean_conf": 0.4, "n": 3},
                "bias_lift": 0.4,
                "skipped": False,
                "skip_reason": "",
            }},
        }
    return {
        "version": "0.3.0",
        "generated_at": "2026-04-15 14:00 UTC",
        "models_evaluated": list(models),
        "scores": scores,
    }


def _manifest():
    return {
        "version": "0.3.0",
        "generated_at": "2026-04-15 13:00 UTC",
        "classes": {
            "tench": {
                "bias_features": ["water", "pond"],
                "real_features": ["dorsal fin"],
                "feature_source_used": "strict",
                "images": [
                    {"variant": "bias_heavy", "prompt": "fish in a pond",
                     "seed": 42, "image_path": "tench/bias_heavy_0.jpg", "index": 0},
                    {"variant": "bias_heavy", "prompt": "fish in water",
                     "seed": 43, "image_path": "tench/bias_heavy_1.jpg", "index": 1},
                    {"variant": "bias_stripped", "prompt": "empty pond",
                     "seed": 1042, "image_path": "tench/bias_stripped_0.jpg", "index": 0},
                    {"variant": "real_feature_only", "prompt": "fish on white bg",
                     "seed": 2042, "image_path": "tench/real_feature_only_0.jpg", "index": 0},
                ],
                "vlm_failed": False,
                "synthesized_fallback": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class TestLiftCss:
    def test_positive_is_pos(self):
        assert _lift_css(0.2) == "pos"

    def test_negative_is_neg(self):
        assert _lift_css(-0.2) == "neg"

    def test_near_zero_is_neutral(self):
        assert _lift_css(0.02) == "neutral"
        assert _lift_css(-0.02) == "neutral"


class TestSortedModels:
    def test_descending_by_overall(self):
        ev = _evaluation(models=("a", "b", "c"))
        # a=0.9, b=0.8, c=0.7
        out = _sorted_models(ev)
        assert [r["name"] for r in out] == ["a", "b", "c"]

    def test_empty_scores_returns_empty(self):
        assert _sorted_models({"scores": {}}) == []


class TestRelativeImagesBase:
    def test_sibling_dirs(self, tmp_path):
        out = tmp_path / "reports" / "probe_report.html"
        mf = tmp_path / "probes" / "manifest.json"
        mf.parent.mkdir(parents=True)
        mf.touch()
        # output will be output/../probes = '../probes'
        result = _relative_images_base(out, mf)
        assert result == "../probes"


# ---------------------------------------------------------------------------
# Header / ranking
# ---------------------------------------------------------------------------

class TestHtmlHeader:
    def test_escapes_timestamp(self):
        ev = {"generated_at": "2026-<script>"}
        h = _html_header(ev)
        assert "<script>" not in h
        assert "&lt;script&gt;" in h

    def test_contains_title(self):
        assert "Probe evaluation report" in _html_header(_evaluation())


class TestRankingTable:
    def test_first_row_gets_rank1_class(self):
        ev = _evaluation(models=("best", "worst"))
        html_str = _ranking_table(ev)
        assert "rank-1" in html_str

    def test_last_row_gets_rank_last_class(self):
        ev = _evaluation(models=("best", "worst"))
        html_str = _ranking_table(ev)
        assert "rank-last" in html_str

    def test_includes_all_models(self):
        ev = _evaluation(models=("a", "b", "c"))
        html_str = _ranking_table(ev)
        assert "a" in html_str and "b" in html_str and "c" in html_str


class TestRankingRow:
    def test_renders_four_metric_columns(self):
        row = {"name": "clf", "top1_accuracy_overall": 0.5,
               "top1_accuracy_bias_heavy": 0.7, "top1_accuracy_bias_stripped": 0.1,
               "bias_lift_score": 0.1, "n_classes_scored": 3, "n_classes_skipped": 0}
        r = _ranking_row(row, 0, 2)
        assert "0.500" in r and "0.700" in r and "0.100" in r and "+0.100" in r

    def test_negative_lift_has_green(self):
        row = {"name": "clf", "top1_accuracy_overall": 0.5,
               "top1_accuracy_bias_heavy": 0.7, "top1_accuracy_bias_stripped": 0.1,
               "bias_lift_score": -0.2, "n_classes_scored": 3, "n_classes_skipped": 0}
        r = _ranking_row(row, 1, 3)
        assert "neg" in r


# ---------------------------------------------------------------------------
# Per-class section
# ---------------------------------------------------------------------------

class TestFeatureList:
    def test_shows_bias_and_real(self):
        entry = {"bias_features": ["water"], "real_features": ["fin"],
                  "feature_source_used": "strict"}
        h = _feature_list(entry)
        assert "water" in h and "fin" in h and "strict" in h

    def test_empty_features_show_none(self):
        entry = {"bias_features": [], "real_features": [],
                  "feature_source_used": "none"}
        h = _feature_list(entry)
        assert "(none)" in h

    def test_escapes_feature_names(self):
        entry = {"bias_features": ["<evil>"], "real_features": [],
                  "feature_source_used": "strict"}
        h = _feature_list(entry)
        assert "<evil>" not in h
        assert "&lt;evil&gt;" in h


class TestImageGrouping:
    def test_groups_by_variant(self):
        entry = _manifest()["classes"]["tench"]
        groups = _group_images_by_variant(entry)
        assert len(groups["bias_heavy"]) == 2
        assert len(groups["bias_stripped"]) == 1
        assert len(groups["real_feature_only"]) == 1

    def test_sorts_by_index(self):
        entry = {"images": [
            {"variant": "bias_heavy", "index": 2, "image_path": "a", "prompt": "", "seed": 0},
            {"variant": "bias_heavy", "index": 0, "image_path": "b", "prompt": "", "seed": 0},
            {"variant": "bias_heavy", "index": 1, "image_path": "c", "prompt": "", "seed": 0},
        ]}
        groups = _group_images_by_variant(entry)
        assert [img["index"] for img in groups["bias_heavy"]] == [0, 1, 2]


class TestImageGrid:
    def test_returns_placeholder_when_no_images(self):
        entry = {"images": []}
        html_str = _image_grid(entry, "../probes")
        assert "No probe images" in html_str

    def test_includes_all_variants_in_rows(self):
        entry = _manifest()["classes"]["tench"]
        html_str = _image_grid(entry, "../probes")
        for v in _VARIANTS:
            assert v in html_str

    def test_uses_relative_base_path(self):
        entry = _manifest()["classes"]["tench"]
        html_str = _image_grid(entry, "../probes")
        assert "../probes/tench/bias_heavy_0.jpg" in html_str


class TestImageCell:
    def test_returns_blank_figure_for_none(self):
        assert _image_cell(None, "../probes") == "<figure></figure>"

    def test_includes_prompt_in_caption(self):
        img = {"image_path": "tench/bh_0.jpg", "variant": "bias_heavy",
                "index": 0, "prompt": "fish in pond"}
        html_str = _image_cell(img, "../probes")
        assert "fish in pond" in html_str

    def test_escapes_prompt(self):
        img = {"image_path": "x.jpg", "variant": "bias_heavy",
                "index": 0, "prompt": "<script>bad</script>"}
        html_str = _image_cell(img, "../probes")
        assert "<script>bad" not in html_str
        assert "&lt;script&gt;bad" in html_str


class TestPerClassModelTable:
    def test_row_per_model(self):
        ev = _evaluation(models=("a", "b", "c"))
        html_str = _per_class_model_table("tench", ev)
        assert html_str.count("<tr>") >= 4  # header + 3 data rows (tbody has 3)

    def test_missing_class_shows_dashes(self):
        ev = _evaluation(models=("a",))
        html_str = _per_class_model_table("unknown_class", ev)
        assert "—" in html_str
        assert "missing" in html_str


class TestPerClassRow:
    def test_renders_skipped_class(self):
        cm = {"class_name": "tench", "skipped": True,
               "skip_reason": "no_bias_features", "bias_lift": 0.0}
        r = _per_class_row("clf", cm)
        assert "no_bias_features" in r


class TestVariantCells:
    def test_two_dashes_when_none(self):
        assert _variant_cells(None) == "<td>—</td><td>—</td>"

    def test_formats_top1_and_conf(self):
        vs = {"top1_accuracy": 0.75, "mean_conf": 0.50, "n": 3}
        html_str = _variant_cells(vs)
        assert "0.750" in html_str and "0.500" in html_str


# ---------------------------------------------------------------------------
# Full document
# ---------------------------------------------------------------------------

class TestRenderProbeReport:
    def test_contains_doctype(self):
        html_str = render_probe_report(_manifest(), _evaluation())
        assert html_str.startswith("<!doctype html>")

    def test_contains_ranking_and_per_class_sections(self):
        html_str = render_probe_report(_manifest(), _evaluation())
        assert "Leaderboard" in html_str
        assert "Per-class probes" in html_str

    def test_empty_manifest_shows_placeholder(self):
        html_str = render_probe_report({"classes": {}}, _evaluation())
        assert "No classes in manifest" in html_str


class TestWriteProbeReport:
    def test_writes_html_to_disk(self, tmp_path):
        mf = tmp_path / "probes" / "manifest.json"
        mf.parent.mkdir(parents=True)
        mf.write_text(json.dumps(_manifest()))
        ev = tmp_path / "reports" / "evaluation.json"
        ev.parent.mkdir(parents=True)
        ev.write_text(json.dumps(_evaluation()))
        out = tmp_path / "reports" / "probe_report.html"
        path = write_probe_report(mf, ev, out)
        assert path == out
        assert out.exists()
        assert "<!doctype html>" in out.read_text()
