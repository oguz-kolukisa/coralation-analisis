"""Tests for src/feature_extractor.py — the per-class feature catalog builder."""
from __future__ import annotations

import copy
import json

import pytest

from src.feature_extractor import (
    extract_catalog,
    load_and_extract,
    write_catalog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _edit(feature, ftype, verdicts, target="positive"):
    """Build one edit_result dict. verdicts = {model: verdict_str}."""
    return {
        "feature_name": feature,
        "feature_type": ftype,
        "edit_type": "feature_removal",
        "edit_instruction": f"Remove {feature}",
        "target": target,
        "original_image": "",
        "edited_image": "",
        "per_model": {m: {"original_confidence": 1.0, "edited_confidence": 0.5, "delta": -0.5}
                      for m in verdicts},
        "verdict": {m: {"verdict": v, "reasoning": ""} for m, v in verdicts.items()},
        "edit_failed": False,
    }


def _result(class_name, edit_results, confusing_classes=None):
    return {
        "class_name": class_name,
        "concepts": [],
        "unique_concepts": [],
        "edit_plans": [],
        "edit_results": edit_results,
        "feature_summaries": [],
        "confusing_classes": confusing_classes or [],
        "summary": {},
    }


@pytest.fixture
def simple_analysis():
    """2 classes, 2 models, mix of verdicts."""
    return {
        "version": "0.1.0",
        "generated_at": "2026-04-14 00:00 UTC",
        "results": [
            _result("tench", [
                _edit("dorsal fin", "target", {"resnet": "essential", "vit": "essential"}),
                _edit("blue water", "environmental", {"resnet": "spurious", "vit": "spurious"}),
                _edit("tail color", "target", {"resnet": "state_bias", "vit": "not_significant"}),
                _edit("net rope", "environmental", {"resnet": "edit_failed", "vit": "edit_failed"}),
            ]),
            _result("goldfish", [
                _edit("orange body", "target", {"resnet": "essential", "vit": "spurious"}),
            ]),
        ],
    }


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------

class TestExtractCatalogShape:
    def test_preserves_version(self, simple_analysis):
        assert extract_catalog(simple_analysis)["version"] == "0.1.0"

    def test_records_source_timestamp(self, simple_analysis):
        assert extract_catalog(simple_analysis)["source_generated_at"] == "2026-04-14 00:00 UTC"

    def test_discovers_all_models(self, simple_analysis):
        assert extract_catalog(simple_analysis)["source_models"] == ["resnet", "vit"]

    def test_has_entry_per_class(self, simple_analysis):
        classes = extract_catalog(simple_analysis)["classes"]
        assert set(classes) == {"tench", "goldfish"}

    def test_each_class_has_per_model_and_consensus(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        assert set(tench) == {"per_model", "consensus", "confusing_classes"}


# ---------------------------------------------------------------------------
# Per-model bucketing
# ---------------------------------------------------------------------------

class TestPerModelBuckets:
    def test_essential_goes_to_real(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        names = [e["feature"] for e in tench["per_model"]["resnet"]["real"]]
        assert "dorsal fin" in names

    def test_spurious_goes_to_bias_spurious(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        names = [e["feature"] for e in tench["per_model"]["resnet"]["bias_spurious"]]
        assert "blue water" in names

    def test_state_bias_goes_to_bias_state(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        names = [e["feature"] for e in tench["per_model"]["resnet"]["bias_state"]]
        assert "tail color" in names

    def test_edit_failed_is_inconclusive(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        names = [e["feature"] for e in tench["per_model"]["vit"]["inconclusive"]]
        assert "net rope" in names
        assert "tail color" in names  # not_significant on vit

    def test_entry_carries_feature_type(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        fin = next(e for e in tench["per_model"]["resnet"]["real"] if e["feature"] == "dorsal fin")
        assert fin["feature_type"] == "target"

    def test_entry_carries_signals_and_count(self, simple_analysis):
        tench = extract_catalog(simple_analysis)["classes"]["tench"]
        fin = next(e for e in tench["per_model"]["resnet"]["real"] if e["feature"] == "dorsal fin")
        assert fin["signals"] == ["essential"]
        assert fin["n_edits"] == 1


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

class TestConsensus:
    def test_strict_real_requires_all_models_agree(self, simple_analysis):
        consensus = extract_catalog(simple_analysis)["classes"]["tench"]["consensus"]
        assert consensus["real_strict"] == ["dorsal fin"]

    def test_any_real_is_union(self):
        analysis = {
            "results": [_result("c", [
                _edit("f1", "target", {"a": "essential", "b": "spurious"}),
            ])],
        }
        cat = extract_catalog(analysis)["classes"]["c"]["consensus"]
        assert cat["real_any"] == ["f1"]
        assert cat["bias_spurious_any"] == ["f1"]
        assert cat["real_strict"] == []

    def test_spurious_consensus_both_models(self, simple_analysis):
        consensus = extract_catalog(simple_analysis)["classes"]["tench"]["consensus"]
        assert consensus["bias_spurious_strict"] == ["blue water"]

    def test_state_bias_strict_requires_all_models(self, simple_analysis):
        consensus = extract_catalog(simple_analysis)["classes"]["tench"]["consensus"]
        assert consensus["bias_state_strict"] == []  # only resnet flagged
        assert consensus["bias_state_any"] == ["tail color"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestConfusingClasses:
    def test_propagates_confusing_classes(self):
        analysis = {"results": [_result("tench", [], confusing_classes=["goldfish", "carp"])]}
        cat = extract_catalog(analysis)
        assert cat["classes"]["tench"]["confusing_classes"] == ["goldfish", "carp"]

    def test_empty_when_missing(self):
        analysis = {"results": [_result("tench", [])]}
        cat = extract_catalog(analysis)
        assert cat["classes"]["tench"]["confusing_classes"] == []


class TestEdgeCases:
    def test_empty_results(self):
        catalog = extract_catalog({"results": []})
        assert catalog["classes"] == {}
        assert catalog["source_models"] == []

    def test_class_with_no_edits(self):
        analysis = {"results": [_result("empty_class", [])]}
        catalog = extract_catalog(analysis)
        assert catalog["classes"]["empty_class"]["per_model"] == {}

    def test_single_model(self):
        analysis = {"results": [_result("c", [
            _edit("f1", "target", {"solo": "essential"}),
        ])]}
        catalog = extract_catalog(analysis)
        assert catalog["source_models"] == ["solo"]
        assert catalog["classes"]["c"]["consensus"]["real_strict"] == ["f1"]

    def test_all_failed_edits_go_inconclusive(self):
        analysis = {"results": [_result("c", [
            _edit("f1", "target", {"m": "edit_failed"}),
            _edit("f2", "target", {"m": "not_significant"}),
        ])]}
        cat = extract_catalog(analysis)["classes"]["c"]["per_model"]["m"]
        assert cat["real"] == []
        assert cat["bias_spurious"] == []
        assert {e["feature"] for e in cat["inconclusive"]} == {"f1", "f2"}

    def test_feature_with_multiple_edits_aggregates_signals(self):
        analysis = {"results": [_result("c", [
            _edit("fin", "target", {"m": "essential"}),
            _edit("fin", "target", {"m": "state_bias"}),
        ])]}
        cat = extract_catalog(analysis)["classes"]["c"]["per_model"]["m"]
        real_fin = next(e for e in cat["real"] if e["feature"] == "fin")
        assert set(real_fin["signals"]) == {"essential", "state_bias"}
        assert real_fin["n_edits"] == 2
        # Same feature appears in both real and state buckets — that's informative
        assert any(e["feature"] == "fin" for e in cat["bias_state"])

    def test_intrinsic_verdict_counts_as_real(self):
        analysis = {"results": [_result("c", [
            _edit("body", "target", {"m": "intrinsic"}),
        ])]}
        cat = extract_catalog(analysis)["classes"]["c"]["per_model"]["m"]
        assert [e["feature"] for e in cat["real"]] == ["body"]

    def test_unknown_verdict_is_inconclusive(self):
        analysis = {"results": [_result("c", [
            _edit("x", "target", {"m": "unknown"}),
        ])]}
        cat = extract_catalog(analysis)["classes"]["c"]["per_model"]["m"]
        assert [e["feature"] for e in cat["inconclusive"]] == ["x"]


# ---------------------------------------------------------------------------
# Mutation safety & I/O
# ---------------------------------------------------------------------------

class TestSafetyAndIO:
    def test_does_not_mutate_input(self, simple_analysis):
        snapshot = copy.deepcopy(simple_analysis)
        extract_catalog(simple_analysis)
        assert simple_analysis == snapshot

    def test_output_is_json_serializable(self, simple_analysis):
        catalog = extract_catalog(simple_analysis)
        json.dumps(catalog)  # must not raise

    def test_load_and_write_roundtrip(self, tmp_path, simple_analysis):
        src = tmp_path / "analysis.json"
        src.write_text(json.dumps(simple_analysis))
        dst = tmp_path / "catalog.json"
        catalog = load_and_extract(src)
        write_catalog(catalog, dst)
        loaded = json.loads(dst.read_text())
        assert loaded == catalog
