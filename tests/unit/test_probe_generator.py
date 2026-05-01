"""Tests for src/probe_generator.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.probe_generator import (
    ClassProbeEntry,
    ProbeConfig,
    ProbeImage,
    ProbeManifest,
    _build_class_entry,
    _manifest_to_dict,
    _new_manifest,
    _persist_image,
    _render_all_variants,
    _render_one,
    _seed_for,
    _select_features,
    _slugify,
    _VARIANTS,
    _vlm_phase,
    _editor_phase,
    generate_probes,
    load_catalog,
    partition_round_robin,
    resolve_n_per_variant,
)
from src.vlm import ProbePromptSet


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

def _catalog_with(class_name="tench", strict=None, any_=None, real=None):
    return {
        "classes": {
            class_name: {
                "consensus": {
                    "bias_spurious_strict": strict or [],
                    "bias_spurious_any": any_ or [],
                    "real_any": real or [],
                    "real_strict": [],
                    "bias_state_strict": [],
                    "bias_state_any": [],
                },
            },
        },
    }


def _promptset(class_name="tench", n=2):
    return ProbePromptSet(
        class_name=class_name,
        bias_heavy=[f"heavy-{i}" for i in range(n)],
        bias_stripped=[f"stripped-{i}" for i in range(n)],
        real_feature_only=[f"real-{i}" for i in range(n)],
        adversarial=[f"adv-{i}" for i in range(n)],
    )


def _mock_editor(success=True):
    editor = MagicMock()
    if success:
        editor.generate_from_text.return_value = Image.new("RGB", (64, 64), "red")
    else:
        editor.generate_from_text.side_effect = RuntimeError("oom")
    return editor


def _mock_models(editor=None, vlm=None, cfg_vals=None):
    models = MagicMock()
    models.vlm.return_value = vlm or MagicMock()
    models.editor.return_value = editor or _mock_editor()
    models.cfg = MagicMock()
    for k, v in (cfg_vals or {}).items():
        setattr(models.cfg, k, v)
    return models


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class TestResolveNPerVariant:
    def test_user_override_wins(self):
        assert resolve_n_per_variant(10, 7) == 7

    def test_auto_formula_ceil_sqrt(self):
        assert resolve_n_per_variant(4, None) == 2
        assert resolve_n_per_variant(5, None) == 3
        assert resolve_n_per_variant(9, None) == 3
        assert resolve_n_per_variant(10, None) == 4
        assert resolve_n_per_variant(16, None) == 4

    def test_floor_of_one_for_zero_or_one(self):
        assert resolve_n_per_variant(0, None) == 1
        assert resolve_n_per_variant(1, None) == 1

    def test_negative_user_value_falls_back_to_auto(self):
        assert resolve_n_per_variant(9, 0) == 3
        assert resolve_n_per_variant(9, -1) == 3


class TestPartitionRoundRobin:
    def test_even_split(self):
        groups = partition_round_robin(["a", "b", "c", "d"], 2)
        # Round-robin: a→g0, b→g1, c→g0, d→g1
        assert groups == [["a", "c"], ["b", "d"]]

    def test_uneven_split_puts_extras_in_early_groups(self):
        groups = partition_round_robin(["a", "b", "c", "d", "e"], 3)
        assert groups == [["a", "d"], ["b", "e"], ["c"]]

    def test_empty_features_produces_empty_groups(self):
        assert partition_round_robin([], 3) == [[], [], []]

    def test_more_groups_than_items_leaves_trailing_empty(self):
        groups = partition_round_robin(["a", "b"], 5)
        assert groups == [["a"], ["b"], [], [], []]

    def test_every_feature_appears_exactly_once(self):
        items = ["water", "glass", "rocks", "plants", "surface", "tank"]
        groups = partition_round_robin(items, 3)
        flat = [x for g in groups for x in g]
        assert sorted(flat) == sorted(items)


class TestSlugify:
    def test_lowercases(self):
        assert _slugify("Tench") == "tench"

    def test_replaces_non_alnum(self):
        assert _slugify("goldfish, Carassius auratus") == "goldfish_carassius_auratus"

    def test_strips_leading_trailing_underscores(self):
        assert _slugify("  hello  ") == "hello"

    def test_handles_empty(self):
        assert _slugify("") == ""


class TestSeedFor:
    def test_base_plus_offsets(self):
        assert _seed_for(0, 0, 42) == 42
        assert _seed_for(1, 0, 42) == 1042
        assert _seed_for(2, 3, 42) == 2045

    def test_different_indexes_give_different_seeds(self):
        seeds = {_seed_for(v, i, 42) for v in range(3) for i in range(3)}
        assert len(seeds) == 9


class TestSelectFeatures:
    def test_strict_when_strict_nonempty(self):
        class_data = {"consensus": {
            "bias_spurious_strict": ["a"], "bias_spurious_any": ["a", "b"],
            "real_any": ["r"]}}
        bias, real, used = _select_features(class_data, "strict")
        assert bias == ["a"]
        assert real == ["r"]
        assert used == "strict"

    def test_strict_falls_back_to_any(self):
        class_data = {"consensus": {
            "bias_spurious_strict": [], "bias_spurious_any": ["b"],
            "real_any": []}}
        bias, _, used = _select_features(class_data, "strict")
        assert bias == ["b"]
        assert used == "any"

    def test_both_empty_returns_none(self):
        class_data = {"consensus": {
            "bias_spurious_strict": [], "bias_spurious_any": [],
            "real_any": []}}
        bias, _, used = _select_features(class_data, "strict")
        assert bias == []
        assert used == "none"

    def test_any_source_uses_any_directly(self):
        class_data = {"consensus": {
            "bias_spurious_strict": ["strict_only"],
            "bias_spurious_any": ["strict_only", "any_too"],
            "real_any": []}}
        bias, _, used = _select_features(class_data, "any")
        assert bias == ["strict_only", "any_too"]
        assert used == "any"

    def test_any_source_returns_none_when_empty(self):
        class_data = {"consensus": {"bias_spurious_any": [], "real_any": []}}
        bias, _, used = _select_features(class_data, "any")
        assert bias == []
        assert used == "none"


class TestPersistImage:
    def test_writes_jpeg(self, tmp_path):
        path = tmp_path / "out.jpg"
        img = Image.new("RGB", (32, 32), "blue")
        _persist_image(img, path)
        assert path.exists()
        assert Image.open(path).size == (32, 32)


class TestLoadCatalog:
    def test_reads_json(self, tmp_path):
        path = tmp_path / "cat.json"
        path.write_text(json.dumps({"classes": {"a": {}}}))
        assert load_catalog(path) == {"classes": {"a": {}}}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRenderOne:
    def test_returns_probeimage_on_success(self, tmp_path):
        editor = _mock_editor()
        probe = _render_one(editor, "tench", "bias_heavy", 0,
                             "a prompt", 0, ProbeConfig(), tmp_path, tmp_path.parent)
        assert isinstance(probe, ProbeImage)
        assert probe.variant == "bias_heavy"
        assert probe.index == 0
        assert probe.seed == 42

    def test_saves_image_to_disk(self, tmp_path):
        editor = _mock_editor()
        _render_one(editor, "tench", "bias_heavy", 0, "p", 0,
                     ProbeConfig(), tmp_path, tmp_path.parent)
        assert (tmp_path / "bias_heavy_0.jpg").exists()

    def test_returns_none_on_failure(self, tmp_path):
        editor = _mock_editor(success=False)
        probe = _render_one(editor, "tench", "bias_heavy", 0, "p", 0,
                             ProbeConfig(), tmp_path, tmp_path.parent)
        assert probe is None

    def test_relative_path_uses_posix(self, tmp_path):
        output_root = tmp_path
        class_dir = output_root / "tench"
        class_dir.mkdir()
        editor = _mock_editor()
        probe = _render_one(editor, "tench", "bias_heavy", 0, "p", 0,
                             ProbeConfig(), class_dir, output_root)
        assert "/" in probe.image_path  # posix, not backslash


class TestRenderAllVariants:
    def test_renders_every_variant_and_index(self, tmp_path):
        editor = _mock_editor()
        entry = _build_class_entry(
            {"consensus": {"bias_spurious_strict": ["x"], "bias_spurious_any": ["x"],
                           "real_any": []}},
            _promptset(n=2), ProbeConfig(n_per_variant=2),
        )
        _render_all_variants(editor, _promptset(n=2), entry,
                              ProbeConfig(n_per_variant=2), tmp_path, tmp_path.parent)
        # 4 variants × 2 images = 8
        assert len(entry.images) == 8

    def test_skips_failed_renders(self, tmp_path):
        editor = _mock_editor(success=False)
        entry = _build_class_entry(
            {"consensus": {"bias_spurious_strict": ["x"], "bias_spurious_any": ["x"],
                           "real_any": []}},
            _promptset(n=2), ProbeConfig(n_per_variant=2),
        )
        _render_all_variants(editor, _promptset(n=2), entry,
                              ProbeConfig(n_per_variant=2), tmp_path, tmp_path.parent)
        assert entry.images == []


class TestBuildClassEntry:
    def test_records_feature_source_used(self):
        class_data = {"consensus": {
            "bias_spurious_strict": [], "bias_spurious_any": ["x"],
            "real_any": ["y"]}}
        entry = _build_class_entry(class_data, _promptset(), ProbeConfig())
        assert entry.feature_source_used == "any"

    def test_marks_synthesized_fallback_when_no_bias(self):
        class_data = {"consensus": {"bias_spurious_strict": [], "bias_spurious_any": [],
                                     "real_any": []}}
        entry = _build_class_entry(class_data, _promptset(), ProbeConfig())
        assert entry.synthesized_fallback is True

    def test_copies_prompts_from_promptset(self):
        entry = _build_class_entry(
            {"consensus": {"bias_spurious_strict": ["b"], "bias_spurious_any": ["b"],
                           "real_any": []}},
            _promptset(n=1), ProbeConfig(),
        )
        assert entry.prompts == {
            "bias_heavy": ["heavy-0"],
            "bias_stripped": ["stripped-0"],
            "real_feature_only": ["real-0"],
            "adversarial": ["adv-0"],
        }

    def test_propagates_confusing_classes(self):
        class_data = {
            "consensus": {"bias_spurious_strict": ["x"], "bias_spurious_any": ["x"],
                          "real_any": []},
            "confusing_classes": ["goldfish", "carp"],
        }
        entry = _build_class_entry(class_data, _promptset(), ProbeConfig())
        assert entry.confusing_classes == ["goldfish", "carp"]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:
    def test_json_serializable(self):
        cfg = ProbeConfig()
        models = _mock_models(cfg_vals={"editor_model": "flux", "vlm_model": "qwen"})
        m = _new_manifest(cfg, models, "source.json")
        m.classes["tench"] = ClassProbeEntry()
        out = _manifest_to_dict(m)
        # Serialize to JSON string — must not raise
        s = json.dumps(out)
        loaded = json.loads(s)
        assert loaded["editor_model"] == "flux"
        assert loaded["source_catalog"] == "source.json"

    def test_image_size_tuple_converts_to_list(self):
        cfg = ProbeConfig(image_size=(512, 768))
        models = _mock_models()
        m = _new_manifest(cfg, models, "")
        out = _manifest_to_dict(m)
        assert out["config"]["image_size"] == [512, 768]


# ---------------------------------------------------------------------------
# End-to-end orchestrator (fully mocked)
# ---------------------------------------------------------------------------

class TestGenerateProbes:
    def _mock_vlm_returning(self, prompt_set):
        vlm = MagicMock()
        vlm.generate_probe_prompts.return_value = prompt_set
        return vlm

    def test_creates_manifest_with_one_class(self, tmp_path):
        catalog = _catalog_with("tench", strict=["water"])
        vlm = self._mock_vlm_returning(_promptset("tench", n=1))
        models = _mock_models(editor=_mock_editor(), vlm=vlm,
                               cfg_vals={"editor_model": "flux", "vlm_model": "qwen"})
        manifest = generate_probes(catalog, tmp_path, ProbeConfig(n_per_variant=1),
                                    models, source_catalog="cat.json")
        assert "tench" in manifest.classes
        assert len(manifest.classes["tench"].images) == 4  # 4 variants × 1

    def test_writes_manifest_json_to_disk(self, tmp_path):
        catalog = _catalog_with("tench", strict=["water"])
        vlm = self._mock_vlm_returning(_promptset("tench", n=1))
        models = _mock_models(vlm=vlm)
        generate_probes(catalog, tmp_path, ProbeConfig(n_per_variant=1), models)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert "tench" in loaded["classes"]

    def test_handles_multiple_classes(self, tmp_path):
        catalog = {"classes": {
            "tench": _catalog_with("tench", strict=["water"])["classes"]["tench"],
            "hen": _catalog_with("hen", strict=["hay"])["classes"]["hen"],
        }}
        vlm = MagicMock()
        vlm.generate_probe_prompts.side_effect = [
            _promptset("tench", n=1), _promptset("hen", n=1),
        ]
        models = _mock_models(vlm=vlm)
        manifest = generate_probes(catalog, tmp_path, ProbeConfig(n_per_variant=1), models)
        assert set(manifest.classes.keys()) == {"tench", "hen"}

    def test_offloads_vlm_and_editor(self, tmp_path):
        catalog = _catalog_with("tench", strict=["water"])
        vlm = self._mock_vlm_returning(_promptset("tench", n=1))
        models = _mock_models(vlm=vlm)
        generate_probes(catalog, tmp_path, ProbeConfig(n_per_variant=1), models)
        models.offload_vlm.assert_called_once()
        models.offload_editor.assert_called_once()

    def test_skips_render_on_editor_exception(self, tmp_path):
        catalog = _catalog_with("tench", strict=["water"])
        vlm = self._mock_vlm_returning(_promptset("tench", n=1))
        models = _mock_models(editor=_mock_editor(success=False), vlm=vlm)
        manifest = generate_probes(catalog, tmp_path, ProbeConfig(n_per_variant=1), models)
        # Entry exists but no images persisted
        assert manifest.classes["tench"].images == []
