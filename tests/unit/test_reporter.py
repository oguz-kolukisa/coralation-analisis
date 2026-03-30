"""Tests for src/reporter.py — NumpyEncoder, helpers, and Reporter class."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.reporter import Reporter, _NumpyEncoder, _get_feature_display_name, _to_relative_path


def _config_with_timing(timing: dict) -> dict:
    """Build a full config dict with timing info attached."""
    cfg = Config().model_dump()
    cfg["timing"] = timing
    return cfg


# ============================================================================
# _NumpyEncoder
# ============================================================================

class TestNumpyEncoderBool:
    def test_numpy_bool_true(self):
        assert json.dumps({"v": np.bool_(True)}, cls=_NumpyEncoder) == '{"v": true}'

    def test_numpy_bool_false(self):
        assert json.dumps({"v": np.bool_(False)}, cls=_NumpyEncoder) == '{"v": false}'


class TestNumpyEncoderInteger:
    def test_int32(self):
        result = json.loads(json.dumps({"v": np.int32(42)}, cls=_NumpyEncoder))
        assert result["v"] == 42

    def test_int64(self):
        result = json.loads(json.dumps({"v": np.int64(-7)}, cls=_NumpyEncoder))
        assert result["v"] == -7

    def test_uint8(self):
        result = json.loads(json.dumps({"v": np.uint8(255)}, cls=_NumpyEncoder))
        assert result["v"] == 255


class TestNumpyEncoderFloat:
    def test_float32(self):
        result = json.loads(json.dumps({"v": np.float32(3.14)}, cls=_NumpyEncoder))
        assert abs(result["v"] - 3.14) < 0.01

    def test_float64(self):
        result = json.loads(json.dumps({"v": np.float64(2.718)}, cls=_NumpyEncoder))
        assert abs(result["v"] - 2.718) < 0.001


class TestNumpyEncoderArray:
    def test_1d_array(self):
        result = json.loads(json.dumps({"v": np.array([1, 2, 3])}, cls=_NumpyEncoder))
        assert result["v"] == [1, 2, 3]

    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = json.loads(json.dumps({"v": arr}, cls=_NumpyEncoder))
        assert result["v"] == [[1, 2], [3, 4]]

    def test_empty_array(self):
        result = json.loads(json.dumps({"v": np.array([])}, cls=_NumpyEncoder))
        assert result["v"] == []


class TestNumpyEncoderFallback:
    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            json.dumps({"v": object()}, cls=_NumpyEncoder)

    def test_native_types_pass_through(self):
        data = {"s": "hello", "i": 42, "f": 3.14, "b": True, "n": None}
        result = json.loads(json.dumps(data, cls=_NumpyEncoder))
        assert result == data


class TestNumpyEncoderNested:
    def test_mixed_numpy_in_nested_dict(self):
        data = {
            "confirmed": np.bool_(True),
            "score": np.float64(0.95),
            "counts": np.array([1, 2]),
            "meta": {"flag": np.bool_(False), "idx": np.int32(3)},
        }
        result = json.loads(json.dumps(data, cls=_NumpyEncoder))
        assert result["confirmed"] is True
        assert result["meta"]["flag"] is False
        assert result["meta"]["idx"] == 3
        assert result["counts"] == [1, 2]


# ============================================================================
# _get_feature_display_name
# ============================================================================

class TestGetFeatureDisplayName:
    def test_returns_feature_name_when_present(self):
        assert _get_feature_display_name({"feature_name": "ears"}) == "ears"

    def test_falls_back_to_truncated_instruction(self):
        long_instr = "Remove the background and replace it with blue"
        result = _get_feature_display_name({"instruction": long_instr})
        assert result == long_instr[:30]
        assert len(result) == 30

    def test_short_instruction_not_truncated(self):
        assert _get_feature_display_name({"instruction": "Remove ears"}) == "Remove ears"

    def test_empty_feature_name_uses_instruction(self):
        result = _get_feature_display_name({"feature_name": "", "instruction": "Remove bg"})
        assert result == "Remove bg"

    def test_no_feature_name_no_instruction(self):
        assert _get_feature_display_name({}) == "Unknown"

    def test_empty_instruction_returns_unknown(self):
        assert _get_feature_display_name({"instruction": ""}) == "Unknown"

    def test_none_feature_name_uses_instruction(self):
        result = _get_feature_display_name({"feature_name": None, "instruction": "test"})
        assert result == "test"


# ============================================================================
# _to_relative_path
# ============================================================================

class TestToRelativePath:
    def test_converts_child_path(self):
        result = _to_relative_path("/output/class/img.jpg", Path("/output"))
        assert result == "class/img.jpg"

    def test_sibling_path_uses_dotdot(self):
        result = _to_relative_path("/other/dir/img.jpg", Path("/output"))
        assert result == "../other/dir/img.jpg"

    def test_same_directory(self):
        result = _to_relative_path("/output/img.jpg", Path("/output"))
        assert result == "img.jpg"

    def test_empty_string_path(self):
        # Empty path resolves to cwd, which won't be relative to /output
        result = _to_relative_path("", Path("/output"))
        assert isinstance(result, str)


# ============================================================================
# Reporter
# ============================================================================

class TestReporterInit:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "new_dir"
        Reporter(out)
        assert out.exists()

    def test_default_config_is_empty_dict(self, tmp_path):
        reporter = Reporter(tmp_path)
        assert reporter.config == {}

    def test_stores_config(self, tmp_path):
        cfg = {"model": "resnet50"}
        reporter = Reporter(tmp_path, config=cfg)
        assert reporter.config == cfg


class TestSaveConsolidatedJson:
    def test_writes_json_file(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{"class_name": "cat", "confirmed": True}]
        path = reporter.save_consolidated_json(results)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == results

    def test_handles_numpy_types(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{"score": np.float64(0.95), "ok": np.bool_(True)}]
        path = reporter.save_consolidated_json(results)
        data = json.loads(path.read_text())
        assert data[0]["score"] == pytest.approx(0.95)
        assert data[0]["ok"] is True

    def test_empty_results(self, tmp_path):
        reporter = Reporter(tmp_path)
        path = reporter.save_consolidated_json([])
        assert json.loads(path.read_text()) == []

    def test_overwrites_existing_file(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.save_consolidated_json([{"v": 1}])
        reporter.save_consolidated_json([{"v": 2}])
        data = json.loads((tmp_path / "analysis_results.json").read_text())
        assert data == [{"v": 2}]


class TestGenerateHtml:
    def test_produces_html_file(self, tmp_path):
        reporter = Reporter(tmp_path)
        path = reporter.generate_html([])
        assert path.suffix == ".html"
        assert path.exists()

    def test_html_contains_doctype(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([])
        content = (tmp_path / "report.html").read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_does_not_contain_error_tags(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([])
        content = (tmp_path / "report.html").read_text()
        assert "Error" not in content
        assert "error" not in content.lower().split("border")[0]  # ignore CSS


class TestGenerateMarkdown:
    def test_produces_md_file(self, tmp_path):
        reporter = Reporter(tmp_path)
        path = reporter.generate_markdown([])
        assert path.suffix == ".md"
        assert path.exists()


class TestGenerateAll:
    def test_returns_all_three_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        paths = reporter.generate_all([])
        assert set(paths.keys()) == {"json", "html", "markdown"}
        for p in paths.values():
            assert p.exists()


class TestConvertPathsToRelative:
    def test_converts_edit_result_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        abs_img = str(tmp_path / "class1" / "orig.jpg")
        results = [{
            "edit_results": [{
                "original_image_path": abs_img,
                "generations": [],
            }],
        }]
        converted = reporter._convert_paths_to_relative(results)
        assert converted[0]["edit_results"][0]["original_image_path"] == "class1/orig.jpg"

    def test_does_not_mutate_original(self, tmp_path):
        reporter = Reporter(tmp_path)
        abs_img = str(tmp_path / "class1" / "orig.jpg")
        results = [{"edit_results": [{"original_image_path": abs_img, "generations": []}]}]
        reporter._convert_paths_to_relative(results)
        assert results[0]["edit_results"][0]["original_image_path"] == abs_img

    def test_converts_generation_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{
            "edit_results": [{
                "original_image_path": str(tmp_path / "orig.jpg"),
                "generations": [
                    {"edited_image_path": str(tmp_path / "edited.jpg")},
                ],
            }],
        }]
        converted = reporter._convert_paths_to_relative(results)
        gen = converted[0]["edit_results"][0]["generations"][0]
        assert gen["edited_image_path"] == "edited.jpg"

    def test_converts_baseline_results_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{
            "baseline_results": [
                {"image_path": str(tmp_path / "base.jpg")},
            ],
        }]
        converted = reporter._convert_paths_to_relative(results)
        assert converted[0]["baseline_results"][0]["image_path"] == "base.jpg"

    def test_converts_confirmed_hypotheses_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{
            "confirmed_hypotheses": [{
                "original_image_path": str(tmp_path / "img.jpg"),
                "generations": [],
            }],
        }]
        converted = reporter._convert_paths_to_relative(results)
        assert converted[0]["confirmed_hypotheses"][0]["original_image_path"] == "img.jpg"

    def test_handles_empty_results(self, tmp_path):
        reporter = Reporter(tmp_path)
        assert reporter._convert_paths_to_relative([]) == []

    def test_handles_missing_optional_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{
            "edit_results": [{
                "original_image_path": str(tmp_path / "img.jpg"),
                "generations": [{"edited_image_path": str(tmp_path / "e.jpg")}],
            }],
        }]
        # No gradcam paths — should not raise
        converted = reporter._convert_paths_to_relative(results)
        assert "original_gradcam_path" not in converted[0]["edit_results"][0]

    def test_converts_gradcam_paths(self, tmp_path):
        reporter = Reporter(tmp_path)
        results = [{
            "edit_results": [{
                "original_image_path": str(tmp_path / "img.jpg"),
                "original_gradcam_path": str(tmp_path / "gc.jpg"),
                "generations": [{
                    "edited_image_path": str(tmp_path / "e.jpg"),
                    "gradcam_image_path": str(tmp_path / "gc2.jpg"),
                    "gradcam_diff_path": str(tmp_path / "diff.jpg"),
                }],
            }],
        }]
        converted = reporter._convert_paths_to_relative(results)
        edit = converted[0]["edit_results"][0]
        assert edit["original_gradcam_path"] == "gc.jpg"
        gen = edit["generations"][0]
        assert gen["gradcam_image_path"] == "gc2.jpg"
        assert gen["gradcam_diff_path"] == "diff.jpg"


# ============================================================================
# Timing display in HTML
# ============================================================================

class TestHtmlTimingDisplay:
    def test_timing_rendered_when_present(self, tmp_path):
        config = _config_with_timing({
            "start": "2026-03-23 10:00:00 UTC",
            "end": "2026-03-23 10:05:30 UTC",
            "duration_seconds": 330,
        })
        reporter = Reporter(tmp_path, config=config)
        reporter.generate_html([])
        html = (tmp_path / "report.html").read_text()
        assert "Execution Timing" in html
        assert "2026-03-23 10:00:00 UTC" in html
        assert "2026-03-23 10:05:30 UTC" in html
        assert "5m 30s" in html

    def test_timing_hours_format(self, tmp_path):
        config = _config_with_timing({
            "start": "2026-03-23 08:00:00 UTC",
            "end": "2026-03-23 09:02:05 UTC",
            "duration_seconds": 3725,
        })
        reporter = Reporter(tmp_path, config=config)
        reporter.generate_html([])
        html = (tmp_path / "report.html").read_text()
        assert "1h 2m 5s" in html

    def test_timing_seconds_only(self, tmp_path):
        config = _config_with_timing({
            "start": "2026-03-23 10:00:00 UTC",
            "end": "2026-03-23 10:00:45 UTC",
            "duration_seconds": 45,
        })
        reporter = Reporter(tmp_path, config=config)
        reporter.generate_html([])
        html = (tmp_path / "report.html").read_text()
        assert "45s" in html

    def test_no_timing_when_absent(self, tmp_path):
        reporter = Reporter(tmp_path, config={})
        reporter.generate_html([])
        html = (tmp_path / "report.html").read_text()
        assert "Execution Timing" not in html


# ============================================================================
# Prefix support
# ============================================================================

class TestReporterPrefix:
    def test_no_prefix_uses_default_names(self, tmp_path):
        reporter = Reporter(tmp_path)
        assert reporter.prefix == ""
        paths = reporter.generate_all([])
        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "analysis_results.json").exists()

    def test_prefix_prepends_to_filenames(self, tmp_path):
        reporter = Reporter(tmp_path, prefix="resnet50")
        assert reporter.prefix == "resnet50_"
        paths = reporter.generate_all([])
        assert (tmp_path / "resnet50_report.html").exists()
        assert (tmp_path / "resnet50_report.md").exists()
        assert (tmp_path / "resnet50_analysis_results.json").exists()

    def test_two_prefixes_coexist(self, tmp_path):
        Reporter(tmp_path, prefix="resnet50").generate_all([])
        Reporter(tmp_path, prefix="vit_l_16").generate_all([])
        assert (tmp_path / "resnet50_report.html").exists()
        assert (tmp_path / "vit_l_16_report.html").exists()


# ============================================================================
# HTML report content validation
# ============================================================================

def _make_sample_result():
    """Build a minimal but realistic result dict for report rendering."""
    return {
        "class_name": "tabby cat",
        "key_features": ["ear shape", "stripe pattern"],
        "essential_features": ["fur texture", "ear shape"],
        "spurious_features": ["indoor background"],
        "detected_features": [
            {"name": "ear shape", "category": "shape",
             "feature_type": "intrinsic", "gradcam_attention": "high"},
            {"name": "wooden floor", "category": "context",
             "feature_type": "contextual", "gradcam_attention": "low"},
        ],
        "knowledge_based_features": [],
        "environmental_patterns": [],
        "gradcam_summary": "Focus on head and torso",
        "model_focus": "Focus on head and torso",
        "baseline_results": [],
        "edit_results": [
            {
                "instruction": "Remove the wooden floor",
                "hypothesis": "Model relies on floor",
                "edit_type": "context_removal",
                "target_type": "positive",
                "priority": 5,
                "original_confidence": 0.95,
                "original_image_path": "/fake/img.jpg",
                "generations": [{
                    "seed": 0, "edited_confidence": 0.40,
                    "delta": -0.55, "edited_image_path": "/fake/edited.jpg",
                    "edit_verified": True, "verification_confidence": 1.0,
                    "verification_description": "",
                    "gradcam_image_path": "", "gradcam_diff_path": "",
                }],
                "mean_edited_confidence": 0.40,
                "mean_delta": -0.55, "std_delta": 0.0,
                "min_delta": -0.55, "max_delta": -0.55,
                "confirmed": True, "confirmation_count": 1,
                "p_value": 0.01, "cohens_d": 2.5, "effect_size": "large",
                "statistically_significant": True,
                "practically_significant": True,
                "validation_method": "statistical",
                "feature_type": "contextual", "feature_name": "wooden floor",
                "source_class": "", "likely_failed": False, "tautological": False,
                "original_gradcam_path": "",
            }
        ],
        "confirmed_hypotheses": [
            {
                "instruction": "Remove the wooden floor",
                "hypothesis": "Model relies on floor",
                "edit_type": "context_removal",
                "target_type": "positive",
                "feature_type": "contextual", "feature_name": "wooden floor",
                "mean_delta": -0.55, "std_delta": 0.0,
                "min_delta": -0.55, "max_delta": -0.55,
                "confirmed": True, "confirmation_count": 1,
                "original_confidence": 0.95,
                "mean_edited_confidence": 0.40,
                "original_image_path": "/fake/img.jpg",
                "original_gradcam_path": "",
                "generations": [{
                    "seed": 0, "edited_confidence": 0.40,
                    "delta": -0.55, "edited_image_path": "/fake/edited.jpg",
                    "edit_verified": True, "verification_confidence": 1.0,
                    "verification_description": "",
                    "gradcam_image_path": "", "gradcam_diff_path": "",
                }],
                "p_value": 0.01, "cohens_d": 2.5, "effect_size": "large",
                "statistically_significant": True,
                "practically_significant": True,
                "validation_method": "statistical",
                "likely_failed": False, "tautological": False,
                "source_class": "", "priority": 5,
            },
        ],
        "iterations_completed": 1,
        "vlm_insights": [],
        "confirmed_shortcuts": ["wooden floor"],
        "feature_importance": [],
        "robustness_score": 3,
        "risk_level": "HIGH",
        "vulnerabilities": ["Relies on background"],
        "recommendations": ["Train with diverse backgrounds"],
        "final_summary": "Model relies on background context.",
        "summary": {
            "total_edits": 1, "total_generations": 1,
            "confirmed_count": 1, "confirmation_rate": 1.0,
            "iterations": 1, "robustness_score": 3, "risk_level": "HIGH",
        },
    }


class TestHtmlReportContent:
    def test_renders_class_name(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([_make_sample_result()])
        html = (tmp_path / "report.html").read_text()
        assert "tabby cat" in html

    def test_renders_spurious_shortcut(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([_make_sample_result()])
        html = (tmp_path / "report.html").read_text()
        assert "SHORTCUT" in html

    def test_renders_feature_tags(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([_make_sample_result()])
        html = (tmp_path / "report.html").read_text()
        assert "indoor background" in html

    def test_renders_risk_level(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([_make_sample_result()])
        html = (tmp_path / "report.html").read_text()
        assert "HIGH" in html

    def test_no_error_messages_in_html(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_html([_make_sample_result()])
        html = (tmp_path / "report.html").read_text()
        assert "Traceback" not in html
        assert "Exception" not in html

    def test_image_paths_are_relative(self, tmp_path):
        result = _make_sample_result()
        img_dir = tmp_path / "images" / "tabby_cat"
        img_dir.mkdir(parents=True)
        result["edit_results"][0]["original_image_path"] = str(img_dir / "img.jpg")
        result["edit_results"][0]["generations"][0]["edited_image_path"] = str(img_dir / "edited.jpg")
        result["confirmed_hypotheses"][0]["original_image_path"] = str(img_dir / "img.jpg")
        result["confirmed_hypotheses"][0]["generations"][0]["edited_image_path"] = str(img_dir / "edited.jpg")
        reporter = Reporter(tmp_path)
        reporter.generate_html([result])
        html = (tmp_path / "report.html").read_text()
        assert str(tmp_path) not in html
        assert "images/tabby_cat/" in html


class TestMarkdownReportContent:
    def test_renders_class_name(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_markdown([_make_sample_result()])
        md = (tmp_path / "report.md").read_text()
        assert "tabby cat" in md

    def test_renders_confirmed_count(self, tmp_path):
        reporter = Reporter(tmp_path)
        reporter.generate_markdown([_make_sample_result()])
        md = (tmp_path / "report.md").read_text()
        assert "1" in md
