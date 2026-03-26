"""Smoke test: results flow through to_dict() into Reporter without data loss.

Mocks all AI models — tests only the orchestration/conversion/report path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.pipeline import ClassAnalysisResult
from src.reporter import Reporter
from tests.conftest import make_edit_result


def _make_class_result(class_name: str) -> ClassAnalysisResult:
    """Build a realistic ClassAnalysisResult without touching any model."""
    result = ClassAnalysisResult(class_name=class_name)
    result.edit_results = [
        make_edit_result(
            instruction="Remove background grass",
            confirmed=True,
            delta=-0.30,
            feature_type="contextual",
        ),
        make_edit_result(
            instruction="Change lighting",
            confirmed=False,
            delta=-0.02,
            feature_type="contextual",
        ),
    ]
    result.detected_features = [
        {"name": "grass", "category": "context",
         "feature_type": "contextual", "gradcam_attention": "high"},
    ]
    result.essential_features = ["body shape"]
    result.spurious_features = ["grass"]
    result.finalize()
    return result


class TestResultConversion:
    def test_to_dict_returns_all_fields(self):
        result = _make_class_result("tabby cat")
        d = result.to_dict()
        assert d["class_name"] == "tabby cat"
        assert d["summary"]["total_edits"] == 2
        assert d["summary"]["confirmed_count"] == 1

    def test_to_dict_edit_results_serializable(self):
        """Ensure edit_results inside the dict are plain dicts, not dataclasses."""
        d = _make_class_result("dog").to_dict()
        assert isinstance(d["edit_results"][0], dict)

    def test_multiple_results_all_convert(self):
        names = ["tabby cat", "golden retriever", "sports car"]
        results = [_make_class_result(n) for n in names]
        dicts = [r.to_dict() for r in results]
        assert len(dicts) == 3
        assert all(d["class_name"] == n for d, n in zip(dicts, names))


class TestReportGeneration:
    def test_reporter_produces_nonempty_files(self, tmp_path):
        results = [_make_class_result("tabby cat")]
        dicts = [r.to_dict() for r in results]

        cfg = Config(output_dir=tmp_path)
        reporter = Reporter(tmp_path, config=cfg.model_dump())
        paths = reporter.generate_all(dicts)

        for key in ("html", "markdown", "json"):
            p = paths[key]
            assert p.exists(), f"{key} report not created"
            assert p.stat().st_size > 0, f"{key} report is empty"

    def test_report_contains_class_name(self, tmp_path):
        results = [_make_class_result("golden retriever")]
        dicts = [r.to_dict() for r in results]

        reporter = Reporter(tmp_path, config=Config().model_dump())
        paths = reporter.generate_all(dicts)

        html = paths["html"].read_text()
        md = paths["markdown"].read_text()
        assert "golden retriever" in html
        assert "golden retriever" in md

    def test_report_includes_all_classes(self, tmp_path):
        names = ["tabby cat", "sports car", "daisy"]
        dicts = [_make_class_result(n).to_dict() for n in names]

        reporter = Reporter(tmp_path, config=Config().model_dump())
        paths = reporter.generate_all(dicts)

        html = paths["html"].read_text()
        for name in names:
            assert name in html, f"class '{name}' missing from HTML report"

    def test_empty_results_does_not_crash(self, tmp_path):
        reporter = Reporter(tmp_path, config=Config().model_dump())
        paths = reporter.generate_all([])

        for key in ("html", "markdown", "json"):
            assert paths[key].exists()
