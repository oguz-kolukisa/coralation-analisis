"""Tests for the new PipelineV2 probe phases (7, 8, 9)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.pipeline_v2 import ClassResultV2, PipelineV2


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        device="cpu",
        low_vram=False,
        output_dir=tmp_path,
        skip_probes=False,
        probe_n_per_variant=1,
    )


@pytest.fixture
def pipeline(cfg, monkeypatch):
    # Skip ModelManager construction (which would touch torch / cuda)
    monkeypatch.setattr("src.pipeline_v2.ModelManager", MagicMock)
    return PipelineV2(cfg)


@pytest.fixture
def fake_results():
    r = ClassResultV2(class_name="tench")
    return [r]


# ---------------------------------------------------------------------------
# _PHASES list — ordering and content
# ---------------------------------------------------------------------------

class TestPhasesList:
    def test_has_six_per_batch_and_three_final(self):
        assert len(PipelineV2._PHASES_PER_BATCH) == 6
        assert len(PipelineV2._PHASES_FINAL) == 3

    def test_per_batch_phases_in_order(self):
        names = [name for name, _ in PipelineV2._PHASES_PER_BATCH]
        assert names == [
            "Classifier: baseline", "VLM: discovery",
            "Dedup + edit generation", "Editor: generate images",
            "Classifier: measure", "VLM: verdict",
        ]

    def test_final_phases_in_order(self):
        names = [name for name, _ in PipelineV2._PHASES_FINAL]
        assert names == [
            "Feature catalog", "Probe: generate", "Probe: evaluate",
        ]


# ---------------------------------------------------------------------------
# Phase 7: Feature catalog (always runs, even with --no-probes)
# ---------------------------------------------------------------------------

class TestFeatureCatalogPhase:
    def test_writes_catalog_file(self, pipeline, fake_results):
        with patch("src.feature_extractor.extract_catalog",
                    return_value={"classes": {}}) as mock_extract:
            pipeline._phase_feature_catalog(fake_results)
            mock_extract.assert_called_once()
        assert pipeline.cfg.feature_catalog_path.exists()

    def test_populates_self_catalog(self, pipeline, fake_results):
        catalog = {"classes": {"tench": {}}, "version": "0.3.0"}
        with patch("src.feature_extractor.extract_catalog", return_value=catalog):
            pipeline._phase_feature_catalog(fake_results)
        assert pipeline._catalog == catalog

    def test_catalog_runs_even_when_skip_probes_true(self, pipeline, fake_results):
        pipeline.cfg.skip_probes = True
        with patch("src.feature_extractor.extract_catalog",
                    return_value={"classes": {}}):
            pipeline._phase_feature_catalog(fake_results)
        # Phase 7 ignores skip_probes
        assert pipeline._catalog is not None


# ---------------------------------------------------------------------------
# Phase 8: Probe generation
# ---------------------------------------------------------------------------

class TestProbeGeneratePhase:
    def test_skipped_when_skip_probes(self, pipeline, fake_results):
        pipeline.cfg.skip_probes = True
        pipeline._catalog = {"classes": {}}
        with patch("src.probe_generator.generate_probes") as mock_gen:
            pipeline._phase_probe_generate(fake_results)
            mock_gen.assert_not_called()
        assert pipeline._manifest is None

    def test_skipped_when_no_catalog(self, pipeline, fake_results):
        pipeline._catalog = None
        with patch("src.probe_generator.generate_probes") as mock_gen:
            pipeline._phase_probe_generate(fake_results)
            mock_gen.assert_not_called()

    def test_calls_generate_probes_with_catalog(self, pipeline, fake_results):
        pipeline._catalog = {"classes": {"tench": {"consensus": {}}}}
        fake_manifest = MagicMock()
        with patch("src.probe_generator.generate_probes",
                    return_value=fake_manifest) as mock_gen, \
             patch("src.probe_generator._manifest_to_dict",
                    return_value={"classes": {}}):
            pipeline._phase_probe_generate(fake_results)
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args
        # Catalog passed as first positional arg
        assert kwargs.args[0] == pipeline._catalog

    def test_populates_self_manifest(self, pipeline, fake_results):
        pipeline._catalog = {"classes": {"tench": {"consensus": {}}}}
        with patch("src.probe_generator.generate_probes",
                    return_value=MagicMock()), \
             patch("src.probe_generator._manifest_to_dict",
                    return_value={"classes": {"tench": {}}}):
            pipeline._phase_probe_generate(fake_results)
        assert pipeline._manifest == {"classes": {"tench": {}}}


# ---------------------------------------------------------------------------
# Phase 9: Probe evaluation
# ---------------------------------------------------------------------------

class TestProbeEvaluatePhase:
    def test_skipped_when_skip_probes(self, pipeline, fake_results):
        pipeline.cfg.skip_probes = True
        pipeline._manifest = {"classes": {}}
        with patch("src.probe_evaluator.evaluate_probes") as mock_eval:
            pipeline._phase_probe_evaluate(fake_results)
            mock_eval.assert_not_called()

    def test_skipped_when_no_manifest(self, pipeline, fake_results):
        pipeline._manifest = None
        with patch("src.probe_evaluator.evaluate_probes") as mock_eval:
            pipeline._phase_probe_evaluate(fake_results)
            mock_eval.assert_not_called()

    def test_calls_evaluate_and_writes_report(self, pipeline, fake_results):
        pipeline._manifest = {"classes": {"tench": {"images": []}}}
        with patch("src.probe_evaluator.evaluate_probes",
                    return_value={"resnet50": MagicMock()}) as mock_eval, \
             patch("src.probe_evaluator.write_report") as mock_write, \
             patch("src.probe_reporter.write_probe_report",
                    return_value=Path("/tmp/probe_report.html")) as mock_html:
            pipeline._phase_probe_evaluate(fake_results)
        mock_eval.assert_called_once()
        mock_write.assert_called_once()
        mock_html.assert_called_once()


# ---------------------------------------------------------------------------
# _results_to_analysis helper
# ---------------------------------------------------------------------------

class TestResultsToAnalysis:
    def test_returns_dict_with_results_key(self, pipeline, fake_results):
        out = pipeline._results_to_analysis(fake_results)
        assert "results" in out
        assert len(out["results"]) == 1

    def test_has_version_and_timestamp(self, pipeline, fake_results):
        out = pipeline._results_to_analysis(fake_results)
        assert "version" in out
        assert "generated_at" in out
        assert "UTC" in out["generated_at"]

    def test_results_serializable(self, pipeline, fake_results):
        out = pipeline._results_to_analysis(fake_results)
        # default=str handles any non-JSON types
        json.dumps(out, default=str)


# ---------------------------------------------------------------------------
# End-to-end phase order under run() with mocked phase methods
# ---------------------------------------------------------------------------

class TestRunPhaseOrder:
    def test_all_phases_invoked_in_order(self, pipeline):
        called: list[str] = []
        all_phases = PipelineV2._PHASES_PER_BATCH + PipelineV2._PHASES_FINAL
        for phase_name, method_name in all_phases:
            def make_recorder(name):
                def _record(_results):
                    called.append(name)
                return _record
            setattr(pipeline, method_name, make_recorder(method_name))
        pipeline._save_checkpoint = lambda _r: None
        pipeline.run(["tench"])
        assert called == [m for _, m in all_phases]
