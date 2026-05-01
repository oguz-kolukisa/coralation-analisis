"""Tests for PipelineV2 checkpoint resume + batch processing."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.pipeline_v2 import ClassResultV2, PipelineV2, PreCompletedResult


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        device="cpu",
        low_vram=False,
        output_dir=tmp_path,
        skip_probes=True,
        resume=True,
    )


@pytest.fixture
def pipeline(cfg, monkeypatch):
    monkeypatch.setattr("src.pipeline_v2.ModelManager", MagicMock)
    return PipelineV2(cfg)


def _write_complete_checkpoint(cfg: Config, class_name: str) -> dict:
    """Write a checkpoint that looks fully completed (has summary)."""
    ckpt_dir = cfg.model_checkpoint_dir(cfg.classifier_model)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "class_name": class_name,
        "summary": {"total_concepts": 3, "total_edits": 0, "total_spurious": 0},
    }
    path = ckpt_dir / f"{class_name.replace(' ', '_')}.json"
    path.write_text(json.dumps(data))
    return data


def _write_partial_checkpoint(cfg: Config, class_name: str) -> None:
    """Write a checkpoint missing the summary field (incomplete)."""
    ckpt_dir = cfg.model_checkpoint_dir(cfg.classifier_model)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{class_name.replace(' ', '_')}.json"
    path.write_text(json.dumps({"class_name": class_name}))


class TestLoadCompletedCheckpoints:
    def test_returns_empty_when_no_checkpoint_dir(self, pipeline):
        completed = pipeline._load_completed_checkpoints(["tench", "goldfish"])
        assert completed == {}

    def test_loads_only_complete_checkpoints(self, pipeline):
        _write_complete_checkpoint(pipeline.cfg, "tench")
        _write_partial_checkpoint(pipeline.cfg, "goldfish")
        completed = pipeline._load_completed_checkpoints(["tench", "goldfish"])
        assert "tench" in completed
        assert "goldfish" not in completed

    def test_skips_corrupt_checkpoint(self, pipeline):
        ckpt_dir = pipeline.cfg.model_checkpoint_dir(pipeline.cfg.classifier_model)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "tench.json").write_text("not valid json")
        completed = pipeline._load_completed_checkpoints(["tench"])
        assert completed == {}

    def test_only_loads_requested_classes(self, pipeline):
        _write_complete_checkpoint(pipeline.cfg, "tench")
        _write_complete_checkpoint(pipeline.cfg, "shark")
        completed = pipeline._load_completed_checkpoints(["tench"])
        assert list(completed.keys()) == ["tench"]


class TestIsCompleteCheckpoint:
    def test_true_when_summary_has_total_concepts(self, pipeline):
        assert pipeline._is_complete_checkpoint(
            {"summary": {"total_concepts": 1}}
        ) is True

    def test_false_when_no_summary(self, pipeline):
        assert pipeline._is_complete_checkpoint({"class_name": "x"}) is False

    def test_false_when_summary_lacks_total_concepts(self, pipeline):
        assert pipeline._is_complete_checkpoint({"summary": {}}) is False


class TestPreCompletedResult:
    def test_to_dict_returns_underlying_data(self):
        data = {"class_name": "tench", "summary": {"total_concepts": 2}}
        wrapped = PreCompletedResult(class_name="tench", data=data)
        assert wrapped.to_dict() == data


class TestRunResumeFlow:
    def test_skips_completed_classes(self, pipeline, monkeypatch):
        _write_complete_checkpoint(pipeline.cfg, "tench")
        # Mock per-batch + final phases as no-ops
        for _, m in PipelineV2._PHASES_PER_BATCH:
            setattr(pipeline, m, lambda _r: None)
        for _, m in PipelineV2._PHASES_FINAL:
            setattr(pipeline, m, lambda _r: None)
        pipeline._save_checkpoint = lambda _r: None
        results = pipeline.run(["tench"])
        # Resumed result is a PreCompletedResult
        assert len(results) == 1
        assert isinstance(results[0], PreCompletedResult)

    def test_processes_only_todo_classes(self, pipeline):
        _write_complete_checkpoint(pipeline.cfg, "tench")
        seen_in_classify: list[list[str]] = []

        def fake_classify(rs):
            seen_in_classify.append([r.class_name for r in rs])
        pipeline._phase_classify = fake_classify
        for _, m in PipelineV2._PHASES_PER_BATCH[1:]:
            setattr(pipeline, m, lambda _r: None)
        for _, m in PipelineV2._PHASES_FINAL:
            setattr(pipeline, m, lambda _r: None)
        pipeline._save_checkpoint = lambda _r: None

        pipeline.run(["tench", "goldfish"])
        # Only goldfish should be passed through per-batch phases
        assert seen_in_classify == [["goldfish"]]

    def test_no_resume_reprocesses_everything(self, pipeline):
        _write_complete_checkpoint(pipeline.cfg, "tench")
        pipeline.cfg.resume = False
        seen: list[str] = []
        pipeline._phase_classify = lambda rs: seen.extend(r.class_name for r in rs)
        for _, m in PipelineV2._PHASES_PER_BATCH[1:]:
            setattr(pipeline, m, lambda _r: None)
        for _, m in PipelineV2._PHASES_FINAL:
            setattr(pipeline, m, lambda _r: None)
        pipeline._save_checkpoint = lambda _r: None
        pipeline.run(["tench"])
        assert seen == ["tench"]


class TestBatching:
    def test_batches_split_correctly(self, pipeline):
        pipeline.cfg.batch_size = 2
        seen_batches: list[list[str]] = []
        pipeline._phase_classify = lambda rs: seen_batches.append(
            [r.class_name for r in rs])
        for _, m in PipelineV2._PHASES_PER_BATCH[1:]:
            setattr(pipeline, m, lambda _r: None)
        for _, m in PipelineV2._PHASES_FINAL:
            setattr(pipeline, m, lambda _r: None)
        pipeline._save_checkpoint = lambda _r: None
        pipeline.run(["a", "b", "c", "d", "e"])
        assert seen_batches == [["a", "b"], ["c", "d"], ["e"]]

    def test_batch_size_zero_means_single_batch(self, pipeline):
        pipeline.cfg.batch_size = 0
        seen_batches: list[list[str]] = []
        pipeline._phase_classify = lambda rs: seen_batches.append(
            [r.class_name for r in rs])
        for _, m in PipelineV2._PHASES_PER_BATCH[1:]:
            setattr(pipeline, m, lambda _r: None)
        for _, m in PipelineV2._PHASES_FINAL:
            setattr(pipeline, m, lambda _r: None)
        pipeline._save_checkpoint = lambda _r: None
        pipeline.run(["a", "b", "c"])
        assert seen_batches == [["a", "b", "c"]]

    def test_checkpoint_saved_per_batch(self, pipeline):
        pipeline.cfg.batch_size = 2
        saved: list[str] = []
        pipeline._save_checkpoint = lambda r: saved.append(r.class_name)
        for _, m in PipelineV2._PHASES_PER_BATCH:
            setattr(pipeline, m, lambda _r: None)
        for _, m in PipelineV2._PHASES_FINAL:
            setattr(pipeline, m, lambda _r: None)
        pipeline.run(["a", "b", "c"])
        # Each new class checkpointed (3 saves total)
        assert saved == ["a", "b", "c"]
