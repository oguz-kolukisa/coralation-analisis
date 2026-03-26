"""Tests for pipeline.py — phase orchestration and batch processing methods."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from src.config import Config
from src.pipeline import (
    AnalysisPipeline, BatchClassState, ClassAnalysisResult,
    EditInput, EditResult, GenerationResult, ImageSet,
    PendingGeneration, DiscoveredFeatures, EditContext,
)
from src.vlm import EditInstruction, FeatureDiscovery, FinalAnalysis
from tests.conftest import make_edit_result, make_generation, make_instruction


@pytest.fixture
def cfg(tmp_path):
    return Config(
        device="cpu", low_vram=False, output_dir=tmp_path,
        resume=False, use_statistical_validation=False,
    )


@pytest.fixture
def pipeline(cfg):
    with patch.object(AnalysisPipeline, "__init__", lambda self, *a, **kw: None):
        p = AnalysisPipeline.__new__(AnalysisPipeline)
        p.cfg = cfg
        p.models = MagicMock()
        p._pbar = None
        from src.analysis.statistics import StatisticalValidator
        p.stat_validator = StatisticalValidator()
        return p


def _make_state(class_name="cat", tmp_path=None):
    class_dir = tmp_path / class_name if tmp_path else Path("/tmp") / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    result = ClassAnalysisResult(class_name=class_name)
    state = BatchClassState(class_name=class_name, class_dir=class_dir, result=result)
    return state


def _make_edit_input(edit="Remove ears", target="positive", edit_type="feature_removal"):
    instr = EditInstruction(edit=edit, hypothesis="h", type=edit_type, target=target, priority=3, image_index=0)
    img = Image.new("RGB", (64, 64))
    return EditInput(instruction=instr, image=img, original_confidence=0.9)


# ============================================================================
# _active_states
# ============================================================================

class TestActiveStates:
    def test_filters_failed(self, pipeline, tmp_path):
        s1 = _make_state("a", tmp_path)
        s2 = _make_state("b", tmp_path)
        s2.failed = True
        assert len(pipeline._active_states([s1, s2])) == 1

    def test_all_active(self, pipeline, tmp_path):
        s1 = _make_state("a", tmp_path)
        s2 = _make_state("b", tmp_path)
        assert len(pipeline._active_states([s1, s2])) == 2

    def test_empty(self, pipeline):
        assert pipeline._active_states([]) == []


# ============================================================================
# _collect_batch_results
# ============================================================================

class TestCollectBatchResults:
    def test_extracts_results(self, pipeline, tmp_path):
        s1 = _make_state("a", tmp_path)
        s2 = _make_state("b", tmp_path)
        results = pipeline._collect_batch_results([s1, s2])
        assert len(results) == 2
        assert results[0].class_name == "a"


# ============================================================================
# _init_batch_states
# ============================================================================

class TestInitBatchStates:
    def test_creates_states_no_resume(self, pipeline, tmp_path):
        pipeline.cfg.resume = False
        states, cached = pipeline._init_batch_states(["cat", "dog"])
        assert len(states) == 2
        assert cached == []

    def test_loads_cached_on_resume(self, pipeline, tmp_path):
        pipeline.cfg.resume = True
        cached_result = ClassAnalysisResult(class_name="cat")
        pipeline._load_checkpoint = MagicMock(return_value=cached_result)
        pipeline._try_load_checkpoint = lambda cn: cached_result if cn == "cat" else None

        states, cached = pipeline._init_batch_states(["cat", "dog"])
        assert len(cached) == 1
        assert len(states) == 1


# ============================================================================
# _try_load_checkpoint
# ============================================================================

class TestTryLoadCheckpoint:
    def test_returns_none_when_resume_disabled(self, pipeline):
        pipeline.cfg.resume = False
        assert pipeline._try_load_checkpoint("cat") is None

    def test_returns_cached_when_found(self, pipeline):
        pipeline.cfg.resume = True
        cached = ClassAnalysisResult(class_name="cat")
        pipeline._load_checkpoint = MagicMock(return_value=cached)
        pipeline._status = MagicMock()
        result = pipeline._try_load_checkpoint("cat")
        assert result is cached

    def test_returns_none_when_not_found(self, pipeline):
        pipeline.cfg.resume = True
        pipeline._load_checkpoint = MagicMock(return_value=None)
        assert pipeline._try_load_checkpoint("cat") is None


# ============================================================================
# Persistence helpers
# ============================================================================

class TestMakeClassDir:
    def test_creates_dir(self, pipeline, tmp_path):
        path = pipeline._make_class_dir("tabby cat")
        assert path.exists()
        assert "tabby_cat" in str(path)


class TestCheckpointPath:
    def test_returns_path(self, pipeline, tmp_path):
        path = pipeline._checkpoint_path("cat")
        assert str(path).endswith("cat/analysis.json")


class TestSaveCheckpoint:
    def test_saves_json(self, pipeline, tmp_path):
        result = ClassAnalysisResult(class_name="cat")
        class_dir = tmp_path / "cat"
        class_dir.mkdir()
        pipeline._save_checkpoint(result, class_dir)
        assert (class_dir / "analysis.json").exists()


class TestLoadCheckpoint:
    def test_loads_existing(self, pipeline, tmp_path):
        result = ClassAnalysisResult(class_name="cat")
        class_dir = tmp_path / "cat"
        class_dir.mkdir()
        path = class_dir / "analysis.json"
        path.write_text(json.dumps(result.to_dict(), default=str))
        loaded = pipeline._load_checkpoint("cat")
        assert loaded is not None
        assert loaded.class_name == "cat"

    def test_returns_none_when_missing(self, pipeline, tmp_path):
        assert pipeline._load_checkpoint("nonexistent") is None

    def test_returns_none_on_corrupt_json(self, pipeline, tmp_path):
        class_dir = tmp_path / "cat"
        class_dir.mkdir()
        (class_dir / "analysis.json").write_text("not json")
        assert pipeline._load_checkpoint("cat") is None


# ============================================================================
# _status
# ============================================================================

class TestStatus:
    def test_no_pbar_is_noop(self, pipeline):
        pipeline._pbar = None
        pipeline._status("test")  # Should not raise

    def test_with_pbar(self, pipeline):
        pipeline._pbar = MagicMock()
        pipeline._status("test")
        pipeline._pbar.set_postfix_str.assert_called_once()


# ============================================================================
# _log_completion
# ============================================================================

class TestLogCompletion:
    def test_logs_summary(self, pipeline):
        pipeline._status = MagicMock()
        result = ClassAnalysisResult(class_name="cat")
        result.risk_level = "HIGH"
        pipeline._log_completion(result)
        pipeline._status.assert_called_once()


# ============================================================================
# _edit_prefix
# ============================================================================

class TestEditPrefix:
    def test_positive_target(self, pipeline):
        inp = _make_edit_input(target="positive")
        assert pipeline._edit_prefix(inp) == "pos_0"

    def test_negative_target(self, pipeline):
        inp = _make_edit_input(target="negative")
        assert pipeline._edit_prefix(inp) == "neg_0"


# ============================================================================
# _deduplicate_inputs
# ============================================================================

class TestDeduplicateInputs:
    def test_removes_exact_duplicates(self, pipeline):
        inp1 = _make_edit_input("Remove the ears")
        inp2 = _make_edit_input("Remove the ears")
        result = pipeline._deduplicate_inputs([inp1, inp2])
        assert len(result) == 1

    def test_keeps_different_edits(self, pipeline):
        inp1 = _make_edit_input("Remove the ears")
        inp2 = _make_edit_input("Change the background to blue")
        result = pipeline._deduplicate_inputs([inp1, inp2])
        assert len(result) == 2

    def test_empty_list(self, pipeline):
        assert pipeline._deduplicate_inputs([]) == []

    def test_removes_near_duplicates(self, pipeline):
        pipeline.cfg.dedup_similarity_threshold = 0.8
        inp1 = _make_edit_input("Remove the ears completely")
        inp2 = _make_edit_input("Remove the ears entirely")
        result = pipeline._deduplicate_inputs([inp1, inp2])
        assert len(result) == 1


# ============================================================================
# _validate_direction
# ============================================================================

class TestValidateDirection:
    def test_positive_removal_negative_delta(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._validate_direction(instr, -0.15) is True

    def test_positive_removal_small_delta(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._validate_direction(instr, -0.001) is False

    def test_positive_addition_positive_delta(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_addition")
        assert pipeline._validate_direction(instr, 0.15) is True

    def test_positive_other_type(self, pipeline):
        instr = make_instruction(target="positive", edit_type="compound")
        assert pipeline._validate_direction(instr, 0.15) is True
        assert pipeline._validate_direction(instr, -0.15) is True

    def test_negative_target(self, pipeline):
        instr = make_instruction(target="negative", edit_type="feature_addition")
        assert pipeline._validate_direction(instr, 0.15) is True
        assert pipeline._validate_direction(instr, -0.15) is False


# ============================================================================
# _merge_confusing_classes
# ============================================================================

class TestMergeConfusingClasses:
    def test_deduplicates(self, pipeline):
        result = pipeline._merge_confusing_classes(["Dog", "Cat"], ["dog", "Fish"])
        assert len(result) == 3

    def test_classifier_first(self, pipeline):
        result = pipeline._merge_confusing_classes(["Dog"], ["dog"])
        assert result == ["Dog"]

    def test_empty_lists(self, pipeline):
        assert pipeline._merge_confusing_classes([], []) == []


# ============================================================================
# Phase methods (mocked dependencies)
# ============================================================================

class TestPhase1VlmKnowledge:
    def test_populates_knowledge_features(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        pipeline.models._vlm = MagicMock()
        pipeline.models._vlm.generate_knowledge_based_features.return_value = {
            "knowledge_based_features": [{"feature": "grass"}],
        }
        pipeline.models.vlm = MagicMock()
        pipeline.models.sampler.return_value.get_label_names.return_value = ["cat", "dog"]
        pipeline.models.offload_vlm = MagicMock()

        pipeline._phase1_vlm_knowledge([state])
        assert len(state.result.knowledge_based_features) == 1

    def test_handles_exception(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        pipeline.models.vlm = MagicMock()
        pipeline.models._vlm = MagicMock()
        pipeline.models._vlm.generate_knowledge_based_features.side_effect = RuntimeError("fail")
        pipeline.models.sampler.return_value.get_label_names.return_value = []
        pipeline.models.offload_vlm = MagicMock()

        pipeline._phase1_vlm_knowledge([state])  # Should not raise


class TestPhase2ClassifierBaseline:
    def test_marks_failed_when_no_images(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline._sample_images = MagicMock(return_value=ImageSet())

        pipeline._phase2_classifier_baseline([state])
        assert state.failed is True

    def test_runs_baseline(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        img = Image.new("RGB", (64, 64))
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline._sample_images = MagicMock(return_value=ImageSet(inspect=[(img, "cat")]))
        pipeline._run_baseline_inner = MagicMock(return_value=[{"conf": 0.9}])

        pipeline._phase2_classifier_baseline([state])
        assert state.failed is False
        assert len(state.result.baseline_results) == 1


class TestPhase4EditorGenerate:
    def test_generates_variants(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        state.edit_inputs = [_make_edit_input()]
        pending = PendingGeneration(
            edited_image_path="/tmp/e.jpg", edit_input_idx=0,
            gen_idx=0, seed=42,
        )
        pipeline.models.editor = MagicMock()
        pipeline.models.offload_editor = MagicMock()
        pipeline._generate_all_variants_tracked = MagicMock(return_value=[pending])

        pipeline._phase4_editor_generate([state])
        assert len(state.pending) == 1


class TestPhase6VlmFinal:
    def test_calls_finalize(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        pipeline.models.vlm = MagicMock()
        pipeline.models.offload_vlm = MagicMock()
        pipeline._finalize_class_batch = MagicMock()

        pipeline._phase6_vlm_final([state])
        pipeline._finalize_class_batch.assert_called_once_with(state)

    def test_handles_exception(self, pipeline, tmp_path):
        state = _make_state("cat", tmp_path)
        pipeline.models.vlm = MagicMock()
        pipeline.models.offload_vlm = MagicMock()
        pipeline._finalize_class_batch = MagicMock(side_effect=RuntimeError("fail"))

        pipeline._phase6_vlm_final([state])  # Should not raise


# ============================================================================
# _classify_features_inner
# ============================================================================

class TestClassifyFeaturesInner:
    def test_updates_edit_results(self, pipeline):
        pipeline.models._vlm = MagicMock()
        pipeline.models._vlm.classify_features.return_value = [
            {"feature_type": "intrinsic", "feature_name": "ears"},
        ]
        er = make_edit_result()
        pipeline._classify_features_inner("cat", [er])
        assert er.feature_type == "intrinsic"
        assert er.feature_name == "ears"


# ============================================================================
# _sample_images
# ============================================================================

class TestSampleImages:
    def test_returns_empty_when_no_positives(self, pipeline):
        pipeline.models.sampler.return_value.sample_positive.return_value = []
        result = pipeline._sample_images("cat")
        assert result.inspect == []

    def test_returns_images(self, pipeline):
        img = Image.new("RGB", (64, 64))
        pipeline.models.sampler.return_value.sample_positive.return_value = [(img, "cat")] * 5
        pipeline.models.sampler.return_value.sample_from_classes.return_value = [(img, "dog")]
        pipeline._find_confusing_classes = MagicMock(return_value=["dog"])

        result = pipeline._sample_images("cat")
        assert len(result.inspect) > 0


# ============================================================================
# _validate_class_names
# ============================================================================

class TestValidateClassNames:
    def test_keeps_valid(self, pipeline):
        clf = MagicMock()
        clf.is_valid_class.return_value = True
        pipeline.models.classifier.return_value = clf
        result = pipeline._validate_class_names(["cat", "dog"])
        assert result == ["cat", "dog"]

    def test_drops_invalid(self, pipeline):
        clf = MagicMock()
        clf.is_valid_class.side_effect = [True, False]
        pipeline.models.classifier.return_value = clf
        result = pipeline._validate_class_names(["cat", "xyz"])
        assert result == ["cat"]


# ============================================================================
# _collect_confusion_counts
# ============================================================================

class TestCollectConfusionCounts:
    def test_counts_non_target(self, pipeline):
        clf = MagicMock()
        pred = MagicMock()
        pred.top_k = [("cat", 0.8), ("dog", 0.15), ("fish", 0.01)]
        clf.predict.return_value = pred
        pipeline.cfg.confusing_class_min_conf = 0.05

        img = Image.new("RGB", (64, 64))
        counts = pipeline._collect_confusion_counts(clf, "cat", [(img, "cat")])
        assert "dog" in counts
        assert "cat" not in counts
        assert "fish" not in counts  # below threshold
