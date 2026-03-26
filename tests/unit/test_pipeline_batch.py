"""Tests for batch pipeline: dataclasses, phase methods, and orchestration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.config import Config
from src.pipeline import (
    AnalysisPipeline, BatchClassState, ClassAnalysisResult, EditContext,
    EditInput, EditResult, GenerationResult, ImageSet, PendingGeneration,
)
from tests.conftest import make_edit_result, make_generation, make_image, make_instruction


@pytest.fixture
def config():
    return Config(
        device="cpu", low_vram=True, output_dir=Path("/tmp/test_batch"),
        resume=False, use_statistical_validation=True, generations_per_edit=2,
    )


@pytest.fixture
def pipeline(config):
    with patch("src.pipeline.ModelManager"):
        return AnalysisPipeline(config)


@pytest.fixture
def sample_state():
    result = ClassAnalysisResult(class_name="tabby cat")
    return BatchClassState(
        class_name="tabby cat",
        class_dir=Path("/tmp/test_batch/tabby_cat"),
        result=result,
    )


@pytest.fixture
def sample_state_with_images(sample_state):
    images = ImageSet()
    images.inspect = [(make_image(), "tabby cat")]
    images.edit = [(make_image(), "tabby cat")]
    images.annotated_inspect = [(make_image(), MagicMock(), "tabby cat")]
    sample_state.images = images
    return sample_state


# =========================================================================
# PendingGeneration dataclass
# =========================================================================

class TestPendingGeneration:
    def test_fields(self):
        pg = PendingGeneration(
            edit_input_idx=2, seed=42,
            edited_image_path="/tmp/edit.jpg", gen_idx=0,
        )
        assert pg.edit_input_idx == 2
        assert pg.seed == 42
        assert pg.edited_image_path == "/tmp/edit.jpg"
        assert pg.gen_idx == 0


# =========================================================================
# BatchClassState dataclass
# =========================================================================

class TestBatchClassState:
    def test_defaults(self):
        state = BatchClassState(
            class_name="cat",
            class_dir=Path("/tmp/cat"),
            result=ClassAnalysisResult(class_name="cat"),
        )
        assert state.images is None
        assert state.edit_inputs == []
        assert state.pending == []
        assert state.failed is False

    def test_failed_flag(self, sample_state):
        assert not sample_state.failed
        sample_state.failed = True
        assert sample_state.failed


# =========================================================================
# _active_states
# =========================================================================

class TestActiveStates:
    def test_filters_failed(self, pipeline):
        s1 = BatchClassState("a", Path("/a"), ClassAnalysisResult("a"))
        s2 = BatchClassState("b", Path("/b"), ClassAnalysisResult("b"), failed=True)
        s3 = BatchClassState("c", Path("/c"), ClassAnalysisResult("c"))
        active = pipeline._active_states([s1, s2, s3])
        assert len(active) == 2
        assert s2 not in active

    def test_all_active(self, pipeline):
        s1 = BatchClassState("a", Path("/a"), ClassAnalysisResult("a"))
        assert pipeline._active_states([s1]) == [s1]

    def test_all_failed(self, pipeline):
        s1 = BatchClassState("a", Path("/a"), ClassAnalysisResult("a"), failed=True)
        assert pipeline._active_states([s1]) == []


# =========================================================================
# _edit_prefix
# =========================================================================

class TestEditPrefix:
    def test_positive_target(self, pipeline):
        instr = make_instruction(target="positive")
        inp = EditInput(make_image(), instr, 0.9)
        assert pipeline._edit_prefix(inp) == "pos_0"

    def test_negative_target(self, pipeline):
        instr = make_instruction(target="negative")
        inp = EditInput(make_image(), instr, 0.5)
        assert pipeline._edit_prefix(inp) == "neg_0"

    def test_image_index_used(self, pipeline):
        instr = make_instruction(target="positive")
        instr.image_index = 3
        inp = EditInput(make_image(), instr, 0.9)
        assert pipeline._edit_prefix(inp) == "pos_3"


# =========================================================================
# _collect_batch_results
# =========================================================================

class TestCollectBatchResults:
    def test_returns_all_results(self, pipeline):
        s1 = BatchClassState("a", Path("/a"), ClassAnalysisResult("a"))
        s2 = BatchClassState("b", Path("/b"), ClassAnalysisResult("b"), failed=True)
        results = pipeline._collect_batch_results([s1, s2])
        assert len(results) == 2
        assert results[0].class_name == "a"
        assert results[1].class_name == "b"


# =========================================================================
# _init_batch_states
# =========================================================================

class TestInitBatchStates:
    def test_creates_states_for_new_classes(self, pipeline, tmp_path):
        pipeline.cfg = Config(
            device="cpu", output_dir=tmp_path, resume=False,
        )
        states, cached = pipeline._init_batch_states(["cat", "dog"])
        assert len(states) == 2
        assert len(cached) == 0
        assert states[0].class_name == "cat"
        assert states[1].class_name == "dog"

    def test_loads_from_checkpoint(self, pipeline, tmp_path):
        pipeline.cfg = Config(
            device="cpu", output_dir=tmp_path, resume=True,
        )
        cached_result = ClassAnalysisResult(class_name="cat")
        with patch.object(pipeline, "_try_load_checkpoint", return_value=cached_result):
            states, cached = pipeline._init_batch_states(["cat"])
        assert len(states) == 0
        assert len(cached) == 1
        assert cached[0].class_name == "cat"


# =========================================================================
# _generate_and_save_variant
# =========================================================================

class TestGenerateAndSaveVariant:
    def test_returns_pending_generation(self, pipeline, tmp_path):
        pipeline.models._editor = MagicMock()
        edited_img = make_image()
        pipeline.models._editor.edit.return_value = edited_img
        instr = make_instruction()
        inp = EditInput(make_image(), instr, 0.9)
        ctx = EditContext("cat", tmp_path, "pos_0", edit_idx=0, iteration=0, base_seed=100)
        result = pipeline._generate_and_save_variant(inp, ctx, gen_idx=0)
        assert result is not None
        assert isinstance(result, PendingGeneration)
        assert result.seed == 100
        assert result.gen_idx == 0
        assert result.edit_input_idx == 0

    def test_returns_none_on_exception(self, pipeline, tmp_path):
        pipeline.models._editor = MagicMock()
        pipeline.models._editor.edit.side_effect = RuntimeError("fail")
        instr = make_instruction()
        inp = EditInput(make_image(), instr, 0.9)
        ctx = EditContext("cat", tmp_path, "pos_0", edit_idx=0, iteration=0, base_seed=0)
        result = pipeline._generate_and_save_variant(inp, ctx, gen_idx=0)
        assert result is None


# =========================================================================
# _classify_saved_variant
# =========================================================================

class TestClassifySavedVariant:
    def test_returns_generation_result(self, pipeline, tmp_path):
        img = make_image()
        img_path = tmp_path / "edited.jpg"
        img.save(img_path)
        pending = PendingGeneration(
            edit_input_idx=0, seed=42,
            edited_image_path=str(img_path), gen_idx=0,
        )
        instr = make_instruction()
        inp = EditInput(make_image(), instr, 0.9)
        ctx = EditContext("cat", tmp_path, "pos_0", edit_idx=0, iteration=0)
        with patch.object(pipeline, "_classify_edited_image", return_value=(0.7, "", "")):
            result = pipeline._classify_saved_variant(pending, inp, ctx)
        assert result is not None
        assert isinstance(result, GenerationResult)
        assert result.seed == 42
        assert result.edited_confidence == 0.7
        assert result.delta == round(0.7 - 0.9, 4)

    def test_returns_none_on_exception(self, pipeline, tmp_path):
        pending = PendingGeneration(
            edit_input_idx=0, seed=42,
            edited_image_path="/nonexistent/path.jpg", gen_idx=0,
        )
        instr = make_instruction()
        inp = EditInput(make_image(), instr, 0.9)
        ctx = EditContext("cat", tmp_path, "pos_0", edit_idx=0, iteration=0)
        result = pipeline._classify_saved_variant(pending, inp, ctx)
        assert result is None


# =========================================================================
# Inner methods (model-management-free)
# =========================================================================

class TestRunBaseline:
    def test_combines_inspect_and_negative_records(self, pipeline):
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline.models.sampler.return_value.sample_from_classes.return_value = []
        pipeline._status = MagicMock()
        with patch.object(pipeline, "_classify_inspect_images", return_value=[
            {"type": "positive", "top_k": [("dog", 0.3)]}
        ]):
            with patch.object(pipeline, "_classify_negative_images", return_value=[{"type": "negative"}]):
                with patch.object(pipeline, "_find_confusing_from_baseline", return_value=["dog"]):
                    result = pipeline._run_baseline("cat", ImageSet(), Path("/tmp"))
        assert len(result) == 2
        assert result[0]["type"] == "positive"
        assert result[1]["type"] == "negative"


class TestGenerateEditInstructionsInner:
    def test_calls_generation_methods(self, pipeline):
        pipeline.models._vlm = MagicMock()
        result = ClassAnalysisResult(class_name="cat")
        result.detected_features = [
            {"name": "ears", "category": "part", "feature_type": "intrinsic", "gradcam_attention": "high"}
        ]
        images = ImageSet()
        images.annotated_inspect = [(make_image(), MagicMock(), "cat")]
        images.edit = [(make_image(), "cat")]
        images.annotated_negatives = []
        with patch.object(pipeline, "_generate_positive_edits", return_value=[]):
            with patch.object(pipeline, "_generate_negative_edits", return_value=[]):
                inputs = pipeline._generate_edit_instructions_inner("cat", images, result)
        assert inputs == []


class TestRunFinalAnalysisInner:
    def test_returns_none_on_failure(self, pipeline):
        pipeline.models._vlm = MagicMock()
        pipeline.models._vlm.final_analysis.side_effect = RuntimeError("fail")
        result = ClassAnalysisResult(class_name="cat")
        result.edit_results = [make_edit_result()]
        result.detected_features = []
        final = pipeline._run_final_analysis_inner("cat", result, make_image())
        assert final is None


class TestClassifyFeaturesInner:
    def test_assigns_feature_types(self, pipeline):
        pipeline.models._vlm = MagicMock()
        pipeline.models._vlm.classify_features.return_value = [
            {"feature_type": "contextual", "feature_name": "background"},
        ]
        er = make_edit_result()
        pipeline._classify_features_inner("cat", [er])
        assert er.feature_type == "contextual"
        assert er.feature_name == "background"

    def test_handles_fewer_classifications_than_results(self, pipeline):
        pipeline.models._vlm = MagicMock()
        pipeline.models._vlm.classify_features.return_value = []
        er = make_edit_result()
        pipeline._classify_features_inner("cat", [er])
        # Should not crash; feature_type remains from make_edit_result default


# =========================================================================
# _generate_all_variants
# =========================================================================

class TestGenerateAllVariants:
    def test_generates_for_each_input_and_gen(self, pipeline, tmp_path):
        state = BatchClassState("cat", tmp_path, ClassAnalysisResult("cat"))
        instr1 = make_instruction(edit="Remove ears")
        instr2 = make_instruction(edit="Remove tail")
        state.edit_inputs = [
            EditInput(make_image(), instr1, 0.9),
            EditInput(make_image(), instr2, 0.8),
        ]
        pending = PendingGeneration(0, 0, "/tmp/e.jpg", 0)
        with patch.object(pipeline, "_generate_and_save_variant", return_value=pending):
            result = pipeline._generate_all_variants(state)
        # 2 inputs x 2 generations_per_edit = 4 calls
        assert len(result) == 4

    def test_skips_failed_generations(self, pipeline, tmp_path):
        state = BatchClassState("cat", tmp_path, ClassAnalysisResult("cat"))
        state.edit_inputs = [EditInput(make_image(), make_instruction(), 0.9)]
        with patch.object(pipeline, "_generate_and_save_variant", return_value=None):
            result = pipeline._generate_all_variants(state)
        assert result == []


