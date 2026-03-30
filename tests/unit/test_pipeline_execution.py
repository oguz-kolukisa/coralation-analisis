"""Tests for pipeline.py — baseline, feature discovery, edit generation, execution, and analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

from src.config import Config
from src.pipeline import (
    AnalysisPipeline, BatchClassState, ClassAnalysisResult, DiscoveredFeatures,
    EditContext, EditInput, EditResult, GenerationResult, ImageSet,
    NegativeSample, PendingGeneration,
)
from src.vlm import EditInstruction, DetectedFeature, FeatureDiscovery, FeatureEditPlan, FinalAnalysis
from tests.conftest import make_edit_result, make_generation, make_instruction


class FakePred(NamedTuple):
    label_name: str = "cat"
    confidence: float = 0.9
    top_k: list = [("cat", 0.9)]
    gradcam_image: Image.Image | None = None
    attention_map: np.ndarray | None = None


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


def _img():
    return Image.new("RGB", (64, 64), "red")


def _edit_input(edit="Remove ears", target="positive", edit_type="feature_removal"):
    instr = EditInstruction(edit=edit, hypothesis="h", type=edit_type, target=target, priority=3, image_index=0)
    return EditInput(instruction=instr, image=_img(), original_confidence=0.9)


# ============================================================================
# _base_record
# ============================================================================

class TestBaseRecord:
    def test_returns_dict(self, pipeline):
        pred = FakePred()
        record = pipeline._base_record("/tmp/img.jpg", "cat", pred, 0.85)
        assert record["image_path"] == "/tmp/img.jpg"
        assert record["true_label"] == "cat"
        assert record["predicted_label"] == "cat"
        assert record["class_confidence"] == 0.85


# ============================================================================
# _save_positive_image
# ============================================================================

class TestSavePositiveImage:
    def test_saves_original(self, pipeline, tmp_path):
        img = _img()
        pred = FakePred(gradcam_image=_img())
        pipeline._save_positive_image(img, pred, tmp_path, 0)
        assert (tmp_path / "pos_0_original.jpg").exists()
        assert (tmp_path / "pos_0_gradcam.jpg").exists()

    def test_no_gradcam(self, pipeline, tmp_path):
        img = _img()
        pred = FakePred(gradcam_image=None)
        pipeline._save_positive_image(img, pred, tmp_path, 0)
        assert (tmp_path / "pos_0_original.jpg").exists()
        assert not (tmp_path / "pos_0_gradcam.jpg").exists()


# ============================================================================
# _classify_inspect_images
# ============================================================================

class TestClassifyInspectImages:
    def test_returns_records(self, pipeline, tmp_path):
        clf = MagicMock()
        clf.predict.return_value = FakePred(gradcam_image=_img())
        clf.get_class_confidence.return_value = 0.85
        pipeline.models.classifier.return_value = clf

        img = _img()
        images = ImageSet(inspect=[(img, "cat")])
        records = pipeline._classify_inspect_images("cat", images, tmp_path)
        assert len(records) == 1
        assert records[0]["type"] == "positive"
        assert len(images.annotated_inspect) == 1


# ============================================================================
# _classify_negative_images
# ============================================================================

class TestClassifyNegativeImages:
    def test_filters_low_confidence(self, pipeline, tmp_path):
        clf = MagicMock()
        clf.predict.return_value = FakePred()
        clf.get_class_confidence.return_value = 0.001  # Below threshold
        pipeline.models.classifier.return_value = clf
        pipeline.cfg.min_negative_confidence = 0.01

        images = ImageSet(negative=[(_img(), "dog")])
        records = pipeline._classify_negative_images("cat", images, tmp_path)
        assert len(records) == 0

    def test_keeps_high_confidence(self, pipeline, tmp_path):
        clf = MagicMock()
        clf.predict.return_value = FakePred()
        clf.get_class_confidence.return_value = 0.15
        pipeline.models.classifier.return_value = clf
        pipeline.cfg.min_negative_confidence = 0.01

        images = ImageSet(negative=[(_img(), "dog")])
        records = pipeline._classify_negative_images("cat", images, tmp_path)
        assert len(records) == 1
        assert records[0]["type"] == "negative"


# ============================================================================
# _run_baseline_inner
# ============================================================================

class TestRunBaselineSplitPhases:
    def test_combines_inspect_and_negative_records(self, pipeline, tmp_path):
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline.models.sampler.return_value.sample_from_classes.return_value = []
        pipeline._status = MagicMock()
        pipeline._classify_inspect_images = MagicMock(
            return_value=[{"type": "positive", "top_k": [("dog", 0.3)]}]
        )
        pipeline._classify_negative_images = MagicMock(return_value=[{"type": "negative"}])
        pipeline._find_confusing_from_baseline = MagicMock(return_value=["dog"])
        images = ImageSet()
        result = pipeline._run_baseline("cat", images, tmp_path)
        assert len(result) == 2


# ============================================================================
# _discover_image_features
# ============================================================================

class TestDiscoverImageFeatures:
    def test_discovers_features(self, pipeline):
        vlm = MagicMock()
        discovery = FeatureDiscovery(
            class_name="cat", gradcam_summary="head focus",
            features=[DetectedFeature("ears", "part", "intrinsic", "top", "high", "pointy")],
            intrinsic_features=["ears"],
        )
        vlm.discover_features.return_value = discovery
        pipeline.models.vlm.return_value = vlm

        pred = FakePred(gradcam_image=_img())
        images = ImageSet(annotated_inspect=[(_img(), pred, "cat")])
        baseline = [{"class_confidence": 0.9}]
        result = pipeline._discover_image_features("cat", images, baseline)
        assert len(result.detected) == 1
        assert "ears" in result.essential

    def test_skips_without_gradcam(self, pipeline):
        vlm = MagicMock()
        pipeline.models.vlm.return_value = vlm

        pred = FakePred(gradcam_image=None)
        images = ImageSet(annotated_inspect=[(_img(), pred, "cat")])
        baseline = [{"class_confidence": 0.9}]
        result = pipeline._discover_image_features("cat", images, baseline)
        assert len(result.detected) == 0
        vlm.discover_features.assert_not_called()


# ============================================================================
# _accumulate_discovery
# ============================================================================

class TestAccumulateDiscovery:
    def test_accumulates(self, pipeline):
        result = DiscoveredFeatures()
        discovery = FeatureDiscovery(
            class_name="cat", gradcam_summary="head focus",
            features=[DetectedFeature("ears", "part", "intrinsic", "top", "high", "pointy")],
            intrinsic_features=["ears"],
        )
        pipeline._accumulate_discovery(discovery, 0, result)
        assert result.gradcam_summary == "head focus"
        assert len(result.detected) == 1
        assert "ears" in result.essential

    def test_separator_between_images(self, pipeline):
        result = DiscoveredFeatures()
        result.gradcam_summary = "first"
        d2 = FeatureDiscovery(class_name="cat", gradcam_summary="second")
        pipeline._accumulate_discovery(d2, 1, result)
        assert " | " in result.gradcam_summary


# ============================================================================
# _deduplicate_features_by_name
# ============================================================================

class TestDeduplicateFeaturesByName:
    def test_removes_duplicates(self, pipeline):
        features = [
            {"name": "ears", "category": "part"},
            {"name": "Ears", "category": "part"},
            {"name": "bg", "category": "ctx"},
        ]
        result = pipeline._deduplicate_features_by_name(features)
        assert len(result) == 2


# ============================================================================
# _to_detected_features
# ============================================================================

class TestToDetectedFeatures:
    def test_converts(self, pipeline):
        dicts = [{"name": "ears", "category": "part", "feature_type": "intrinsic", "gradcam_attention": "high"}]
        result = pipeline._to_detected_features(dicts)
        assert len(result) == 1
        assert isinstance(result[0], DetectedFeature)
        assert result[0].name == "ears"


# ============================================================================
# _wrap_positive_edits
# ============================================================================

class TestWrapPositiveEdits:
    def test_wraps(self, pipeline):
        edits = [FeatureEditPlan("ears", "Remove ears", "removal", "high", "test")]
        result = pipeline._wrap_positive_edits(_img(), edits, 0.9, 0)
        assert len(result) == 1
        assert result[0].instruction.target == "positive"
        assert result[0].instruction.priority == 5

    def test_low_impact_priority(self, pipeline):
        edits = [FeatureEditPlan("bg", "Change bg", "modification", "low", "test")]
        result = pipeline._wrap_positive_edits(_img(), edits, 0.9, 0)
        assert result[0].instruction.priority == 3


# ============================================================================
# _make_edit_context
# ============================================================================

class TestMakeEditContext:
    def test_positive(self, pipeline, tmp_path):
        instr = make_instruction(target="positive")
        ctx = pipeline._make_edit_context("cat", tmp_path, instr)
        assert "pos" in ctx.prefix
        assert ctx.class_name == "cat"

    def test_negative(self, pipeline, tmp_path):
        instr = make_instruction(target="negative")
        ctx = pipeline._make_edit_context("cat", tmp_path, instr)
        assert "neg" in ctx.prefix


# ============================================================================
# _expected_direction
# ============================================================================

class TestExpectedDirection:
    def test_positive_removal(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_removal")
        assert pipeline._expected_direction(instr) == "negative"

    def test_negative_target(self, pipeline):
        instr = make_instruction(target="negative", edit_type="feature_addition")
        assert pipeline._expected_direction(instr) == "positive"

    def test_feature_addition_positive(self, pipeline):
        instr = make_instruction(target="positive", edit_type="feature_addition")
        assert pipeline._expected_direction(instr) == "positive"

    def test_other_type(self, pipeline):
        instr = make_instruction(target="positive", edit_type="compound")
        assert pipeline._expected_direction(instr) == "negative"


# ============================================================================
# _build_edit_result
# ============================================================================

class TestBuildEditResult:
    def test_builds_result(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.2), make_generation(delta=-0.3)]
        result = pipeline._build_edit_result(instr, 0.9, "/tmp/img.jpg", gens)
        assert isinstance(result, EditResult)
        assert result.mean_delta == pytest.approx(-0.25)
        assert result.original_confidence == 0.9

    def test_single_generation_zero_std(self, pipeline):
        instr = make_instruction()
        gens = [make_generation(delta=-0.2)]
        result = pipeline._build_edit_result(instr, 0.9, "/tmp/img.jpg", gens)
        assert result.std_delta == 0.0


# ============================================================================
# _classify_edited_image
# ============================================================================

class TestClassifyEditedImage:
    def test_no_gradcam(self, pipeline, tmp_path):
        pipeline.cfg.compute_edit_gradcam = False
        clf = MagicMock()
        clf.get_class_confidence.return_value = 0.7
        pipeline.models.classifier.return_value = clf

        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        inp = _edit_input()
        conf, gc, diff = pipeline._classify_edited_image(_img(), inp, ctx, 0)
        assert conf == 0.7
        assert gc == ""
        assert diff == ""

    def test_with_gradcam(self, pipeline, tmp_path):
        pipeline.cfg.compute_edit_gradcam = True
        clf = MagicMock()
        clf.predict_with_gradcam.return_value = (0.7, FakePred(gradcam_image=_img(), attention_map=np.zeros((7, 7))))
        pipeline.models.classifier.return_value = clf
        pipeline._save_edit_gradcam = MagicMock(return_value="/tmp/gc.jpg")
        pipeline._save_gradcam_diff = MagicMock(return_value="/tmp/diff.jpg")

        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        inp = _edit_input()
        conf, gc, diff = pipeline._classify_edited_image(_img(), inp, ctx, 0)
        assert conf == 0.7
        assert gc == "/tmp/gc.jpg"


# ============================================================================
# _save_edit_gradcam
# ============================================================================

class TestSaveEditGradcam:
    def test_saves(self, pipeline, tmp_path):
        pred = FakePred(gradcam_image=_img())
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        path = pipeline._save_edit_gradcam(pred, ctx, 0)
        assert path != ""
        assert Path(path).exists()

    def test_no_gradcam_returns_empty(self, pipeline, tmp_path):
        pred = FakePred(gradcam_image=None)
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        assert pipeline._save_edit_gradcam(pred, ctx, 0) == ""


# ============================================================================
# _save_gradcam_diff
# ============================================================================

class TestSaveGradcamDiff:
    def test_saves(self, pipeline, tmp_path):
        inp = _edit_input()
        inp.original_attention_map = np.zeros((7, 7))
        pred = FakePred(attention_map=np.ones((7, 7)))
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        path = pipeline._save_gradcam_diff(inp, pred, _img(), ctx, 0)
        assert path != ""

    def test_no_original_map(self, pipeline, tmp_path):
        inp = _edit_input()
        inp.original_attention_map = None
        pred = FakePred(attention_map=np.ones((7, 7)))
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        assert pipeline._save_gradcam_diff(inp, pred, _img(), ctx, 0) == ""

    def test_no_edited_map(self, pipeline, tmp_path):
        inp = _edit_input()
        inp.original_attention_map = np.zeros((7, 7))
        pred = FakePred(attention_map=None)
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        assert pipeline._save_gradcam_diff(inp, pred, _img(), ctx, 0) == ""


# ============================================================================
# _save_original_gradcam
# ============================================================================

class TestSaveOriginalGradcam:
    def test_disabled(self, pipeline, tmp_path):
        pipeline.cfg.compute_edit_gradcam = False
        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        assert pipeline._save_original_gradcam(inp, ctx) == ""

    def test_no_attention_map(self, pipeline, tmp_path):
        pipeline.cfg.compute_edit_gradcam = True
        inp = _edit_input()
        inp.original_attention_map = None
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        assert pipeline._save_original_gradcam(inp, ctx) == ""


# ============================================================================
# _test_single_edit
# ============================================================================

class TestTestSingleEdit:
    def test_returns_none_when_no_generations(self, pipeline, tmp_path):
        pipeline._save_original_gradcam = MagicMock(return_value="")
        pipeline._generate_edit_variants = MagicMock(return_value=[])

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        assert pipeline._test_single_edit(inp, ctx) is None

    def test_returns_result(self, pipeline, tmp_path):
        pipeline._save_original_gradcam = MagicMock(return_value="")
        pipeline._generate_edit_variants = MagicMock(return_value=[make_generation(delta=-0.2)])
        pipeline._build_edit_result = MagicMock(return_value=make_edit_result())

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        result = pipeline._test_single_edit(inp, ctx)
        assert result is not None


# ============================================================================
# _generate_one_variant
# ============================================================================

class TestGenerateOneVariant:
    def test_success(self, pipeline, tmp_path):
        editor = MagicMock()
        editor.edit.return_value = _img()
        pipeline.models.editor.return_value = editor
        pipeline._classify_edited_image = MagicMock(return_value=(0.7, "", ""))

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        ctx.base_seed = 0
        result = pipeline._generate_one_variant(inp, ctx, 0)
        assert result is not None
        assert isinstance(result, GenerationResult)

    def test_exception_returns_none(self, pipeline, tmp_path):
        editor = MagicMock()
        editor.edit.side_effect = RuntimeError("fail")
        pipeline.models.editor.return_value = editor

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        ctx.base_seed = 0
        result = pipeline._generate_one_variant(inp, ctx, 0)
        assert result is None


# ============================================================================
# _generate_and_save_variant
# ============================================================================

class TestGenerateAndSaveVariant:
    def test_success(self, pipeline, tmp_path):
        pipeline.models._editor = MagicMock()
        pipeline.models._editor.edit.return_value = _img()

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        ctx.base_seed = 0
        result = pipeline._generate_and_save_variant(inp, ctx, 0)
        assert result is not None
        assert isinstance(result, PendingGeneration)

    def test_exception_returns_none(self, pipeline, tmp_path):
        pipeline.models._editor = MagicMock()
        pipeline.models._editor.edit.side_effect = RuntimeError("fail")

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        ctx.base_seed = 0
        result = pipeline._generate_and_save_variant(inp, ctx, 0)
        assert result is None


# ============================================================================
# _classify_saved_variant
# ============================================================================

class TestClassifySavedVariant:
    def test_success(self, pipeline, tmp_path):
        img = _img()
        path = str(tmp_path / "test.jpg")
        img.save(path)

        pipeline._classify_edited_image = MagicMock(return_value=(0.7, "", ""))
        pending = PendingGeneration(
            edited_image_path=path, edit_input_idx=0, gen_idx=0, seed=42,
        )
        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        result = pipeline._classify_saved_variant(pending, inp, ctx)
        assert result is not None

    def test_failure_returns_none(self, pipeline, tmp_path):
        pending = PendingGeneration("/nonexistent.jpg", 0, 0, 42)
        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        result = pipeline._classify_saved_variant(pending, inp, ctx)
        assert result is None


# ============================================================================
# _attach_original_gradcams
# ============================================================================

class TestAttachOriginalGradcams:
    def test_disabled(self, pipeline):
        pipeline.cfg.compute_edit_gradcam = False
        inp = _edit_input()
        pipeline._attach_original_gradcams([inp], "cat")
        assert not hasattr(inp, "original_attention_map") or inp.original_attention_map is None

    def test_caches_per_image(self, pipeline):
        pipeline.cfg.compute_edit_gradcam = True
        clf = MagicMock()
        attn = np.zeros((7, 7))
        pred = FakePred(attention_map=attn)
        clf.predict_with_gradcam.return_value = (0.9, pred)
        pipeline.models.classifier.return_value = clf

        inp1 = _edit_input()
        inp2 = EditInput(instruction=inp1.instruction, image=inp1.image, original_confidence=0.9)
        pipeline._attach_original_gradcams([inp1, inp2], "cat")
        # Should only call once since same image object
        assert clf.predict_with_gradcam.call_count == 1


# ============================================================================
# _negative_to_inputs
# ============================================================================

class TestNegativeToInputs:
    def test_returns_inputs(self, pipeline):
        vlm = MagicMock()
        instr = EditInstruction(edit="Add bg", hypothesis="h", type="feature_addition", target="negative", priority=3, image_index=0)
        analysis = MagicMock()
        analysis.edit_instructions = [instr]
        vlm.analyze_negative.return_value = analysis
        pipeline.models.vlm.return_value = vlm

        sample = NegativeSample(_img(), "cat", "dog", 0.3, 0)
        result = pipeline._negative_to_inputs(sample)
        assert len(result) == 1


# ============================================================================
# _analyze_results / _run_final_analysis_inner / _build_analysis_summary
# ============================================================================

class TestAnalyzeResults:
    def test_returns_none_when_no_edits(self, pipeline):
        result = ClassAnalysisResult(class_name="cat")
        images = ImageSet()
        pipeline._classify_features = MagicMock()
        pipeline._run_final_analysis = MagicMock()
        out = pipeline._analyze_results("cat", result, images)
        pipeline._run_final_analysis.assert_not_called()

    def test_runs_analysis(self, pipeline):
        result = ClassAnalysisResult(class_name="cat")
        result.edit_results = [make_edit_result()]
        img = _img()
        pred = FakePred()
        images = ImageSet(
            annotated_inspect=[(img, pred, "cat")],
        )
        pipeline._classify_features = MagicMock()
        final = FinalAnalysis(class_name="cat", robustness_score=8)
        pipeline._run_final_analysis = MagicMock(return_value=final)
        out = pipeline._analyze_results("cat", result, images)
        assert out.robustness_score == 8


# ============================================================================
# _analyze_class
# ============================================================================

class TestAnalyzeClass:
    def test_returns_early_when_no_images(self, pipeline):
        pipeline._discover_knowledge_features = MagicMock(return_value=[])
        pipeline._sample_images = MagicMock(return_value=ImageSet())
        pipeline._status = MagicMock()

        result = pipeline._analyze_class("cat")
        assert result.class_name == "cat"

    def test_runs_all_phases(self, pipeline, tmp_path):
        pipeline._discover_knowledge_features = MagicMock(return_value=[])
        img = _img()
        images = ImageSet(inspect=[(img, "cat")])
        pipeline._sample_images = MagicMock(return_value=images)
        pipeline._run_all_phases = MagicMock()
        pipeline._save_checkpoint = MagicMock()
        pipeline._log_completion = MagicMock()
        pipeline._status = MagicMock()

        result = pipeline._analyze_class("cat")
        pipeline._run_all_phases.assert_called_once()


# ============================================================================
# _run_all_phases
# ============================================================================

class TestRunAllPhases:
    def test_calls_all_phases(self, pipeline, tmp_path):
        pipeline._run_baseline = MagicMock(return_value=[])
        pipeline._discover_image_features = MagicMock(return_value=DiscoveredFeatures())
        pipeline._run_edits = MagicMock(return_value=[])
        pipeline._analyze_results = MagicMock(return_value=None)

        result = ClassAnalysisResult(class_name="cat")
        images = ImageSet()
        pipeline._run_all_phases("cat", images, result, tmp_path)
        pipeline._run_baseline.assert_called_once()


# ============================================================================
# _discover_knowledge_features
# ============================================================================

class TestDiscoverKnowledgeFeatures:
    def test_returns_features(self, pipeline):
        pipeline._status = MagicMock()
        pipeline.models.sampler.return_value.get_label_names.return_value = ["cat"]
        pipeline.models.vlm.return_value.generate_knowledge_based_features.return_value = {
            "knowledge_based_features": [{"feature": "grass"}]
        }
        result = pipeline._discover_knowledge_features("cat")
        assert len(result) == 1

    def test_returns_empty_on_failure(self, pipeline):
        pipeline._status = MagicMock()
        pipeline.models.sampler.return_value.get_label_names.side_effect = RuntimeError("fail")
        result = pipeline._discover_knowledge_features("cat")
        assert result == []


# ============================================================================
# _generate_all_variants
# ============================================================================

class TestGenerateAllVariants:
    def test_generates_variants(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.edit_inputs = [_edit_input()]
        pending = PendingGeneration(
            edited_image_path="/tmp/e.jpg", edit_input_idx=0, gen_idx=0, seed=42,
        )
        pipeline._generate_and_save_variant = MagicMock(return_value=pending)
        pipeline.cfg.generations_per_edit = 2

        result = pipeline._generate_all_variants(state)
        assert len(result) == 2

    def test_skips_none_results(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.edit_inputs = [_edit_input()]
        pipeline._generate_and_save_variant = MagicMock(return_value=None)
        pipeline.cfg.generations_per_edit = 1

        result = pipeline._generate_all_variants(state)
        assert len(result) == 0


# ============================================================================
# _classify_all_variants / _group_classified_generations / _assemble_edit_results
# ============================================================================

class TestClassifyAllVariants:
    def test_end_to_end(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        inp = _edit_input()
        state.edit_inputs = [inp]
        gen = make_generation(delta=-0.2)
        state.pending = [PendingGeneration(edit_input_idx=0, seed=42, edited_image_path="/tmp/e.jpg", gen_idx=0)]

        pipeline._attach_original_gradcams = MagicMock()
        pipeline._classify_saved_variant = MagicMock(return_value=gen)
        pipeline._build_single_batch_result = MagicMock(return_value=make_edit_result())

        result = pipeline._classify_all_variants(state)
        assert len(result) == 1


class TestAssembleEditResults:
    def test_skips_empty_groups(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.edit_inputs = [_edit_input(), _edit_input()]
        grouped = {0: [make_generation()]}  # Only group 0 has generations
        pipeline._build_single_batch_result = MagicMock(return_value=make_edit_result())

        result = pipeline._assemble_edit_results(state, grouped)
        assert len(result) == 1


class TestBuildSingleBatchResult:
    def test_builds(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        inp = _edit_input()
        gens = [make_generation(delta=-0.2)]
        pipeline._save_original_gradcam = MagicMock(return_value="")
        pipeline._build_edit_result = MagicMock(return_value=make_edit_result())

        result = pipeline._build_single_batch_result(state, inp, 0, gens)
        assert isinstance(result, EditResult)


# ============================================================================
# _phase3_vlm_features_edits
# ============================================================================

class TestPhase3VlmFeaturesEdits:
    def test_success(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.images = ImageSet()
        pipeline.models.vlm = MagicMock()
        pipeline.models.offload_vlm = MagicMock()
        pipeline._discover_image_features = MagicMock(return_value=DiscoveredFeatures())
        pipeline._generate_edit_instructions_inner = MagicMock(return_value=[])
        pipeline._deduplicate_inputs = MagicMock(return_value=[])

        pipeline._phase3_vlm_features_edits([state])
        assert state.failed is False

    def test_marks_failed_on_error(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.images = ImageSet()
        pipeline.models.vlm = MagicMock()
        pipeline.models.offload_vlm = MagicMock()
        pipeline._discover_image_features = MagicMock(side_effect=RuntimeError("fail"))

        pipeline._phase3_vlm_features_edits([state])
        assert state.failed is True


# ============================================================================
# _phase5_classifier_measure
# ============================================================================

class TestPhase5ClassifierMeasure:
    def test_success(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline._classify_all_variants_tracked = MagicMock(return_value=[make_edit_result()])

        pipeline._phase5_classifier_measure([state])
        assert len(state.result.edit_results) == 1

    def test_marks_failed_on_error(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline._classify_all_variants_tracked = MagicMock(side_effect=RuntimeError("fail"))

        pipeline._phase5_classifier_measure([state])
        assert state.failed is True


# ============================================================================
# _finalize_class_batch
# ============================================================================

class TestFinalizeClassBatch:
    def test_early_return_no_edits(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.images = ImageSet()
        pipeline._save_checkpoint = MagicMock()

        pipeline._finalize_class_batch(state)
        pipeline._save_checkpoint.assert_called_once()

    def test_runs_full_analysis(self, pipeline, tmp_path):
        state = BatchClassState(
            class_name="cat", class_dir=tmp_path,
            result=ClassAnalysisResult(class_name="cat"),
        )
        state.result.edit_results = [make_edit_result()]
        img = _img()
        pred = FakePred()
        state.images = ImageSet(
            annotated_inspect=[(img, pred, "cat")],
        )
        pipeline._classify_features_inner = MagicMock()
        pipeline._run_final_analysis_inner = MagicMock(return_value=FinalAnalysis(class_name="cat"))
        pipeline._save_checkpoint = MagicMock()
        pipeline._log_completion = MagicMock()

        pipeline._finalize_class_batch(state)
        pipeline._classify_features_inner.assert_called_once()


# ============================================================================
# _classify_features / _run_final_analysis
# ============================================================================

class TestClassifyFeatures:
    def test_updates_features(self, pipeline):
        pipeline._status = MagicMock()
        vlm = MagicMock()
        vlm.classify_features.return_value = [{"feature_type": "intrinsic", "feature_name": "ears"}]
        pipeline.models.vlm.return_value = vlm

        er = make_edit_result()
        pipeline._classify_features("cat", [er])
        assert er.feature_type == "intrinsic"


class TestRunFinalAnalysis:
    def test_returns_result(self, pipeline):
        pipeline.models.vlm = MagicMock()
        pipeline.models.offload_vlm = MagicMock()
        final = FinalAnalysis(class_name="cat", robustness_score=8)
        pipeline._run_final_analysis_inner = MagicMock(return_value=final)

        result = pipeline._run_final_analysis("cat", ClassAnalysisResult(class_name="cat"), _img())
        assert result.robustness_score == 8


# ============================================================================
# _run_baseline / _generate_edit_instructions / _run_edits
# ============================================================================

class TestRunBaseline:
    def test_baseline_classifies_and_finds_negatives(self, pipeline, tmp_path):
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline.models.sampler.return_value.sample_from_classes.return_value = []
        pipeline._status = MagicMock()
        pipeline._classify_inspect_images = MagicMock(
            return_value=[{"type": "positive", "top_k": []}]
        )
        pipeline._classify_negative_images = MagicMock(return_value=[])
        pipeline._find_confusing_from_baseline = MagicMock(return_value=[])

        result = pipeline._run_baseline("cat", ImageSet(), tmp_path)
        assert len(result) == 1
        pipeline._find_confusing_from_baseline.assert_called_once()


class TestGenerateEditInstructions:
    def test_wraps_inner(self, pipeline):
        pipeline.models.vlm = MagicMock()
        pipeline.models.offload_vlm = MagicMock()
        pipeline._generate_edit_instructions_inner = MagicMock(return_value=[_edit_input()])

        result = pipeline._generate_edit_instructions("cat", ImageSet(), ClassAnalysisResult(class_name="cat"))
        assert len(result) == 1


# ============================================================================
# _generate_positive_edits / _generate_negative_edits
# ============================================================================

class TestGeneratePositiveEdits:
    def test_generates(self, pipeline):
        pipeline.models._vlm = MagicMock()
        edits = [FeatureEditPlan("ears", "Remove ears", "removal", "high", "test")]
        pipeline.models._vlm.generate_feature_edits.return_value = edits
        pipeline._current_detected = [
            {"name": "ears", "category": "part", "feature_type": "intrinsic", "gradcam_attention": "high"},
        ]

        img = _img()
        pred = FakePred()
        positives = [(img, pred, "cat")]
        baseline = [{"class_confidence": 0.9}]
        result = pipeline._generate_positive_edits("cat", positives, baseline)
        assert len(result) == 1


class TestGenerateNegativeEdits:
    def test_generates(self, pipeline):
        img = _img()
        pred = FakePred()
        images = ImageSet(annotated_negatives=[(img, pred, "dog")])
        baseline = [{"class_confidence": 0.9}, {"class_confidence": 0.15}]

        instr = EditInstruction(edit="Add fur", hypothesis="h", type="feature_addition", target="negative", priority=3, image_index=0)
        pipeline._analyze_one_negative = MagicMock(return_value=[
            EditInput(img, instr, 0.15)
        ])
        pipeline._status = MagicMock()

        result = pipeline._generate_negative_edits("cat", images, baseline)
        assert len(result) == 1


# ============================================================================
# _run_edits
# ============================================================================

class TestRunEdits:
    def test_returns_empty_when_no_inputs(self, pipeline, tmp_path):
        result = ClassAnalysisResult(class_name="cat")
        pipeline._generate_edit_instructions = MagicMock(return_value=[])
        pipeline._deduplicate_inputs = MagicMock(return_value=[])
        pipeline._status = MagicMock()

        out = pipeline._run_edits(result, ImageSet(), tmp_path)
        assert out == []

    def test_applies_edits(self, pipeline, tmp_path):
        result = ClassAnalysisResult(class_name="cat")
        inp = _edit_input()
        pipeline._generate_edit_instructions = MagicMock(return_value=[inp])
        pipeline._deduplicate_inputs = MagicMock(return_value=[inp])
        pipeline._apply_edits = MagicMock(return_value=[make_edit_result()])
        pipeline._status = MagicMock()

        out = pipeline._run_edits(result, ImageSet(), tmp_path)
        assert len(out) == 1


# ============================================================================
# _analyze_one_negative
# ============================================================================

class TestAnalyzeOneNegative:
    def test_success(self, pipeline):
        inp = _edit_input(target="negative")
        pipeline._negative_to_inputs = MagicMock(return_value=[inp])

        sample = NegativeSample(_img(), "cat", "dog", 0.3, 0)
        result = pipeline._analyze_one_negative(sample)
        assert len(result) == 1

    def test_returns_empty_on_generic_exception(self, pipeline):
        import torch
        pipeline._negative_to_inputs = MagicMock(side_effect=torch.cuda.OutOfMemoryError("OOM"))
        pipeline.models.offload_all = MagicMock()
        # Second call also fails
        pipeline._negative_to_inputs.side_effect = [
            torch.cuda.OutOfMemoryError("OOM"), RuntimeError("fail")
        ]

        sample = NegativeSample(_img(), "cat", "dog", 0.3, 0)
        result = pipeline._analyze_one_negative(sample)
        assert result == []


# ============================================================================
# run_class
# ============================================================================

class TestRunClass:
    def test_returns_cached(self, pipeline):
        cached = ClassAnalysisResult(class_name="cat")
        pipeline._try_load_checkpoint = MagicMock(return_value=cached)
        result = pipeline.run_class("cat")
        assert result is cached

    def test_runs_analysis(self, pipeline):
        pipeline._try_load_checkpoint = MagicMock(return_value=None)
        pipeline._analyze_class = MagicMock(return_value=ClassAnalysisResult(class_name="cat"))
        result = pipeline.run_class("cat")
        assert result.class_name == "cat"


# ============================================================================
# _apply_edits (unit, mocked)
# ============================================================================

class TestApplyEditsUnit:
    def test_applies_and_collects(self, pipeline, tmp_path):
        pipeline.models.editor = MagicMock()
        pipeline.models.classifier = MagicMock()
        pipeline.models.offload_editor = MagicMock()
        pipeline.models.offload_classifier = MagicMock()
        pipeline._attach_original_gradcams = MagicMock()
        pipeline._test_single_edit = MagicMock(return_value=make_edit_result())
        pipeline.cfg.low_vram = True

        inp = _edit_input()
        results = pipeline._apply_edits([inp], "cat", tmp_path)
        assert len(results) == 1
        pipeline.models.offload_editor.assert_called_once()

    def test_skips_none_results(self, pipeline, tmp_path):
        pipeline.models.editor = MagicMock()
        pipeline.models.classifier = MagicMock()
        pipeline._attach_original_gradcams = MagicMock()
        pipeline._test_single_edit = MagicMock(return_value=None)
        pipeline.cfg.low_vram = False

        inp = _edit_input()
        results = pipeline._apply_edits([inp], "cat", tmp_path)
        assert len(results) == 0


# ============================================================================
# _generate_edit_variants (unit, mocked)
# ============================================================================

class TestGenerateEditVariants:
    def test_generates_n_variants(self, pipeline, tmp_path):
        pipeline.cfg.generations_per_edit = 3
        pipeline._generate_one_variant = MagicMock(return_value=make_generation())

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        result = pipeline._generate_edit_variants(inp, ctx)
        assert len(result) == 3

    def test_skips_none(self, pipeline, tmp_path):
        pipeline.cfg.generations_per_edit = 2
        pipeline._generate_one_variant = MagicMock(side_effect=[make_generation(), None])

        inp = _edit_input()
        ctx = EditContext("cat", tmp_path, "pos_0", 0, 0)
        result = pipeline._generate_edit_variants(inp, ctx)
        assert len(result) == 1


class TestBuildAnalysisSummary:
    def test_builds_summary(self, pipeline):
        er = make_edit_result(instruction="Remove background")
        detected = [{"name": "background", "feature_type": "contextual"}]
        pipeline._current_detected = detected
        summary = pipeline._build_analysis_summary([er], detected)
        assert len(summary) == 1
        assert "feature" in summary[0]
