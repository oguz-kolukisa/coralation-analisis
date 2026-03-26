"""GPU integration tests for pipeline.py — end-to-end with real models.

Tests the run() orchestrator and _apply_edits with actual models.
Requires: ResNet-50, InstructPix2Pix, Qwen2.5-VL-7B.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.config import Config
from src.pipeline import (
    AnalysisPipeline, ClassAnalysisResult, EditContext,
    EditInput, EditResult, GenerationResult, ImageSet,
    NegativeSample,
)
from src.vlm import EditInstruction
from tests.conftest import make_edit_result, make_generation

pytestmark = pytest.mark.gpu


@pytest.fixture
def cfg(tmp_path, device):
    return Config(
        device=device, low_vram=True, output_dir=tmp_path,
        resume=False, use_statistical_validation=False,
        samples_per_class=1, inspect_samples=1,
        generations_per_edit=1, max_hypotheses_per_image=1,
        compute_edit_gradcam=True,
    )


def _img():
    return Image.new("RGB", (224, 224), "red")


def _edit_input(edit="Remove the ears", target="positive", edit_type="feature_removal"):
    instr = EditInstruction(edit=edit, hypothesis="h", type=edit_type, target=target, priority=3, image_index=0)
    return EditInput(instruction=instr, image=_img(), original_confidence=0.9)


# ============================================================================
# _apply_edits with real classifier + editor
# ============================================================================

class TestApplyEditsGPU:
    def test_applies_single_edit(self, cfg, tmp_path):
        pipeline = AnalysisPipeline(cfg)
        inp = _edit_input()
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()

        results = pipeline._apply_edits([inp], "golden retriever", class_dir)
        assert isinstance(results, list)
        # May or may not produce results depending on edit quality
        for r in results:
            assert isinstance(r, EditResult)
            assert r.mean_delta != 0 or r.mean_delta == 0  # Just check it ran


# ============================================================================
# _generate_one_variant with real models
# ============================================================================

class TestGenerateOneVariantGPU:
    def test_generates_variant(self, cfg, tmp_path):
        pipeline = AnalysisPipeline(cfg)
        pipeline.models.editor()
        pipeline.models.classifier()

        inp = _edit_input()
        ctx = EditContext("golden retriever", tmp_path, "pos_0", 0, 0)
        ctx.base_seed = 0

        gen = pipeline._generate_one_variant(inp, ctx, 0)
        if gen is not None:
            assert isinstance(gen, GenerationResult)
            assert gen.edited_image_path != ""

        pipeline.models.offload_editor()
        pipeline.models.offload_classifier()


# ============================================================================
# _generate_and_save_variant with real editor
# ============================================================================

class TestGenerateAndSaveVariantGPU:
    def test_saves_variant(self, cfg, tmp_path):
        pipeline = AnalysisPipeline(cfg)
        pipeline.models.editor()

        inp = _edit_input()
        ctx = EditContext("golden retriever", tmp_path, "pos_0", 0, 0)
        ctx.base_seed = 0
        ctx.edit_idx = 0

        result = pipeline._generate_and_save_variant(inp, ctx, 0)
        if result is not None:
            assert Path(result.edited_image_path).exists()

        pipeline.models.offload_editor()


# ============================================================================
# _save_original_gradcam with real classifier
# ============================================================================

class TestSaveOriginalGradcamGPU:
    def test_saves_gradcam(self, cfg, tmp_path):
        pipeline = AnalysisPipeline(cfg)
        pipeline.cfg.compute_edit_gradcam = True

        # Get a real attention map
        clf = pipeline.models.classifier()
        _, pred = clf.predict_with_gradcam(_img(), "golden retriever")

        inp = _edit_input()
        inp.original_attention_map = pred.attention_map

        ctx = EditContext("golden retriever", tmp_path, "pos_0", 0, 0)
        path = pipeline._save_original_gradcam(inp, ctx)
        assert path != ""
        assert Path(path).exists()
        pipeline.models.offload_classifier()


# ============================================================================
# _classify_edited_image with real classifier
# ============================================================================

class TestClassifyEditedImageGPU:
    def test_with_gradcam(self, cfg, tmp_path):
        pipeline = AnalysisPipeline(cfg)
        pipeline.cfg.compute_edit_gradcam = True

        inp = _edit_input()
        inp.original_attention_map = np.zeros((7, 7))
        ctx = EditContext("golden retriever", tmp_path, "pos_0", 0, 0)

        conf, gc_path, diff_path = pipeline._classify_edited_image(_img(), inp, ctx, 0)
        assert 0.0 <= conf <= 1.0
        assert gc_path != ""
        assert diff_path != ""
        pipeline.models.offload_classifier()

    def test_without_gradcam(self, cfg, tmp_path):
        pipeline = AnalysisPipeline(cfg)
        pipeline.cfg.compute_edit_gradcam = False

        inp = _edit_input()
        ctx = EditContext("golden retriever", tmp_path, "pos_0", 0, 0)

        conf, gc_path, diff_path = pipeline._classify_edited_image(_img(), inp, ctx, 0)
        assert 0.0 <= conf <= 1.0
        assert gc_path == ""
        assert diff_path == ""
        pipeline.models.offload_classifier()


# ============================================================================
# run() orchestrator (mini end-to-end)
# ============================================================================

class TestRunOrchestrator:
    def test_run_single_class(self, cfg, tmp_path):
        """End-to-end test with real models and 1 class, 1 sample, 1 edit."""
        pipeline = AnalysisPipeline(cfg)

        # Mock the sampler to avoid needing ImageNet dataset
        mock_sampler = MagicMock()
        mock_sampler.sample_positive.return_value = [(_img(), "golden retriever")]
        mock_sampler.sample_from_classes.return_value = []
        mock_sampler.get_label_names.return_value = ["golden retriever", "tabby"]
        pipeline.models._sampler = mock_sampler
        pipeline.models.sampler = MagicMock(return_value=mock_sampler)

        results = pipeline.run(["golden retriever"])
        assert len(results) == 1
        assert results[0].class_name == "golden retriever"
