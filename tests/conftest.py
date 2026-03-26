"""Shared fixtures for all tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
from PIL import Image


def pytest_addoption(parser):
    """Register --gpu command-line flag."""
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU integration tests (requires CUDA and model weights)",
    )

from src.config import Config
from src.pipeline import (
    ClassAnalysisResult, EditContext, EditInput, EditResult,
    GenerationResult, ImageSet, DiscoveredFeatures, NegativeSample,
)
from src.vlm import EditInstruction


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def make_generation(seed=0, conf=0.8, delta=-0.1, path="edit.jpg"):
    return GenerationResult(
        seed=seed, edited_confidence=conf,
        delta=delta, edited_image_path=path,
    )


def make_edit_result(
    instruction="Remove background", confirmed=False, delta=-0.2,
    feature_type="contextual", generations=None,
):
    gens = generations or [make_generation(delta=delta)]
    return EditResult(
        instruction=instruction, hypothesis="Test hypothesis",
        edit_type="feature_removal", target_type="positive",
        priority=3, original_confidence=0.9,
        original_image_path="orig.jpg", generations=gens,
        mean_edited_confidence=0.9 + delta,
        mean_delta=delta, std_delta=0.01,
        min_delta=delta, max_delta=delta,
        confirmed=confirmed, confirmation_count=1 if confirmed else 0,
        feature_type=feature_type, feature_name="background",
    )


def make_instruction(edit="Remove the ears", target="positive", edit_type="feature_removal"):
    return EditInstruction(
        edit=edit, hypothesis="Test", type=edit_type,
        target=target, priority=3, image_index=0,
    )


def make_image(size=(64, 64), color="red"):
    return Image.new("RGB", size, color)


class FakePrediction(NamedTuple):
    label_name: str = "tabby cat"
    confidence: float = 0.85
    top_k: list = [("tabby cat", 0.85), ("tiger cat", 0.10)]
    gradcam_image: Image.Image | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return Config(
        device="cpu", low_vram=False, output_dir="/tmp/test_output",
        resume=False, use_statistical_validation=True,
    )


@pytest.fixture
def sample_generation():
    return make_generation()


@pytest.fixture
def sample_edit_result():
    return make_edit_result(confirmed=True, delta=-0.25)


@pytest.fixture
def sample_class_result():
    result = ClassAnalysisResult(class_name="tabby cat")
    result.edit_results = [
        make_edit_result(confirmed=True, delta=-0.3, instruction="Remove ears"),
        make_edit_result(confirmed=False, delta=-0.02, instruction="Change lighting"),
        make_edit_result(confirmed=True, delta=-0.2, instruction="Remove whiskers"),
    ]
    result.detected_features = [
        {"name": "ears", "category": "object_part", "feature_type": "intrinsic", "gradcam_attention": "high"},
        {"name": "background", "category": "context", "feature_type": "contextual", "gradcam_attention": "medium"},
        {"name": "whiskers", "category": "object_part", "feature_type": "intrinsic", "gradcam_attention": "high"},
    ]
    result.essential_features = ["ears", "whiskers"]
    result.gradcam_summary = "Focus on head region | Attention on body"
    return result


@pytest.fixture
def sample_image():
    return make_image()
