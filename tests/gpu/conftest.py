"""Fixtures for GPU integration tests.

These tests require:
- CUDA-capable GPU with sufficient VRAM
- Downloaded model weights (ResNet-50 at minimum)

Run with: uv run python -m pytest tests/gpu/ -v --gpu
Skip by default: tests are marked with @pytest.mark.gpu
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless --gpu flag is provided."""
    if not config.getoption("--gpu", default=False):
        skip_gpu = pytest.mark.skip(reason="GPU tests require --gpu flag")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    """Return CUDA device if available, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture(scope="session")
def cpu_device():
    return "cpu"


@pytest.fixture
def sample_image():
    """A simple 224x224 RGB image for classifier tests."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def large_image():
    """A larger image that needs resizing for editor tests."""
    return Image.new("RGB", (1024, 768), color="blue")


@pytest.fixture
def small_image():
    """A tiny image for edge-case testing."""
    return Image.new("RGB", (32, 32), color="green")
