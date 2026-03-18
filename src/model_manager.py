"""
Model lifecycle manager for the analysis pipeline.

Single responsibility: load/offload models for VRAM efficiency.
Uses a singleton-like pattern: requesting any model auto-offloads all others.
"""
from __future__ import annotations

import logging
import os

import torch

from .classifier import ImageNetClassifier
from .config import Config
from .dataset import ImageNetSampler
from .editor import ImageEditor
from .vlm import QwenVLAnalyzer

logger = logging.getLogger(__name__)


class ModelManager:
    """Loads/offloads models for VRAM efficiency."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._classifier: ImageNetClassifier | None = None
        self._vlm: QwenVLAnalyzer | None = None
        self._editor: ImageEditor | None = None
        self._sampler: ImageNetSampler | None = None

    def _ensure_only(self, model_name: str):
        """Delete all models except the requested one to free memory."""
        if not self.cfg.low_vram:
            return
        wrappers = [
            ("classifier", self._classifier),
            ("vlm", self._vlm),
            ("editor", self._editor),
        ]
        for name, wrapper in wrappers:
            if name != model_name and wrapper and wrapper.loaded:
                wrapper.offload()
        torch.cuda.empty_cache()

    def classifier(self) -> ImageNetClassifier:
        """Get classifier, loading if needed. Auto-offloads others in low_vram."""
        self._ensure_only("classifier")
        if self._classifier is None:
            self._classifier = ImageNetClassifier(
                model_name=self.cfg.classifier_model,
                device=self.cfg.device,
                attention_method=self.cfg.attention_method,
            )
        elif not self._classifier.loaded:
            self._classifier.load_to_gpu()
        return self._classifier

    def vlm(self) -> QwenVLAnalyzer:
        """Get VLM, loading if needed. Auto-offloads others in low_vram."""
        self._ensure_only("vlm")
        if self._vlm is None:
            self._vlm = QwenVLAnalyzer(
                model_name=self.cfg.vlm_model,
                device=self.cfg.device,
                dtype=self.cfg.vlm_dtype,
            )
        elif not self._vlm.loaded:
            self._vlm.load_to_gpu()
        return self._vlm

    def editor(self) -> ImageEditor:
        """Get editor, loading if needed. Auto-offloads others in low_vram."""
        self._ensure_only("editor")
        if self._editor is None:
            self._editor = ImageEditor(
                model_name=self.cfg.editor_model,
                device=self.cfg.device,
                dtype=self.cfg.diffusion_dtype,
                use_8bit=self.cfg.use_8bit_editor,
            )
        elif not self._editor.loaded:
            self._editor.load_to_gpu()
        return self._editor

    def sampler(self) -> ImageNetSampler:
        """Get dataset sampler, initializing if needed."""
        if self._sampler is None:
            self._sampler = ImageNetSampler(
                dataset_name=self.cfg.hf_dataset,
                split=self.cfg.hf_dataset_split,
                hf_token=self.cfg.hf_token or os.environ.get("HF_TOKEN"),
                max_scan=self.cfg.max_scan,
                seed=self.cfg.random_seed,
            )
        return self._sampler

    def offload_classifier(self):
        """Fully delete classifier to free VRAM."""
        if self._classifier and self._classifier.loaded:
            self._classifier.offload()

    def offload_vlm(self):
        """Fully delete VLM to free VRAM."""
        if self._vlm and self._vlm.loaded:
            self._vlm.offload()

    def offload_editor(self):
        """Fully delete editor to free VRAM."""
        if self._editor and self._editor.loaded:
            self._editor.offload()

    def offload_all(self):
        """Offload all models and clear CUDA cache."""
        self.offload_classifier()
        self.offload_vlm()
        self.offload_editor()
        torch.cuda.empty_cache()
