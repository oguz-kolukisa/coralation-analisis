"""
Analysis Pipeline - Main orchestration for bias/shortcut discovery.

## How It Works

1. **Knowledge Discovery**: VLM uses world knowledge to identify potential shortcuts
   (e.g., for "cat", suggests "yarn ball", "milk bowl" as commonly associated features)

2. **Image Inspection**: Analyze multiple images with Grad-CAM to find visual features
   - Inspect more images than we edit (cheaper to analyze than edit)
   - Discover both intrinsic (object parts) and contextual (background) features

3. **Hypothesis Generation**: VLM creates edit instructions for each feature
   - "Remove the pointed ears" to test if model relies on ears
   - "Replace background with white" to test for background bias

4. **Counterfactual Testing**: Apply edits and measure confidence changes
   - Multiple generations per edit for statistical robustness
   - If removing contextual feature drops confidence → SHORTCUT found

5. **Feature Classification**: VLM classifies each feature as intrinsic/contextual
   - Combined with confidence delta to determine shortcut vs essential

Usage:
    pipeline = AnalysisPipeline(config)
    results = pipeline.run(["tabby cat", "golden retriever"])
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .analysis import StatisticalValidator
from .config import Config
from .model_manager import ModelManager
from .models.attention_maps import compute_attention_diff, render_diff_heatmap
from .vlm import (
    DetectedFeature, EditInstruction, FinalAnalysis,
)

logger = logging.getLogger(__name__)


def _flush_logs():
    """Flush all log handlers to ensure logs are written immediately."""
    for handler in logging.root.handlers:
        handler.flush()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GenerationResult:
    """Result from a single image edit generation."""
    seed: int
    edited_confidence: float
    delta: float
    edited_image_path: str
    edit_verified: bool = True
    verification_confidence: float = 1.0
    verification_description: str = ""
    gradcam_image_path: str = ""
    gradcam_diff_path: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> GenerationResult:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class EditResult:
    """Complete result from testing one edit hypothesis across multiple generations."""
    instruction: str
    hypothesis: str
    edit_type: str
    target_type: str
    priority: int
    original_confidence: float
    original_image_path: str
    original_gradcam_path: str = ""
    generations: list[GenerationResult] = field(default_factory=list)

    mean_edited_confidence: float = 0.0
    mean_delta: float = 0.0
    std_delta: float = 0.0
    min_delta: float = 0.0
    max_delta: float = 0.0
    confirmed: bool = False
    confirmation_count: int = 0

    p_value: float = 1.0
    cohens_d: float = 0.0
    effect_size: str = ""
    statistically_significant: bool = False
    practically_significant: bool = False

    feature_type: str = ""
    feature_name: str = ""
    likely_failed: bool = False

    @property
    def edited_confidence(self) -> float:
        return self.mean_edited_confidence

    @property
    def delta(self) -> float:
        return self.mean_delta

    @property
    def edited_image_path(self) -> str:
        return self.generations[0].edited_image_path if self.generations else ""

    @classmethod
    def from_dict(cls, data: dict) -> EditResult:
        generations = [GenerationResult.from_dict(g) for g in data.pop("generations", [])]
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid, generations=generations)


@dataclass
class ImageSet:
    """All sampled images for one class analysis."""
    inspect: list[tuple[Image.Image, str]] = field(default_factory=list)
    edit: list[tuple[Image.Image, str]] = field(default_factory=list)
    negative: list[tuple[Image.Image, str]] = field(default_factory=list)
    confusing_classes: list[str] = field(default_factory=list)
    annotated_inspect: list = field(default_factory=list)
    annotated_negatives: list = field(default_factory=list)

    @property
    def positives(self):
        """Edit-subset of annotated inspect images."""
        return self.annotated_inspect[:len(self.edit)]


@dataclass
class NegativeSample:
    """A negative image with its classification context."""
    image: Image.Image
    class_name: str
    true_label: str
    confidence: float
    index: int


@dataclass
class EditInput:
    """One edit to test: image + instruction + original confidence."""
    image: Image.Image
    instruction: EditInstruction
    original_confidence: float
    original_attention_map: np.ndarray | None = None


@dataclass
class EditContext:
    """Groups parameters for a single edit operation."""
    class_name: str
    class_dir: Path
    prefix: str
    edit_idx: int
    iteration: int
    base_seed: int = 0


@dataclass
class PendingGeneration:
    """An edited image saved to disk, awaiting classification."""
    edit_input_idx: int
    seed: int
    edited_image_path: str
    gen_idx: int


@dataclass
class BatchClassState:
    """Per-class state for batched pipeline execution."""
    class_name: str
    class_dir: Path
    result: ClassAnalysisResult
    images: ImageSet | None = None
    edit_inputs: list[EditInput] = field(default_factory=list)
    pending: list[PendingGeneration] = field(default_factory=list)
    failed: bool = False


@dataclass
class DiscoveredFeatures:
    """Result of image-based feature discovery."""
    detected: list[dict] = field(default_factory=list)
    essential: list[str] = field(default_factory=list)
    gradcam_summary: str = ""


@dataclass
class ClassAnalysisResult:
    """Complete analysis results for one class."""
    class_name: str

    key_features: list[str] = field(default_factory=list)
    essential_features: list[str] = field(default_factory=list)
    spurious_features: list[str] = field(default_factory=list)
    detected_features: list[dict] = field(default_factory=list)
    knowledge_based_features: list[dict] = field(default_factory=list)
    gradcam_summary: str = ""
    model_focus: str = ""

    baseline_results: list[dict] = field(default_factory=list)

    edit_results: list[EditResult] = field(default_factory=list)
    confirmed_hypotheses: list[EditResult] = field(default_factory=list)

    iterations_completed: int = 0
    vlm_insights: list[str] = field(default_factory=list)
    confirmed_shortcuts: list[str] = field(default_factory=list)

    feature_importance: list[dict] = field(default_factory=list)
    robustness_score: int = 5
    risk_level: str = "MEDIUM"
    vulnerabilities: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    final_summary: str = ""

    def apply_discovered_features(self, discovered: DiscoveredFeatures):
        """Absorb feature discovery results."""
        self.detected_features = discovered.detected
        self.essential_features = discovered.essential
        self.gradcam_summary = discovered.gradcam_summary

    def apply_final_analysis(self, final: FinalAnalysis | None):
        """Absorb final VLM analysis into this result."""
        if not final:
            return
        self.feature_importance = final.feature_importance
        self.robustness_score = final.robustness_score
        self.risk_level = final.risk_level
        self.vulnerabilities = final.vulnerabilities
        self.recommendations = final.recommendations
        self.final_summary = final.summary
        self._extract_shortcuts(final.confirmed_shortcuts)

    def _extract_shortcuts(self, shortcuts: list):
        for s in shortcuts:
            name = s.get("feature", str(s)) if isinstance(s, dict) else str(s)
            self.confirmed_shortcuts.append(name)
            if name not in self.spurious_features:
                self.spurious_features.append(name)

    def finalize(self):
        """Populate derived fields. Call after all phases complete."""
        self.confirmed_hypotheses = [e for e in self.edit_results if e.confirmed]
        self.iterations_completed = 1
        self._populate_key_features()
        self._populate_model_focus()
        self._deduplicate_features()

    def _populate_key_features(self):
        if self.key_features or not self.detected_features:
            return
        seen = set()
        self.key_features = [
            f["name"] for f in self.detected_features
            if f.get("feature_type") == "intrinsic"
            and f.get("gradcam_attention") == "high"
            and not (f["name"] in seen or seen.add(f["name"]))
        ][:5]

    def _populate_model_focus(self):
        if self.model_focus or not self.gradcam_summary:
            return
        summaries = self.gradcam_summary.split("|")
        if summaries:
            self.model_focus = summaries[0].strip()

    def _deduplicate_features(self):
        if not self.essential_features or not self.spurious_features:
            return
        essential_set = {f.lower() for f in self.essential_features}
        self.spurious_features = [
            f for f in self.spurious_features if f.lower() not in essential_set
        ]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["summary"] = {
            "total_edits": len(self.edit_results),
            "total_generations": sum(len(e.generations) for e in self.edit_results),
            "confirmed_count": len(self.confirmed_hypotheses),
            "confirmation_rate": round(
                len(self.confirmed_hypotheses) / max(len(self.edit_results), 1), 2
            ),
            "iterations": self.iterations_completed,
            "robustness_score": self.robustness_score,
            "risk_level": self.risk_level,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ClassAnalysisResult:
        edit_dicts = data.pop("edit_results", [])
        data.pop("confirmed_hypotheses", None)
        data.pop("summary", None)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        result = cls(**valid)
        result.edit_results = [EditResult.from_dict(e) for e in edit_dicts]
        result.confirmed_hypotheses = [e for e in result.edit_results if e.confirmed]
        return result


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

class AnalysisPipeline:
    """Main orchestrator for bias/shortcut discovery."""

    def __init__(self, config: Config):
        self.cfg = config
        self.models = ModelManager(config)
        self._pbar = None
        self.stat_validator = StatisticalValidator(
            alpha=config.statistical_alpha,
            min_effect_size=config.min_effect_size,
            min_samples=2,
        )

    # =========================================================================
    # ORCHESTRATION — phase-first, 6 model swaps total
    # =========================================================================

    def run(self, classes: list[str]) -> list[ClassAnalysisResult]:
        """Run analysis for all classes with phase-first model loading."""
        states, cached_results = self._init_batch_states(classes)

        if states:
            phase_fns = [
                self._phase1_vlm_knowledge,
                self._phase2_classifier_baseline,
                self._phase3_vlm_features_edits,
                self._phase4_editor_generate,
                self._phase5_classifier_measure,
                self._phase6_vlm_final,
            ]
            phase_names = [
                "VLM: knowledge",
                "Classifier: baseline",
                "VLM: features+edits",
                "Editor: generate",
                "Classifier: measure",
                "VLM: final analysis",
            ]

            n_classes = len(states)
            self._pbar = tqdm(
                total=len(phase_fns), desc="Pipeline", unit="phase", ncols=120,
                bar_format="{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )
            self._pbar.set_postfix_str(f"{n_classes} classes")
            for phase_fn, name in zip(phase_fns, phase_names):
                self._pbar.set_description(f"Phase: {name}")
                phase_fn(states)
                self._pbar.update(1)
            self._pbar.close()
            self._pbar = None
            self._print_summary(states)

        return cached_results + self._collect_batch_results(states)

    def _print_summary(self, states: list[BatchClassState]):
        """Print a summary of what was processed."""
        active = [s for s in states if not s.failed]
        failed = [s for s in states if s.failed]
        total_baseline = sum(len(s.result.baseline_results) for s in active)
        total_edits = sum(len(s.edit_inputs) for s in active)
        total_generated = sum(len(s.pending) for s in active)
        total_confirmed = sum(len(s.result.confirmed_hypotheses) for s in active)
        print(f"\n{'='*60}")
        print(f"  Pipeline complete: {len(active)} classes"
              f"{f' ({len(failed)} failed)' if failed else ''}")
        print(f"  Baseline classifications:  {total_baseline:,}")
        print(f"  Edit hypotheses:           {total_edits:,}")
        print(f"  Generated images:          {total_generated:,}")
        print(f"  Confirmed biases:          {total_confirmed:,}")
        print(f"{'='*60}\n")

    def _init_batch_states(
        self, classes: list[str],
    ) -> tuple[list[BatchClassState], list[ClassAnalysisResult]]:
        """Create per-class state, loading from checkpoint where available."""
        states = []
        cached_results = []
        for class_name in classes:
            cached = self._try_load_checkpoint(class_name)
            if cached:
                cached_results.append(cached)
                continue
            class_dir = self._make_class_dir(class_name)
            result = ClassAnalysisResult(class_name=class_name)
            states.append(BatchClassState(
                class_name=class_name, class_dir=class_dir, result=result,
            ))
        return states, cached_results

    def _active_states(self, states: list[BatchClassState]) -> list[BatchClassState]:
        """Filter to non-failed states."""
        return [s for s in states if not s.failed]

    def _phase1_vlm_knowledge(self, states: list[BatchClassState]):
        """Phase 1: VLM discovers knowledge-based features for all classes."""
        self.models.vlm()
        active = self._active_states(states)
        for i, state in enumerate(active):
            label = f"  Knowledge [{i+1}/{len(active)}] {state.class_name}"
            self._pbar.set_postfix_str(label.strip()) if self._pbar else None
            try:
                all_classes = self.models.sampler().get_label_names()
                features = self.models._vlm.generate_knowledge_based_features(
                    state.class_name, all_classes,
                )
                state.result.knowledge_based_features = features.get(
                    "knowledge_based_features", [],
                )
            except Exception as e:
                logger.warning("Knowledge discovery failed for %s: %s", state.class_name, e)
        self.models.offload_vlm()

    def _phase2_classifier_baseline(self, states: list[BatchClassState]):
        """Phase 2: Classifier baseline + negative class selection from predictions."""
        self.models.classifier()
        active = self._active_states(states)
        self._phase2a_positive_baseline(active)
        self._phase2b_negative_sampling(active)
        self.models.offload_classifier()

    def _phase2a_positive_baseline(self, active: list[BatchClassState]):
        """Phase 2a: Sample positives + baseline classification with Grad-CAM."""
        for i, state in enumerate(active):
            label = f"  Baseline [{i+1}/{len(active)}] {state.class_name}"
            self._pbar.set_postfix_str(label.strip()) if self._pbar else None
            try:
                state.images = self._sample_images(state.class_name)
                if not state.images.inspect:
                    state.failed = True
                    continue
                state.result.baseline_results = self._classify_inspect_images(
                    state.class_name, state.images, state.class_dir,
                )
            except Exception as e:
                logger.error("Baseline failed for %s: %s", state.class_name, e)
                state.failed = True

    def _phase2b_negative_sampling(self, active: list[BatchClassState]):
        """Phase 2b: Find confusing classes from all predictions, sample negatives."""
        active = [s for s in active if not s.failed]
        for i, state in enumerate(active):
            label = f"  Negatives [{i+1}/{len(active)}] {state.class_name}"
            self._pbar.set_postfix_str(label.strip()) if self._pbar else None
            try:
                self._sample_and_classify_negatives(state)
            except Exception as e:
                logger.warning("Negative sampling failed for %s: %s", state.class_name, e)

    def _sample_and_classify_negatives(self, state: BatchClassState):
        """Sample negatives from confusing classes found in baseline predictions."""
        confusing = self._find_confusing_from_baseline(
            state.class_name, state.result.baseline_results,
        )
        state.images.confusing_classes = confusing
        sampler = self.models.sampler()
        state.images.negative = sampler.sample_from_classes(
            confusing, n_per_class=self.cfg.negative_samples_per_class,
        )
        neg_records = self._classify_negative_images(
            state.class_name, state.images, state.class_dir,
        )
        state.result.baseline_results += neg_records

    def _find_confusing_from_baseline(
        self, class_name: str, baseline_results: list[dict],
    ) -> list[str]:
        """Extract confusing classes from all baseline top-k predictions."""
        counts: dict[str, float] = {}
        for record in baseline_results:
            if record.get("type") != "positive":
                continue
            for label, conf in record.get("top_k", []):
                if label.lower() != class_name.lower() and conf > self.cfg.confusing_class_min_conf:
                    counts[label] = counts.get(label, 0) + conf
        sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
        return [label for label, _ in sorted_classes[:self.cfg.top_negative_classes]]

    def _phase3_vlm_features_edits(self, states: list[BatchClassState]):
        """Phase 3: VLM discovers features and generates edit instructions."""
        self.models.vlm()
        active = self._active_states(states)
        for i, state in enumerate(active):
            label = f"  Features+edits [{i+1}/{len(active)}] {state.class_name}"
            self._pbar.set_postfix_str(label.strip()) if self._pbar else None
            try:
                discovered = self._discover_image_features(
                    state.class_name, state.images, state.result.baseline_results,
                )
                state.result.apply_discovered_features(discovered)
                inputs = self._generate_edit_instructions_inner(
                    state.class_name, state.images, state.result,
                )
                state.edit_inputs = self._deduplicate_inputs(inputs)
            except Exception as e:
                logger.error("Feature/edit discovery failed for %s: %s", state.class_name, e)
                state.failed = True
        self.models.offload_vlm()

    def _phase4_editor_generate(self, states: list[BatchClassState]):
        """Phase 4: Editor generates all edited images, saved to disk."""
        self.models.editor()
        active = self._active_states(states)
        total_edits = sum(len(s.edit_inputs) for s in active)
        total_imgs = total_edits * self.cfg.generations_per_edit
        img_bar = tqdm(total=total_imgs, desc="    Generating images", leave=False, unit="img")
        for i, state in enumerate(active):
            img_bar.set_description(f"    {state.class_name}: Generating images")
            state.pending = self._generate_all_variants_tracked(state, img_bar)
        img_bar.close()
        self.models.offload_editor()

    def _generate_all_variants(self, state: BatchClassState) -> list[PendingGeneration]:
        """Generate all edit variants for one class using the editor."""
        total = len(state.edit_inputs) * self.cfg.generations_per_edit
        pbar = tqdm(total=total, desc=f"    {state.class_name}: Variants", leave=False, unit="img")
        pending = self._generate_variants_loop(state, pbar)
        pbar.close()
        return pending

    def _generate_all_variants_tracked(
        self, state: BatchClassState, pbar: tqdm,
    ) -> list[PendingGeneration]:
        """Generate all edit variants, updating an external progress bar."""
        return self._generate_variants_loop(state, pbar)

    def _generate_variants_loop(
        self, state: BatchClassState, pbar: tqdm,
    ) -> list[PendingGeneration]:
        """Core loop: generate variants and update the given progress bar."""
        pending = []
        for j, inp in enumerate(state.edit_inputs):
            ctx = self._make_edit_context(
                state.class_name, state.class_dir, inp.instruction,
            )
            ctx.edit_idx = j
            ctx.base_seed = j * 100
            for gen_idx in range(self.cfg.generations_per_edit):
                result = self._generate_and_save_variant(inp, ctx, gen_idx)
                if result:
                    pending.append(result)
                pbar.update(1)
        return pending

    def _phase5_classifier_measure(self, states: list[BatchClassState]):
        """Phase 5: Classifier measures impact of all edits."""
        self.models.classifier()
        active = self._active_states(states)
        total_pending = sum(len(s.pending) for s in active)
        img_bar = tqdm(total=total_pending, desc="    Classifying edits", leave=False, unit="img")
        for i, state in enumerate(active):
            img_bar.set_description(f"    {state.class_name}: Classifying edits")
            try:
                state.result.edit_results = self._classify_all_variants_tracked(state, img_bar)
            except Exception as e:
                logger.error("Classification failed for %s: %s", state.class_name, e)
                state.failed = True
        img_bar.close()
        self.models.offload_classifier()

    def _classify_all_variants(self, state: BatchClassState) -> list[EditResult]:
        """Classify all pending variants and build EditResults."""
        self._attach_original_gradcams(state.edit_inputs, state.class_name)
        grouped = self._group_classified_generations(state)
        return self._assemble_edit_results(state, grouped)

    def _classify_all_variants_tracked(
        self, state: BatchClassState, pbar: tqdm,
    ) -> list[EditResult]:
        """Classify all pending variants, updating an external progress bar."""
        self._attach_original_gradcams(state.edit_inputs, state.class_name)
        grouped = self._group_classified_generations(state, pbar)
        return self._assemble_edit_results(state, grouped)

    def _group_classified_generations(
        self, state: BatchClassState, pbar: tqdm | None = None,
    ) -> dict[int, list[GenerationResult]]:
        """Classify each pending variant and group by edit index."""
        grouped: dict[int, list[GenerationResult]] = {}
        own_bar = pbar is None
        if own_bar:
            pbar = tqdm(total=len(state.pending), desc=f"    {state.class_name}: Classify", leave=False, unit="img")
        for p in state.pending:
            inp = state.edit_inputs[p.edit_input_idx]
            ctx = self._make_edit_context(
                state.class_name, state.class_dir, inp.instruction,
            )
            ctx.edit_idx = p.edit_input_idx
            gen = self._classify_saved_variant(p, inp, ctx)
            if gen:
                grouped.setdefault(p.edit_input_idx, []).append(gen)
            pbar.update(1)
        if own_bar:
            pbar.close()
        return grouped

    def _assemble_edit_results(
        self, state: BatchClassState,
        grouped: dict[int, list[GenerationResult]],
    ) -> list[EditResult]:
        """Build EditResults from grouped generations."""
        results = []
        for j, inp in enumerate(state.edit_inputs):
            gens = grouped.get(j, [])
            if not gens:
                continue
            er = self._build_single_batch_result(state, inp, j, gens)
            results.append(er)
        return results

    def _build_single_batch_result(
        self, state: BatchClassState, inp: EditInput,
        idx: int, gens: list[GenerationResult],
    ) -> EditResult:
        """Build one EditResult for a batch-processed edit."""
        orig_path = str(
            (state.class_dir / f"{self._edit_prefix(inp)}_original.jpg").resolve()
        )
        ctx = self._make_edit_context(
            state.class_name, state.class_dir, inp.instruction,
        )
        ctx.edit_idx = idx
        orig_gradcam = self._save_original_gradcam(inp, ctx)
        er = self._build_edit_result(
            inp.instruction, inp.original_confidence, orig_path, gens,
        )
        er.original_gradcam_path = orig_gradcam
        return er

    def _edit_prefix(self, inp: EditInput) -> str:
        """Build the file prefix for an edit input."""
        target = "pos" if inp.instruction.target == "positive" else "neg"
        return f"{target}_{inp.instruction.image_index}"

    def _phase6_vlm_final(self, states: list[BatchClassState]):
        """Phase 6: VLM classifies features and runs final analysis."""
        self.models.vlm()
        active = self._active_states(states)
        for i, state in enumerate(active):
            label = f"  Final [{i+1}/{len(active)}] {state.class_name}"
            self._pbar.set_postfix_str(label.strip()) if self._pbar else None
            try:
                self._finalize_class_batch(state)
            except Exception as e:
                logger.error("Final analysis failed for %s: %s", state.class_name, e)
        self.models.offload_vlm()

    def _finalize_class_batch(self, state: BatchClassState):
        """Classify features and run final analysis for one class."""
        if not state.result.edit_results or not state.images.positives:
            state.result.finalize()
            self._save_checkpoint(state.result, state.class_dir)
            return
        self._classify_features_inner(state.class_name, state.result.edit_results)
        base_image = state.images.positives[0][0]
        final = self._run_final_analysis_inner(
            state.class_name, state.result, base_image,
        )
        state.result.apply_final_analysis(final)
        state.result.finalize()
        self._save_checkpoint(state.result, state.class_dir)
        self._log_completion(state.result)

    def _classify_features_inner(self, class_name: str, edit_results: list[EditResult]):
        """Classify features without model lifecycle management."""
        vlm = self.models._vlm
        features = [
            {"instruction": er.instruction, "hypothesis": er.hypothesis}
            for er in edit_results
        ]
        classified = vlm.classify_features(class_name, features)
        for i, er in enumerate(edit_results):
            if i < len(classified):
                er.feature_type = classified[i].get("feature_type", "")
                er.feature_name = classified[i].get("feature_name", "")

    def _collect_batch_results(self, states: list[BatchClassState]) -> list[ClassAnalysisResult]:
        """Collect results from all states (including failed ones)."""
        return [s.result for s in states]

    def run_class(self, class_name: str) -> ClassAnalysisResult:
        """Run complete analysis for one class."""
        cached = self._try_load_checkpoint(class_name)
        if cached:
            return cached
        return self._analyze_class(class_name)

    def _try_load_checkpoint(self, class_name: str) -> ClassAnalysisResult | None:
        """Try loading from checkpoint if resume is enabled."""
        if not self.cfg.resume:
            return None
        cached = self._load_checkpoint(class_name)
        if cached:
            self._status("[SKIP] Loaded from checkpoint")
        return cached

    def _analyze_class(self, class_name: str) -> ClassAnalysisResult:
        """Execute all analysis phases for one class."""
        class_dir = self._make_class_dir(class_name)
        result = ClassAnalysisResult(class_name=class_name)
        result.knowledge_based_features = self._discover_knowledge_features(class_name)

        images = self._sample_images(class_name)
        if not images.inspect:
            self._status("[!] No images found")
            return result

        self._run_all_phases(class_name, images, result, class_dir)
        result.finalize()
        self._save_checkpoint(result, class_dir)
        self._log_completion(result)
        return result

    def _run_all_phases(self, class_name, images, result, class_dir):
        """Run baseline, discovery, edits, and final analysis."""
        result.baseline_results = self._run_baseline(class_name, images, class_dir)
        result.apply_discovered_features(self._discover_image_features(class_name, images, result.baseline_results))
        result.edit_results = self._run_edits(result, images, class_dir)
        result.apply_final_analysis(self._analyze_results(class_name, result, images))

    # =========================================================================
    # PHASES — each does ONE thing, <=20 lines
    # =========================================================================

    def _discover_knowledge_features(self, class_name: str) -> list[dict]:
        """Ask VLM what shortcuts might exist based on world knowledge."""
        self._status("VLM knowledge discovery...")
        try:
            all_classes = self.models.sampler().get_label_names()
            result = self.models.vlm().generate_knowledge_based_features(class_name, all_classes)
            return result.get("knowledge_based_features", [])
        except Exception as e:
            logger.warning("Knowledge-based discovery failed: %s", e)
            return []

    def _sample_images(self, class_name: str) -> ImageSet:
        """Sample positive images for analysis (negatives added after baseline)."""
        sampler = self.models.sampler()
        effective_inspect = max(self.cfg.inspect_samples, self.cfg.samples_per_class)
        all_positives = sampler.sample_positive(class_name, n=effective_inspect)
        if not all_positives:
            return ImageSet()

        return ImageSet(
            inspect=all_positives[:effective_inspect],
            edit=all_positives[:self.cfg.samples_per_class],
        )

    def _find_confusing_classes(self, class_name: str, edit_images: list) -> list[str]:
        """Combine classifier and VLM confusing classes."""
        self._status("Finding confusing classes...")
        classifier_classes = self._classifier_confusing_classes(class_name, edit_images)
        vlm_classes = self._vlm_confusing_classes(class_name)
        merged = self._merge_confusing_classes(classifier_classes, vlm_classes)
        return merged[:self.cfg.top_negative_classes]

    def _classifier_confusing_classes(self, class_name: str, edit_images: list) -> list[str]:
        """Get confusing classes from classifier predictions."""
        classifier = self.models.classifier()
        counts = self._collect_confusion_counts(classifier, class_name, edit_images)
        sorted_confusing = sorted(counts.items(), key=lambda x: -x[1])
        return [label for label, _ in sorted_confusing]

    def _collect_confusion_counts(self, classifier, class_name: str, images: list) -> dict:
        """Count how often non-target classes appear in top-k predictions."""
        counts: dict[str, int] = {}
        for img, _ in images[:3]:
            pred = classifier.predict(img, target_class_name=class_name, top_k=self.cfg.top_k_classes)
            for label, conf in pred.top_k:
                if label.lower() != class_name.lower() and conf > self.cfg.confusing_class_min_conf:
                    counts[label] = counts.get(label, 0) + 1
        return counts

    def _vlm_confusing_classes(self, class_name: str) -> list[str]:
        """Get confusing classes from VLM semantic knowledge."""
        all_classes = self.models.sampler().get_label_names()
        try:
            candidates = self.models.vlm().select_confusing_classes(
                class_name, all_classes, num_classes=self.cfg.top_negative_classes
            )
            return self._validate_class_names(candidates)
        except Exception as e:
            logger.warning("VLM confusing class selection failed for %s: %s", class_name, e)
            return []

    def _merge_confusing_classes(self, classifier_classes: list[str], vlm_classes: list[str]) -> list[str]:
        """Deduplicate and merge confusing classes, classifier-first priority."""
        seen: set[str] = set()
        merged: list[str] = []
        for name in classifier_classes + vlm_classes:
            key = name.lower()
            if key not in seen:
                merged.append(name)
                seen.add(key)
        return merged

    def _validate_class_names(self, candidates: list[str]) -> list[str]:
        """Filter class names to those resolvable in ImageNet labels."""
        classifier = self.models.classifier()
        valid = []
        for name in candidates:
            if classifier.is_valid_class(name):
                valid.append(name)
            else:
                logger.warning("Dropping invalid confusing class: '%s'", name)
        return valid

    def _run_baseline(self, class_name: str, images: ImageSet, class_dir: Path) -> list[dict]:
        """Run baseline: classify positives, find confusing classes, classify negatives."""
        self.models.classifier()
        self._status("Running baseline + Grad-CAM...")
        inspect_records = self._classify_inspect_images(class_name, images, class_dir)
        confusing = self._find_confusing_from_baseline(class_name, inspect_records)
        images.confusing_classes = confusing
        sampler = self.models.sampler()
        images.negative = sampler.sample_from_classes(
            confusing, n_per_class=self.cfg.negative_samples_per_class,
        )
        neg_records = self._classify_negative_images(class_name, images, class_dir)
        self.models.offload_classifier()
        return inspect_records + neg_records

    def _classify_inspect_images(self, class_name: str, images: ImageSet, class_dir: Path) -> list[dict]:
        """Classify inspection images, populating images.annotated_inspect."""
        clf = self.models.classifier()
        records = []
        for i, (img, true_label) in tqdm(
            enumerate(images.inspect), total=len(images.inspect),
            desc=f"    {class_name}: Inspect+GradCAM", leave=False, unit="img",
        ):
            pred = clf.predict(img, target_class_name=class_name, top_k=self.cfg.top_k_classes, compute_gradcam=True)
            conf = clf.get_class_confidence(img, class_name)
            self._save_positive_image(img, pred, class_dir, i)
            record = self._base_record(str((class_dir / f"pos_{i}_original.jpg").resolve()), true_label, pred, conf)
            record["type"] = "positive"
            record["for_editing"] = i < self.cfg.samples_per_class
            records.append(record)
            images.annotated_inspect.append((img, pred, true_label))
        return records

    def _classify_negative_images(self, class_name: str, images: ImageSet, class_dir: Path) -> list[dict]:
        """Classify negative images, filtering low-confidence."""
        clf = self.models.classifier()
        records = []
        for i, (img, true_label) in tqdm(
            enumerate(images.negative), total=len(images.negative),
            desc=f"    {class_name}: Negatives", leave=False, unit="img",
        ):
            pred = clf.predict(img, target_class_name=class_name, top_k=self.cfg.top_k_classes, compute_gradcam=False)
            conf = clf.get_class_confidence(img, class_name)
            if conf < self.cfg.min_negative_confidence:
                continue
            orig_path = class_dir / f"neg_{i}_original.jpg"
            img.save(orig_path)
            record = self._base_record(str(orig_path), true_label, pred, conf)
            record["type"] = "negative"
            record["confusing_class"] = true_label
            records.append(record)
            images.annotated_negatives.append((img, pred, true_label))
        return records

    def _save_positive_image(self, img, pred, class_dir: Path, index: int):
        """Save original image and its Grad-CAM overlay."""
        img.save(class_dir / f"pos_{index}_original.jpg")
        if pred.gradcam_image:
            pred.gradcam_image.save(class_dir / f"pos_{index}_gradcam.jpg")

    def _base_record(self, path: str, true_label: str, pred, conf: float) -> dict:
        """Build common fields for a baseline record."""
        return {
            "image_path": path, "true_label": true_label,
            "predicted_label": pred.label_name,
            "predicted_confidence": pred.confidence,
            "class_confidence": conf, "top_k": pred.top_k,
        }

    def _discover_image_features(
        self, class_name: str, images: ImageSet, baseline: list[dict],
    ) -> DiscoveredFeatures:
        """Analyze images with Grad-CAM to discover visual features."""
        vlm = self.models.vlm()
        result = DiscoveredFeatures()
        pbar = tqdm(
            total=len(images.annotated_inspect),
            desc=f"    {class_name}: Discover features", leave=False, unit="img",
        )
        for idx, (img, pred, _) in enumerate(images.annotated_inspect):
            if pred.gradcam_image:
                conf = baseline[idx]["class_confidence"]
                try:
                    discovery = vlm.discover_features(img, pred.gradcam_image, class_name, conf)
                    self._accumulate_discovery(discovery, idx, result)
                except Exception as e:
                    logger.warning("Feature discovery failed for image %d: %s", idx, e)
            n_feat = len(result.detected)
            n_essential = len(result.essential)
            pbar.set_postfix_str(f"{n_feat} features, {n_essential} essential")
            pbar.update(1)
        pbar.close()
        result.detected = self._deduplicate_features_by_name(result.detected)
        return result

    def _accumulate_discovery(self, discovery, idx: int, result: DiscoveredFeatures):
        """Add a single image's discovery to accumulated results."""
        separator = " | " if result.gradcam_summary else ""
        result.gradcam_summary += separator + discovery.gradcam_summary
        for f in discovery.features:
            result.detected.append({
                "name": f.name, "category": f.category,
                "feature_type": f.feature_type,
                "gradcam_attention": f.gradcam_attention, "source_image": idx,
            })
        for ef in discovery.intrinsic_features:
            if ef not in result.essential:
                result.essential.append(ef)

    def _deduplicate_features_by_name(self, features: list[dict]) -> list[dict]:
        """Remove duplicate features by name (case-insensitive)."""
        seen = set()
        unique = []
        for f in features:
            key = f["name"].lower()
            if key not in seen:
                unique.append(f)
                seen.add(key)
        return unique

    # =========================================================================
    # EDIT GENERATION & APPLICATION
    # =========================================================================

    def _run_edits(
        self, result: ClassAnalysisResult, images: ImageSet, class_dir: Path,
    ) -> list[EditResult]:
        """Generate edit instructions, deduplicate, and apply them."""
        inputs = self._generate_edit_instructions(result.class_name, images, result)
        inputs = self._deduplicate_inputs(inputs)
        if not inputs:
            return []
        self._status(f"Applying {len(inputs)} edits...")
        return self._apply_edits(inputs, result.class_name, class_dir)

    def _generate_edit_instructions(
        self, class_name: str, images: ImageSet, result: ClassAnalysisResult,
    ) -> list[EditInput]:
        """Generate edit instructions for positive and negative images."""
        self.models.vlm()
        result_inputs = self._generate_edit_instructions_inner(class_name, images, result)
        self.models.offload_vlm()
        return result_inputs

    def _generate_edit_instructions_inner(
        self, class_name: str, images: ImageSet, result: ClassAnalysisResult,
    ) -> list[EditInput]:
        """Generate edit instructions without model lifecycle management."""
        self._status(f"Generating edits for {len(images.positives)} images...")
        self._current_detected = result.detected_features
        inputs = self._generate_positive_edits(class_name, images.positives, result.baseline_results)
        inputs += self._generate_negative_edits(class_name, images, result.baseline_results)
        return inputs

    def _generate_positive_edits(
        self, class_name: str, positives: list, baseline: list[dict],
    ) -> list[EditInput]:
        """Generate feature-removal edits for positive images."""
        vlm = self.models._vlm
        features = self._to_detected_features(self._current_detected[:10])
        inputs = []
        pbar = tqdm(
            total=len(positives),
            desc=f"    {class_name}: Positive edits", leave=False, unit="img",
        )
        for idx, (img, pred, _) in enumerate(positives):
            conf = baseline[idx]["class_confidence"]
            try:
                edits = vlm.generate_feature_edits(img, features, class_name)
                new_edits = self._wrap_positive_edits(img, edits, conf, idx)
                inputs += new_edits
                pbar.set_postfix_str(f"{len(inputs)} edits total, {len(new_edits)} this img")
            except Exception as e:
                logger.warning("Edit generation failed for image %d: %s", idx, e)
                pbar.set_postfix_str(f"{len(inputs)} edits total, 0 this img")
            pbar.update(1)
        pbar.close()
        return inputs

    def _wrap_positive_edits(self, img, edits, conf: float, idx: int) -> list[EditInput]:
        """Convert VLM edit plans into EditInput objects."""
        return [
            EditInput(img, EditInstruction(
                edit=fe.edit_instruction, hypothesis=fe.hypothesis,
                type="feature_removal" if fe.edit_type == "removal" else fe.edit_type,
                target="positive",
                priority=5 if fe.expected_impact == "high" else 3,
                image_index=idx,
            ), conf)
            for fe in edits
        ]

    def _generate_negative_edits(
        self, class_name: str, images: ImageSet, baseline: list[dict],
    ) -> list[EditInput]:
        """Generate feature-addition edits for negative images."""
        inputs = []
        inspect_count = len(images.annotated_inspect)
        negatives = images.annotated_negatives[:self.cfg.max_edits_per_hypothesis]
        pbar = tqdm(
            total=len(negatives),
            desc=f"    {class_name}: Negative edits", leave=False, unit="img",
        )
        for i, (img, pred, true_label) in enumerate(negatives):
            sample = NegativeSample(img, class_name, true_label, baseline[inspect_count + i]["class_confidence"], i)
            new_edits = self._analyze_one_negative(sample)
            inputs += new_edits
            pbar.set_postfix_str(f"{len(inputs)} edits total, {len(new_edits)} this img")
            pbar.update(1)
        pbar.close()
        return inputs

    def _analyze_one_negative(self, sample: NegativeSample) -> list[EditInput]:
        """Generate edits for one negative image, with OOM retry."""
        try:
            return self._negative_to_inputs(sample)
        except torch.cuda.OutOfMemoryError:
            self.models.offload_all()
            try:
                return self._negative_to_inputs(sample)
            except Exception:
                return []

    def _negative_to_inputs(self, sample: NegativeSample) -> list[EditInput]:
        """Ask VLM for negative edits and wrap as EditInputs."""
        vlm = self.models.vlm()
        analysis = vlm.analyze_negative(
            sample.image, sample.class_name, sample.true_label, sample.confidence,
            max_hypotheses=self.cfg.max_hypotheses_per_image,
        )
        for instr in analysis.edit_instructions:
            instr.image_index = sample.index
        return [EditInput(sample.image, instr, sample.confidence) for instr in analysis.edit_instructions]

    def _to_detected_features(self, feature_dicts: list[dict]) -> list[DetectedFeature]:
        """Convert feature dicts to DetectedFeature objects."""
        return [
            DetectedFeature(
                name=f["name"], category=f["category"],
                feature_type=f["feature_type"], location="",
                gradcam_attention=f["gradcam_attention"], reasoning="",
            )
            for f in feature_dicts
        ]

    def _apply_edits(
        self, inputs: list[EditInput], class_name: str, class_dir: Path,
    ) -> list[EditResult]:
        """Apply all edit instructions. Returns list of EditResults."""
        self.models.editor()
        self.models.classifier()
        self._attach_original_gradcams(inputs, class_name)
        results = []
        for j, inp in tqdm(
            enumerate(inputs), total=len(inputs),
            desc="    Apply edits", leave=False, unit="edit",
        ):
            ctx = self._make_edit_context(class_name, class_dir, inp.instruction)
            ctx.edit_idx = j
            edit_result = self._test_single_edit(inp, ctx)
            if edit_result:
                results.append(edit_result)
            _flush_logs()
        if self.cfg.low_vram:
            self.models.offload_editor()
            self.models.offload_classifier()
        return results

    def _attach_original_gradcams(self, inputs: list[EditInput], class_name: str):
        """Attach original attention maps to inputs for diff computation."""
        if not self.cfg.compute_edit_gradcam:
            return
        seen: dict[int, np.ndarray | None] = {}
        for inp in inputs:
            img_id = id(inp.image)
            if img_id not in seen:
                _, pred = self.models.classifier().predict_with_gradcam(inp.image, class_name)
                seen[img_id] = pred.attention_map
            inp.original_attention_map = seen[img_id]

    def _make_edit_context(self, class_name: str, class_dir: Path, instr) -> EditContext:
        """Create an EditContext for one edit operation."""
        prefix = f"{'pos' if instr.target == 'positive' else 'neg'}_{instr.image_index}"
        return EditContext(class_name, class_dir, prefix, edit_idx=0, iteration=0)

    def _test_single_edit(self, inp: EditInput, ctx: EditContext) -> EditResult | None:
        """Test one edit hypothesis across multiple generations."""
        orig_path = str((ctx.class_dir / f"{ctx.prefix}_original.jpg").resolve())
        orig_gradcam_path = self._save_original_gradcam(inp, ctx)
        generations = self._generate_edit_variants(inp, ctx)
        if not generations:
            return None
        result = self._build_edit_result(inp.instruction, inp.original_confidence, orig_path, generations)
        result.original_gradcam_path = orig_gradcam_path
        return result

    def _save_original_gradcam(self, inp: EditInput, ctx: EditContext) -> str:
        """Save original image's Grad-CAM overlay and return the path."""
        if not self.cfg.compute_edit_gradcam or inp.original_attention_map is None:
            return ""
        generator = self.models.classifier()._attention_generator
        if not generator:
            return ""
        overlay = generator.overlay_on_image(inp.original_attention_map, inp.image)
        path = ctx.class_dir / f"{ctx.prefix}_edit_{ctx.edit_idx}_orig_gradcam.jpg"
        overlay.save(path)
        return str(path)

    def _generate_edit_variants(self, inp: EditInput, ctx: EditContext) -> list[GenerationResult]:
        """Generate N edited versions with different seeds."""
        ctx.base_seed = (ctx.iteration * 1000 + ctx.edit_idx) * 100
        generations = []
        for gen_idx in tqdm(
            range(self.cfg.generations_per_edit),
            desc="      Generations", leave=False, unit="gen",
        ):
            gen = self._generate_one_variant(inp, ctx, gen_idx)
            if gen:
                generations.append(gen)
        return generations

    def _generate_one_variant(
        self, inp: EditInput, ctx: EditContext, gen_idx: int,
    ) -> GenerationResult | None:
        """Edit one image, classify it, save, and return the result."""
        seed = ctx.base_seed + gen_idx
        try:
            edited_img = self.models.editor().edit(inp.image, inp.instruction.edit, seed=seed)
            path = ctx.class_dir / f"{ctx.prefix}_iter{ctx.iteration}_edit_{ctx.edit_idx}_gen_{gen_idx}.jpg"
            edited_img.save(path)
            new_conf, gradcam_path, diff_path = self._classify_edited_image(
                edited_img, inp, ctx, gen_idx,
            )
            delta = round(new_conf - inp.original_confidence, 4)
            return GenerationResult(
                seed=seed, edited_confidence=new_conf,
                delta=delta, edited_image_path=str(path),
                gradcam_image_path=gradcam_path,
                gradcam_diff_path=diff_path,
            )
        except torch.cuda.OutOfMemoryError:
            self.models.offload_all()
            return None
        except Exception as e:
            logger.warning("Edit failed: %s", e)
            return None

    def _generate_and_save_variant(
        self, inp: EditInput, ctx: EditContext, gen_idx: int,
    ) -> PendingGeneration | None:
        """Editor-only: generate edited image and save to disk."""
        seed = ctx.base_seed + gen_idx
        try:
            editor = self.models._editor
            edited_img = editor.edit(inp.image, inp.instruction.edit, seed=seed)
            path = ctx.class_dir / f"{ctx.prefix}_iter{ctx.iteration}_edit_{ctx.edit_idx}_gen_{gen_idx}.jpg"
            edited_img.save(path)
            return PendingGeneration(
                edit_input_idx=ctx.edit_idx, seed=seed,
                edited_image_path=str(path), gen_idx=gen_idx,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.warning("Edit generation failed: %s", e)
            return None

    def _classify_saved_variant(
        self, pending: PendingGeneration, inp: EditInput, ctx: EditContext,
    ) -> GenerationResult | None:
        """Classifier-only: classify a previously saved edited image."""
        try:
            edited_img = Image.open(pending.edited_image_path).convert("RGB")
            new_conf, gradcam_path, diff_path = self._classify_edited_image(
                edited_img, inp, ctx, pending.gen_idx,
            )
            delta = round(new_conf - inp.original_confidence, 4)
            return GenerationResult(
                seed=pending.seed, edited_confidence=new_conf,
                delta=delta, edited_image_path=pending.edited_image_path,
                gradcam_image_path=gradcam_path,
                gradcam_diff_path=diff_path,
            )
        except Exception as e:
            logger.warning("Classification of saved variant failed: %s", e)
            return None

    def _classify_edited_image(
        self, edited_img: Image.Image, inp: EditInput,
        ctx: EditContext, gen_idx: int,
    ) -> tuple[float, str, str]:
        """Classify edited image, optionally computing Grad-CAM and diff."""
        if not self.cfg.compute_edit_gradcam:
            conf = self.models.classifier().get_class_confidence(edited_img, ctx.class_name)
            return conf, "", ""
        conf, pred = self.models.classifier().predict_with_gradcam(edited_img, ctx.class_name)
        gradcam_path = self._save_edit_gradcam(pred, ctx, gen_idx)
        diff_path = self._save_gradcam_diff(inp, pred, edited_img, ctx, gen_idx)
        return conf, gradcam_path, diff_path

    def _save_edit_gradcam(self, pred, ctx: EditContext, gen_idx: int) -> str:
        """Save edited image's Grad-CAM overlay, return path."""
        if not pred.gradcam_image:
            return ""
        path = ctx.class_dir / f"{ctx.prefix}_edit_{ctx.edit_idx}_gen_{gen_idx}_gradcam.jpg"
        pred.gradcam_image.save(path)
        return str(path)

    def _save_gradcam_diff(
        self, inp: EditInput, pred, edited_img: Image.Image,
        ctx: EditContext, gen_idx: int,
    ) -> str:
        """Compute and save diff heatmap, return path."""
        if inp.original_attention_map is None or pred.attention_map is None:
            return ""
        diff = compute_attention_diff(inp.original_attention_map, pred.attention_map)
        diff_img = render_diff_heatmap(diff, edited_img)
        path = ctx.class_dir / f"{ctx.prefix}_edit_{ctx.edit_idx}_gen_{gen_idx}_diff.jpg"
        diff_img.save(path)
        return str(path)

    def _expected_direction(self, instr: EditInstruction) -> str:
        """Determine expected direction of confidence change."""
        if instr.target == "positive" and instr.type == "feature_removal":
            return "negative"
        if instr.target == "negative" or instr.type == "feature_addition":
            return "positive"
        return "any"

    def _build_edit_result(self, instr, orig_conf: float, orig_path: str, generations: list[GenerationResult]) -> EditResult:
        """Assemble final EditResult from components."""
        deltas = [g.delta for g in generations]
        mean_delta = round(sum(deltas) / len(deltas), 4)
        std_delta = round(float(np.std(deltas, ddof=1)), 4) if len(deltas) > 1 else 0.0
        mean_conf = round(sum(g.edited_confidence for g in generations) / len(generations), 4)
        is_likely_failed = abs(mean_delta) < self.cfg.min_meaningful_delta
        count = sum(1 for g in generations if self._validate_direction(instr, g.delta))
        sig = self._compute_significance(deltas, instr, is_likely_failed, count)

        return EditResult(
            instruction=instr.edit, hypothesis=instr.hypothesis,
            edit_type=instr.type, target_type=instr.target,
            priority=instr.priority, original_confidence=orig_conf,
            original_image_path=orig_path, generations=generations,
            mean_edited_confidence=mean_conf,
            mean_delta=mean_delta, std_delta=std_delta,
            min_delta=round(min(deltas), 4), max_delta=round(max(deltas), 4),
            likely_failed=is_likely_failed, **sig,
        )

    def _compute_significance(self, deltas, instr, is_likely_failed, count) -> dict:
        """Compute statistical and practical significance of an edit."""
        direction = self._expected_direction(instr)
        if self.cfg.use_statistical_validation and len(deltas) >= 2:
            return self._statistical_sig(deltas, direction, is_likely_failed, count)
        return self._threshold_sig(deltas, is_likely_failed, count)

    def _statistical_sig(self, deltas, direction, is_likely_failed, count) -> dict:
        """Compute significance using t-test and Cohen's d."""
        stat = self.stat_validator.validate(deltas, direction)
        return {
            "confirmed": stat.confirmed and not is_likely_failed,
            "confirmation_count": count,
            "p_value": round(stat.p_value, 4),
            "cohens_d": round(stat.cohens_d, 2),
            "effect_size": stat.effect_size_interpretation,
            "statistically_significant": stat.statistically_significant,
            "practically_significant": stat.practically_significant,
        }

    def _threshold_sig(self, deltas, is_likely_failed, count) -> dict:
        """Compute significance using simple threshold check."""
        mean_delta = sum(deltas) / len(deltas)
        return {
            "confirmed": (count > len(deltas) / 2) and not is_likely_failed,
            "confirmation_count": count,
            "p_value": 1.0, "cohens_d": 0.0, "effect_size": "unknown",
            "statistically_significant": False,
            "practically_significant": abs(mean_delta) >= self.cfg.confidence_delta_threshold,
        }

    # =========================================================================
    # FINAL ANALYSIS
    # =========================================================================

    def _analyze_results(
        self, class_name: str, result: ClassAnalysisResult, images: ImageSet,
    ) -> FinalAnalysis | None:
        """Classify features and produce final VLM analysis."""
        if not result.edit_results or not images.positives:
            return None
        self._classify_features(class_name, result.edit_results)
        return self._run_final_analysis(class_name, result, images.positives[0][0])

    def _classify_features(self, class_name: str, edit_results: list[EditResult]):
        """Have VLM classify each feature as intrinsic/contextual."""
        self._status("Classifying features...")
        vlm = self.models.vlm()
        features = [{"instruction": er.instruction, "hypothesis": er.hypothesis} for er in edit_results]
        classified = vlm.classify_features(class_name, features)
        for i, er in enumerate(edit_results):
            if i < len(classified):
                er.feature_type = classified[i].get("feature_type", "")
                er.feature_name = classified[i].get("feature_name", "")

    def _run_final_analysis(
        self, class_name: str, result: ClassAnalysisResult, base_image: Image.Image,
    ) -> FinalAnalysis | None:
        """Have VLM produce final robustness assessment."""
        self.models.vlm()
        final = self._run_final_analysis_inner(class_name, result, base_image)
        self.models.offload_vlm()
        return final

    def _run_final_analysis_inner(
        self, class_name: str, result: ClassAnalysisResult, base_image: Image.Image,
    ) -> FinalAnalysis | None:
        """Final analysis without model lifecycle management."""
        self._status("Final analysis...")
        summary = self._build_analysis_summary(result.edit_results, result.detected_features)
        try:
            return self.models._vlm.final_analysis(base_image, class_name, summary)
        except Exception as e:
            logger.warning("Final analysis failed: %s", e)
            return None

    def _build_analysis_summary(
        self, edit_results: list[EditResult], detected_features: list[dict],
    ) -> list[dict]:
        """Build feature analysis summary for VLM final analysis."""
        feature_types = {
            f.get("name", "").lower(): f.get("feature_type", "unknown")
            for f in detected_features
        }
        results = []
        for er in edit_results:
            words = er.instruction.lower().split()
            feature = words[1] if len(words) > 1 else "unknown"
            feat_type = self._match_feature_type(feature, feature_types)
            results.append({
                "feature": feature, "edit": er.instruction,
                "delta": er.mean_delta, "confirmed": er.confirmed,
                "feature_type": feat_type,
            })
        return results

    def _match_feature_type(self, feature: str, feature_types: dict) -> str:
        """Match a feature word to its type from detected features."""
        for fname, ftype in feature_types.items():
            if feature in fname or fname in feature:
                return ftype
        return "unknown"

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _status(self, msg: str):
        """Update progress bar status message."""
        if self._pbar:
            self._pbar.set_postfix_str(msg, refresh=True)

    def _deduplicate_inputs(self, inputs: list[EditInput]) -> list[EditInput]:
        """Remove duplicate edit instructions within the same image."""
        if not inputs:
            return inputs
        by_image = self._group_inputs_by_image(inputs)
        deduplicated = self._dedup_per_image(by_image)
        if len(deduplicated) < len(inputs):
            logger.debug("Deduplicated %d -> %d edits (per-image)", len(inputs), len(deduplicated))
        return deduplicated

    def _group_inputs_by_image(self, inputs: list[EditInput]) -> dict[int, list[EditInput]]:
        """Group edit inputs by their source image index."""
        by_image: dict[int, list[EditInput]] = {}
        for inp in inputs:
            by_image.setdefault(inp.instruction.image_index, []).append(inp)
        return by_image

    def _dedup_per_image(self, by_image: dict[int, list[EditInput]]) -> list[EditInput]:
        """Deduplicate edits within each image group."""
        threshold = self.cfg.dedup_similarity_threshold
        deduplicated = []
        for group in by_image.values():
            seen: list[str] = []
            for inp in group:
                norm = self._normalize_edit_text(inp.instruction.edit)
                is_dup = any(
                    SequenceMatcher(None, norm, s).ratio() >= threshold
                    for s in seen
                )
                if not is_dup:
                    deduplicated.append(inp)
                    seen.append(norm)
        return deduplicated

    @staticmethod
    def _normalize_edit_text(text: str) -> str:
        """Normalize edit instruction text for comparison."""
        return ' '.join(re.sub(r'[^\w\s]', '', text.lower()).split())

    def _validate_direction(self, instr: EditInstruction, delta: float) -> bool:
        """Check if confidence change matches expected direction."""
        threshold = self.cfg.confidence_delta_threshold
        if instr.target == "positive":
            if instr.type == "feature_removal":
                return delta <= -threshold
            if instr.type == "feature_addition":
                return delta >= threshold
            return abs(delta) >= threshold
        return delta >= threshold

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _make_class_dir(self, class_name: str) -> Path:
        """Create and return the output directory for a class."""
        class_dir = (self.cfg.output_dir / class_name.replace(" ", "_")).resolve()
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir

    def _checkpoint_path(self, class_name: str) -> Path:
        """Get path to checkpoint file for a class."""
        return self.cfg.output_dir / class_name.replace(" ", "_") / "analysis.json"

    def _save_checkpoint(self, result: ClassAnalysisResult, class_dir: Path):
        """Save analysis result to JSON checkpoint."""
        path = class_dir / "analysis.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    def _load_checkpoint(self, class_name: str) -> ClassAnalysisResult | None:
        """Load existing analysis from checkpoint. Returns None if not found."""
        path = self._checkpoint_path(class_name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            logger.info("Loaded checkpoint for '%s'", class_name)
            return ClassAnalysisResult.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return None

    def _log_completion(self, result: ClassAnalysisResult):
        """Log summary after completing a class analysis."""
        confirmed = len(result.confirmed_hypotheses)
        total = len(result.edit_results)
        icon = {"LOW": "~", "MEDIUM": "!", "HIGH": "X"}.get(result.risk_level, "?")
        self._status(f"Done: {icon} {result.risk_level} | {confirmed}/{total} shortcuts")
        logger.debug("COMPLETE: %s - %d shortcuts, risk=%s",
                     result.class_name, confirmed, result.risk_level)
        _flush_logs()
