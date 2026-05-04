"""
Pipeline V2 — Concept-based bias/shortcut discovery.

Flow:
1. Sample Collection — positive + negative images
2. Baseline Classification — classify, Grad-CAM, find confused classes
3. Feature Discovery — 3 VLM passes: target, negative, environmental concepts
4. Feature Dedup — merge duplicate concepts across images
5. Edit Generation — one VLM call per unique feature per edit type
6. Edit Mapping — pair edits with images that have the feature
7. Image Editing — run editor on each pair
8. Impact Measurement — re-classify with ALL models
9. Feature Verdict — VLM judges essential/spurious/state_bias
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .__version__ import __version__
from .analysis import StatisticalValidator
from .config import Config
from .model_manager import ModelManager
from .vlm import FeatureConcept, VerdictContext

logger = logging.getLogger(__name__)


def _tqdm_write(msg: str):
    """Print a message without breaking tqdm progress bars."""
    tqdm.write(msg)


def _build_feature_summary(name: str, results: list) -> dict:
    """Build majority-vote summary for one feature across all test images."""
    from collections import Counter
    per_model_votes: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for model, v in r.verdict.items():
            verdict = v.get("verdict", "") if isinstance(v, dict) else ""
            if verdict and verdict not in ("not_significant", "edit_failed"):
                per_model_votes[model].append(verdict)
    majority = {}
    for model, votes in per_model_votes.items():
        counts = Counter(votes)
        winner, count = counts.most_common(1)[0]
        majority[model] = {
            "verdict": winner,
            "agreement": f"{count}/{len(votes)}",
            "conflicted": count < len(votes),
        }
    return {
        "feature_name": name,
        "feature_type": results[0].edit.feature_type if results else "",
        "test_count": len(results),
        "per_model_majority": majority,
    }


def _images_too_similar(original: Image.Image, edited: Image.Image, threshold: float = 0.99) -> bool:
    """Return True if edited image is nearly identical to original."""
    orig_arr = np.asarray(original.convert("RGB").resize((256, 256))).astype(np.float32)
    edit_arr = np.asarray(edited.convert("RGB").resize((256, 256))).astype(np.float32)
    mse = np.mean((orig_arr - edit_arr) ** 2)
    max_mse = 255.0 ** 2
    similarity = 1.0 - (mse / max_mse)
    return similarity >= threshold


def _obvious_verdict(feature_type, edit_type, target, delta, threshold=0.10) -> dict | None:
    """Return verdict for clear-cut cases, None if ambiguous."""
    neg_threshold = -threshold
    # Target feature removal + big drop = essential
    if feature_type == "target" and edit_type == "feature_removal" and delta < neg_threshold:
        return {"verdict": "essential", "reasoning": "body part removal drops confidence"}
    # Environment removal + big drop = spurious
    if feature_type == "environmental" and edit_type == "environment_removal" and delta < neg_threshold:
        return {"verdict": "spurious", "reasoning": "background removal drops confidence"}
    # Environment change + big drop = spurious
    if feature_type == "environmental" and edit_type == "environment_change" and delta < neg_threshold:
        return {"verdict": "spurious", "reasoning": "background change drops confidence"}
    # Negative edit + big increase = spurious
    if target == "negative" and delta > threshold:
        return {"verdict": "spurious", "reasoning": "non-target image fooled by edit"}
    # State change on target + big drop = state_bias
    if feature_type == "target" and edit_type == "state_change" and delta < neg_threshold:
        return {"verdict": "state_bias", "reasoning": "model requires specific feature state"}
    return None


def _open_grad(rec):
    """Open gradcam image for a record, or None if not present."""
    if rec.gradcam_path:
        try:
            return Image.open(rec.gradcam_path)
        except Exception:
            return None
    return None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImageRecord:
    """An image with classification context."""
    image: Image.Image
    path: str
    label: str
    confidence: float
    target_type: str  # "positive" | "negative"
    gradcam_path: str = ""
    attention_map: np.ndarray | None = field(default=None, repr=False)
    concepts: list[FeatureConcept] = field(default_factory=list)


@dataclass
class EditPlan:
    """One edit to apply: ties a feature concept to an edit instruction."""
    feature_name: str
    feature_type: str     # "target" | "negative" | "environmental"
    edit_type: str         # feature_removal | state_change | environment_removal | etc.
    edit_instruction: str  # actual instruction for image editor
    target: str            # "positive" | "negative"


@dataclass
class EditResultV2:
    """Result of applying one edit to one image."""
    edit: EditPlan
    image_record: ImageRecord
    edited_image_path: str
    per_model: dict = field(default_factory=dict)  # model_name -> ModelMeasurement
    verdict: dict = field(default_factory=dict)     # model_name -> verdict string
    edit_failed: bool = False                       # True if edit produced no visible change


@dataclass
class ModelMeasurement:
    """Classifier measurement for one edit-image pair."""
    original_confidence: float
    edited_confidence: float
    delta: float


@dataclass
class PreCompletedResult:
    """Wraps a previously-saved checkpoint dict for re-emission."""
    class_name: str
    data: dict

    def to_dict(self) -> dict:
        return self.data


@dataclass
class ClassResultV2:
    """Complete analysis for one class."""
    class_name: str
    concepts: list[FeatureConcept] = field(default_factory=list)
    unique_concepts: list[FeatureConcept] = field(default_factory=list)
    edit_plans: list[EditPlan] = field(default_factory=list)
    edit_results: list[EditResultV2] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)
    confusing_classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "class_name": self.class_name,
            "concepts": [asdict(c) for c in self.concepts],
            "unique_concepts": [asdict(c) for c in self.unique_concepts],
            "edit_plans": [asdict(e) for e in self.edit_plans],
            "edit_results": [self._result_to_dict(r) for r in self.edit_results],
            "feature_summaries": self._aggregate_features(),
            "confusing_classes": self.confusing_classes,
            "summary": self._summary(),
        }

    def _aggregate_features(self) -> list[dict]:
        """Aggregate verdicts per feature across all images for majority vote."""
        groups: dict[str, list[EditResultV2]] = defaultdict(list)
        for r in self.edit_results:
            groups[r.edit.feature_name].append(r)
        summaries = []
        for fname, results in groups.items():
            summaries.append(_build_feature_summary(fname, results))
        return summaries

    def _result_to_dict(self, r: EditResultV2) -> dict:
        return {
            "feature_name": r.edit.feature_name,
            "feature_type": r.edit.feature_type,
            "edit_type": r.edit.edit_type,
            "edit_instruction": r.edit.edit_instruction,
            "target": r.edit.target,
            "original_image": r.image_record.path,
            "gradcam_image": r.image_record.gradcam_path,
            "edited_image": r.edited_image_path,
            "per_model": {
                name: asdict(m) for name, m in r.per_model.items()
            },
            "verdict": r.verdict,
            "edit_failed": r.edit_failed,
        }

    def _summary(self) -> dict:
        total = len(self.edit_results)
        spurious = sum(
            1 for r in self.edit_results
            for v in r.verdict.values()
            if isinstance(v, dict) and v.get("verdict") == "spurious"
        )
        return {
            "total_concepts": len(self.unique_concepts),
            "total_edits": total,
            "total_spurious": spurious,
        }


# =============================================================================
# PIPELINE V2
# =============================================================================

class PipelineV2:
    """Concept-based bias discovery pipeline."""

    def __init__(self, config: Config):
        self.cfg = config
        self.models = ModelManager(config)
        self.stat_validator = StatisticalValidator(
            alpha=config.statistical_alpha,
            min_effect_size=config.min_effect_size,
            min_samples=2,
        )
        # Inter-phase state populated by the new probe phases.
        self._catalog: dict | None = None
        self._manifest: dict | None = None

    # Per-batch phases: re-run for each batch of classes
    _PHASES_PER_BATCH = [
        ("Classifier: baseline",     "_phase_classify"),
        ("VLM: discovery",           "_phase_discover"),
        ("Dedup + edit generation",  "_phase_generate_edits"),
        ("Editor: generate images",  "_phase_edit_images"),
        ("Classifier: measure",      "_phase_measure"),
        ("VLM: verdict",             "_phase_verdict"),
    ]
    # Final phases: run once at end on union of (resumed + new) results
    _PHASES_FINAL = [
        ("Feature catalog",          "_phase_feature_catalog"),
        ("Probe: generate",          "_phase_probe_generate"),
        ("Probe: evaluate",          "_phase_probe_evaluate"),
    ]

    def run(self, classes: list[str]) -> list:
        """Run pipeline with resume + batched checkpointing."""
        t0 = time.time()
        completed = self._load_completed_checkpoints(classes) if self.cfg.resume else {}
        todo = [c for c in classes if c not in completed]
        self._log_resume_status(classes, completed, todo)

        new_by_class = self._process_batches(todo)
        all_results = self._merge_results(classes, completed, new_by_class)
        self._run_final_phases(all_results)

        logger.debug("PIPELINE DONE: %.1fs (%d resumed, %d new)",
                     time.time() - t0, len(completed), len(new_by_class))
        return all_results

    def run_multi_model(self, classes: list[str]) -> list:
        """Run with all models (measurement phase handles multi-model)."""
        return self.run(classes)

    def _process_batches(self, todo: list[str]) -> dict:
        """Run per-batch phases on TODO classes; return new ClassResultV2 by name."""
        if not todo:
            return {}
        batch_size = self.cfg.batch_size if self.cfg.batch_size > 0 else len(todo)
        new_by_class: dict[str, ClassResultV2] = {}
        n_batches = (len(todo) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = todo[bi * batch_size : (bi + 1) * batch_size]
            logger.info("BATCH %d/%d: %d classes", bi + 1, n_batches, len(batch))
            results = self._run_one_batch(batch)
            for r in results:
                new_by_class[r.class_name] = r
        return new_by_class

    def _run_one_batch(self, batch: list[str]) -> list[ClassResultV2]:
        """Run all per-batch phases on one batch and checkpoint each result."""
        results = [ClassResultV2(class_name=c) for c in batch]
        self._run_per_batch_phases(results)
        for r in results:
            self._save_checkpoint(r)
        return results

    def _run_per_batch_phases(self, results: list[ClassResultV2]) -> None:
        """Iterate per-batch phases on the given results."""
        phases = tqdm(
            self._PHASES_PER_BATCH, desc="Pipeline", unit="phase",
            leave=True, bar_format="{desc} |{bar:15}| {n}/{total} {postfix}",
        )
        for phase_name, method_name in phases:
            phases.set_postfix_str(phase_name)
            step_t0 = time.time()
            try:
                getattr(self, method_name)(results)
            except Exception as e:
                _tqdm_write(f"[ERROR] {phase_name}: {e}")
                logger.error("Phase '%s' failed: %s", phase_name, e, exc_info=True)
            logger.debug("PHASE %s: %.1fs", phase_name, time.time() - step_t0)
        phases.close()

    def _run_final_phases(self, all_results: list) -> None:
        """Run feature catalog + probe phases once on the full result set."""
        for phase_name, method_name in self._PHASES_FINAL:
            step_t0 = time.time()
            try:
                getattr(self, method_name)(all_results)
            except Exception as e:
                _tqdm_write(f"[ERROR] {phase_name}: {e}")
                logger.error("Final phase '%s' failed: %s", phase_name, e, exc_info=True)
            logger.debug("FINAL PHASE %s: %.1fs", phase_name, time.time() - step_t0)

    def _log_resume_status(self, classes: list[str], completed: dict, todo: list[str]) -> None:
        """Print resume summary at start of run."""
        if completed:
            logger.info("Resume: %d/%d classes already complete, %d to do",
                        len(completed), len(classes), len(todo))
            _tqdm_write(f"[RESUME] {len(completed)}/{len(classes)} classes loaded "
                        f"from checkpoints, {len(todo)} remaining")

    def _merge_results(self, classes: list[str], completed: dict, new_by_class: dict) -> list:
        """Return results in original class order, using checkpoint dicts where available."""
        return [completed.get(c) or new_by_class.get(c) for c in classes if completed.get(c) or new_by_class.get(c)]

    def _load_completed_checkpoints(self, classes: list[str]) -> dict:
        """Load complete checkpoints for any class that has one. Returns {class_name: PreCompletedResult}."""
        ckpt_dir = self.cfg.model_checkpoint_dir(self.cfg.classifier_model)
        if not ckpt_dir.exists():
            return {}
        completed: dict[str, PreCompletedResult] = {}
        for c in classes:
            data = self._read_checkpoint_file(ckpt_dir, c)
            if data is not None and self._is_complete_checkpoint(data):
                completed[c] = PreCompletedResult(class_name=c, data=data)
        return completed

    def _read_checkpoint_file(self, ckpt_dir: Path, class_name: str) -> dict | None:
        """Read one checkpoint JSON, returning None on missing/corrupt."""
        path = ckpt_dir / f"{class_name.replace(' ', '_')}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning("Bad checkpoint for %s: %s", class_name, e)
            return None

    def _is_complete_checkpoint(self, data: dict) -> bool:
        """True if checkpoint represents a fully-completed class (passed all per-batch phases)."""
        summary = data.get("summary", {})
        return bool(summary) and "total_concepts" in summary

    # =========================================================================
    # PHASE 1: CLASSIFIER — sample + baseline (load classifier once)
    # =========================================================================

    def _phase_classify(self, results: list[ClassResultV2]):
        """Sample and classify all classes with one classifier load."""
        clf = self.models.classifier()
        sampler = self.models.sampler()
        pbar = tqdm(results, desc="  Classifying", unit="class", leave=True)
        for result in pbar:
            pbar.set_description(f"  Classifying [{result.class_name[:20]}]")
            try:
                self._step_sample_classify(result, clf, sampler)
            except Exception as e:
                _tqdm_write(f"  [ERROR] {result.class_name}: {e}")
                logger.error("Classify failed for %s: %s", result.class_name, e)
        pbar.close()
        self.models.offload_classifier()

    # =========================================================================
    # PHASE 2: VLM — discovery (load VLM once, flatten all images)
    # =========================================================================

    def _phase_discover(self, results: list[ClassResultV2]):
        """Discover features for all classes — batched VLM calls."""
        vlm = self.models.vlm()
        bs = max(1, int(getattr(self.cfg, "vlm_batch_size", 4)))
        positives, negatives = self._gather_discovery_inputs(results)
        if positives:
            self._batched_discover_positives(vlm, positives, bs)
        if negatives:
            self._batched_discover_negatives(vlm, negatives, bs)
        self.models.offload_vlm()

    def _gather_discovery_inputs(self, results):
        """Split images into (positives_with_refs, negatives_with_refs)."""
        positives = []
        negatives = []
        for r in results:
            for i, rec in enumerate(r.images):
                if rec.target_type == "positive":
                    positives.append((r, rec, i))
                else:
                    negatives.append((r, rec, i))
        return positives, negatives

    def _batched_discover_positives(self, vlm, positives, bs):
        """Run target + env discovery for all positive images, batched."""
        target_items = [(rec.image, _open_grad(rec), r.class_name, i) for (r, rec, i) in positives]
        env_items = [(rec.image, r.class_name, i) for (r, rec, i) in positives]
        pbar = tqdm(total=len(positives) * 2, desc="  Discovering [pos]", unit="call", leave=True)
        target_lists = vlm.discover_target_features_batch(target_items, batch_size=bs)
        pbar.update(len(positives))
        env_lists = vlm.discover_environmental_features_batch(env_items, batch_size=bs)
        pbar.update(len(positives))
        pbar.close()
        for (r, rec, _), tlist, elist in zip(positives, target_lists, env_lists):
            for concepts in (tlist, elist):
                rec.concepts.extend(concepts)
                r.concepts.extend(concepts)

    def _batched_discover_negatives(self, vlm, negatives, bs):
        """Run negative-feature discovery for all negative images, batched."""
        items = [(rec.image, _open_grad(rec), r.class_name, rec.label, rec.confidence, i)
                 for (r, rec, i) in negatives]
        pbar = tqdm(total=len(negatives), desc="  Discovering [neg]", unit="call", leave=True)
        neg_lists = vlm.discover_negative_features_batch(items, batch_size=bs)
        pbar.update(len(negatives))
        pbar.close()
        for (r, rec, _), concepts in zip(negatives, neg_lists):
            rec.concepts.extend(concepts)
            r.concepts.extend(concepts)

    # =========================================================================
    # PHASE 3: DEDUP + EDIT GEN (VLM loaded once, flatten all features)
    # =========================================================================

    def _phase_generate_edits(self, results: list[ClassResultV2]):
        """Dedup features and generate edits for all classes."""
        vlm = self.models.vlm()
        # First dedup all classes
        for result in results:
            self._step_deduplicate_features(result)
        # Flatten: all (result, concept, edit_type) across all classes
        all_tasks = self._collect_edit_tasks(results)
        pbar = tqdm(all_tasks, desc="  Generating edits", unit="feat", leave=True)
        for result, concept, edit_type, image_rec in pbar:
            pbar.set_description(f"  Edits [{result.class_name[:15]}:{concept.name[:10]}]")
            self._gen_edits(vlm, result, image_rec, concept, edit_type)
        pbar.close()
        self.models.offload_vlm()

    def _collect_edit_tasks(self, results):
        """Flatten all (result, concept, edit_type, sample_image) tuples."""
        tasks = []
        for result in results:
            pos = self._first_image(result, "positive")
            neg = self._first_image(result, "negative")
            target = [c for c in result.unique_concepts if c.type == "target"]
            env = [c for c in result.unique_concepts if c.type == "environmental"]
            negc = [c for c in result.unique_concepts if c.type == "negative"]
            if pos:
                for c in target:
                    tasks.append((result, c, "feature_removal", pos))
                    tasks.append((result, c, "state_change", pos))
                for c in env:
                    tasks.append((result, c, "environment_removal", pos))
                    tasks.append((result, c, "environment_change", pos))
            if neg:
                for c in negc:
                    tasks.append((result, c, "negative_modify", neg))
                for c in env:
                    tasks.append((result, c, "negative_environment_add", neg))
        return tasks

    # =========================================================================
    # PHASE 4: EDITOR — generate edited images (load editor once)
    # =========================================================================

    def _phase_edit_images(self, results: list[ClassResultV2]):
        """Apply all edits for all classes with one editor load (batched)."""
        editor = self.models.editor()
        all_pairs = self._collect_all_edit_pairs(results)
        if not all_pairs:
            self.models.offload_editor()
            return
        bs = max(1, int(getattr(self.cfg, "edit_batch_size", 8)))
        n = len(all_pairs)
        pbar = tqdm(total=n, desc="  Editing", unit="edit", leave=True)
        for start in range(0, n, bs):
            chunk = all_pairs[start:start + bs]
            self._edit_chunk(editor, chunk, pbar)
        pbar.close()
        self.models.offload_editor()

    def _edit_chunk(self, editor, chunk, pbar):
        """Run one batched edit and persist outputs/failures."""
        imgs = [c[2].image for c in chunk]
        prompts = [c[1].edit_instruction for c in chunk]
        first_class = chunk[0][0].class_name[:20]
        pbar.set_description(f"  Editing [{first_class}]")
        try:
            edited_list = editor.edit_batch(imgs, prompts, seed=42)
        except Exception as e:
            _tqdm_write(f"  [WARN] batch edit failed: {e}; falling back to per-image")
            edited_list = [None] * len(chunk)
        for (result, plan, image_rec, idx), edited in zip(chunk, edited_list):
            self._persist_edit(result, plan, image_rec, idx, edited, editor)
            pbar.update(1)

    def _persist_edit(self, result, plan, image_rec, idx, edited, editor):
        """Save one edited image and record the result."""
        class_dir = self._make_class_dir(result.class_name)
        safe_name = plan.feature_name.replace(" ", "_")[:20]
        edit_path = class_dir / f"{image_rec.target_type}_{safe_name}_{plan.edit_type}_{idx}.jpg"
        if edited is None:
            try:
                edited = editor.edit(image_rec.image, plan.edit_instruction)
            except Exception as e:
                _tqdm_write(f"  [WARN] Edit failed: {plan.feature_name}: {e}")
                logger.warning("Edit failed: %s — %s", plan.edit_instruction[:50], e)
                return
        edited.save(edit_path)
        failed = _images_too_similar(image_rec.image, edited)
        if failed:
            _tqdm_write(f"  [WARN] Edit unchanged: {plan.feature_name}")
        result.edit_results.append(EditResultV2(
            edit=plan, image_record=image_rec,
            edited_image_path=str(edit_path), edit_failed=failed,
        ))

    def _collect_all_edit_pairs(self, results):
        """Flatten all edit-image pairs across all classes."""
        all_pairs = []
        idx = 0
        for result in results:
            for plan, image_rec in self._map_edits_to_images(result):
                all_pairs.append((result, plan, image_rec, idx))
                idx += 1
        return all_pairs

    def _map_edits_to_images(self, result):
        """Pair each edit plan with images that have the matching feature."""
        pairs = []
        for plan in result.edit_plans:
            matching = [
                rec for rec in result.images
                if rec.target_type == plan.target
                and any(c.name.lower() == plan.feature_name.lower() for c in rec.concepts)
            ]
            if not matching:
                matching = [r for r in result.images if r.target_type == plan.target]
            for rec in matching[:self.cfg.samples_per_class]:
                pairs.append((plan, rec))
        return pairs

    # =========================================================================
    # PHASE 5: CLASSIFIER — measure impact (load each model once)
    # =========================================================================

    def _phase_measure(self, results: list[ClassResultV2]):
        """Measure impact for all classes with each classifier."""
        for model_name in self.cfg.classifier_models:
            clf = self.models.classifier(model_name)
            all_items = [(r, er) for r in results for er in r.edit_results]
            pbar = tqdm(all_items, desc=f"  {model_name}", unit="edit", leave=True)
            for result, er in pbar:
                pbar.set_description(f"  {model_name} [{result.class_name[:20]}]")
                try:
                    orig = Image.open(er.image_record.path).convert("RGB")
                    edited = Image.open(er.edited_image_path).convert("RGB")
                    oc = clf.get_class_confidence(orig, result.class_name)
                    ec = clf.get_class_confidence(edited, result.class_name)
                    er.per_model[model_name] = ModelMeasurement(
                        original_confidence=round(oc, 4),
                        edited_confidence=round(ec, 4),
                        delta=round(ec - oc, 4),
                    )
                except Exception as e:
                    logger.warning("Measure failed %s/%s: %s", result.class_name, model_name, e)
            pbar.close()
            self.models.offload_classifier(model_name)

    # =========================================================================
    # PHASE 6: VLM — verdict (load VLM once)
    # =========================================================================

    def _phase_verdict(self, results: list[ClassResultV2]):
        """Judge verdicts — skip VLM for obvious cases."""
        threshold = self.cfg.confidence_delta_threshold
        all_edits = [(r, er) for r in results for er in r.edit_results]

        # Mark failed edits (image unchanged)
        for result, er in all_edits:
            if er.edit_failed:
                er.verdict = {
                    n: {"verdict": "edit_failed", "reasoning": "edited image too similar to original"}
                    for n in er.per_model
                }

        # Mark non-significant edits (exclude already-failed)
        for result, er in all_edits:
            if er.edit_failed:
                continue
            if not any(abs(m.delta) >= threshold for m in er.per_model.values()):
                er.verdict = {
                    n: {"verdict": "not_significant", "reasoning": f"delta below {threshold:.0%}"}
                    for n in er.per_model
                }

        all_sig = [
            (r, er) for r, er in all_edits
            if not er.edit_failed and "not_significant" not in str(er.verdict)
        ]

        # Split into obvious (rule-based) and ambiguous (need VLM)
        need_vlm = []
        auto_count = 0
        for result, er in all_sig:
            obvious, ambiguous_models = self._auto_verdict(er)
            er.verdict = obvious
            if ambiguous_models:
                need_vlm.append((result, er, ambiguous_models))
            else:
                auto_count += 1

        logger.debug("Verdict: %d auto, %d need VLM", auto_count, len(need_vlm))

        if need_vlm:
            vlm = self.models.vlm()
            pbar = tqdm(need_vlm, desc="  Verdicts (VLM)", unit="edit", leave=True)
            for result, er, ambiguous_models in pbar:
                pbar.set_description(f"  Verdict [{result.class_name[:20]}]")
                try:
                    vlm_verdicts = self._judge_one(vlm, er, result.class_name, ambiguous_models)
                    er.verdict.update(vlm_verdicts)
                except Exception as e:
                    logger.warning("Verdict failed for %s/%s: %s", result.class_name, er.edit.feature_name, e)
                self._fill_missing_verdicts(er)
            pbar.close()
            self.models.offload_vlm()

    def _auto_verdict(self, er: EditResultV2) -> tuple[dict, list[str]]:
        """Return (obvious_verdicts, ambiguous_model_names)."""
        edit = er.edit
        threshold = self.cfg.confidence_delta_threshold
        verdicts = {}
        ambiguous = []
        for model_name, m in er.per_model.items():
            v = _obvious_verdict(edit.feature_type, edit.edit_type, edit.target, m.delta, threshold)
            if v:
                verdicts[model_name] = v
            else:
                ambiguous.append(model_name)
        return verdicts, ambiguous

    @staticmethod
    def _fill_missing_verdicts(er: EditResultV2):
        """Fill models missing from verdict dict with 'unknown'."""
        for model_name in er.per_model:
            if model_name not in er.verdict:
                er.verdict[model_name] = {
                    "verdict": "unknown",
                    "reasoning": "VLM did not return verdict",
                }

    # =========================================================================
    # STEP 1-2: SAMPLE + CLASSIFY
    # =========================================================================

    def _step_sample_classify(self, result: ClassResultV2, clf=None, sampler=None):
        """Sample images and run baseline classification with Grad-CAM."""
        class_dir = self._make_class_dir(result.class_name)
        clf = clf or self.models.classifier()
        sampler = sampler or self.models.sampler()
        self._classify_positives(result, sampler, clf, class_dir)
        result.confusing_classes = self._top_confusing(result.confusing_classes)
        logger.debug("Confusing classes for %s: %s", result.class_name, result.confusing_classes)
        self._classify_negatives(result, sampler, clf, class_dir)
        logger.debug("Sampled %d images (%d pos, %d neg)",
                     len(result.images),
                     sum(1 for r in result.images if r.target_type == "positive"),
                     sum(1 for r in result.images if r.target_type == "negative"))

    def _classify_positives(self, result, sampler, clf, class_dir):
        """Sample and classify positive images."""
        positives = sampler.sample_positive(result.class_name, n=self.cfg.samples_per_class)
        for i, (img, label) in enumerate(positives):
            path = class_dir / f"pos_{i}_original.jpg"
            img.save(path)
            pred = clf.predict(img, target_class_name=result.class_name,
                               top_k=self.cfg.top_k_classes, compute_gradcam=True)
            record = self._pred_to_record(pred, img, label, "positive", path)
            result.images.append(record)
            self._collect_confusing(pred, result)

    def _classify_negatives(self, result, sampler, clf, class_dir):
        """Sample and classify negative images."""
        if not result.confusing_classes:
            return
        neg_images = sampler.sample_from_classes(
            result.confusing_classes, n_per_class=self.cfg.negative_samples_per_class,
        )
        for i, (img, label) in enumerate(neg_images):
            conf = clf.get_class_confidence(img, result.class_name)
            if conf < self.cfg.min_negative_confidence:
                continue
            path = class_dir / f"neg_{i}_original.jpg"
            img.save(path)
            pred = clf.predict(img, target_class_name=result.class_name,
                               top_k=self.cfg.top_k_classes, compute_gradcam=True)
            record = self._pred_to_record(pred, img, label, "negative", path)
            record.confidence = conf
            result.images.append(record)

    def _pred_to_record(self, pred, img, label, target_type, img_path):
        """Convert classifier prediction to ImageRecord."""
        gc_path = ""
        if pred.gradcam_image is not None:
            gc_path = str(img_path).replace("_original.jpg", "_gradcam.jpg")
            pred.gradcam_image.save(gc_path)
        return ImageRecord(
            image=img, path=str(img_path), label=label, confidence=pred.confidence,
            target_type=target_type, gradcam_path=gc_path,
            attention_map=pred.attention_map if pred.gradcam_image else None,
        )

    def _collect_confusing(self, pred, result):
        """Extract confused classes from top-K predictions."""
        for label, conf in pred.top_k:
            if label.lower() != result.class_name.lower() and conf > self.cfg.confusing_class_min_conf:
                result.confusing_classes.append(label)

    def _top_confusing(self, class_list: list[str]) -> list[str]:
        """Return top-N most frequent confusing classes."""
        counts: dict[str, int] = {}
        for c in class_list:
            counts[c] = counts.get(c, 0) + 1
        sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
        return [c for c, _ in sorted_classes[:self.cfg.top_negative_classes]]

    # =========================================================================
    # STEP 3: FEATURE DISCOVERY
    # =========================================================================

    def _discover_positive(self, vlm, result, rec, idx):
        """Discover target + environmental features for one positive image."""
        gc = Image.open(rec.gradcam_path) if rec.gradcam_path else None
        try:
            target = vlm.discover_target_features(rec.image, gc, result.class_name, idx)
            env = vlm.discover_environmental_features(rec.image, result.class_name, idx)
            for concepts in [target, env]:
                rec.concepts.extend(concepts)
                result.concepts.extend(concepts)
            logger.debug("pos_%d: %d target + %d env features", idx, len(target), len(env))
        except Exception as e:
            _tqdm_write(f"  [WARN] Discovery failed for pos_{idx}: {e}")
            logger.warning("Discovery failed for pos_%d: %s", idx, e)

    def _discover_negative(self, vlm, result, rec, idx):
        """Discover negative features for one negative image."""
        gc = Image.open(rec.gradcam_path) if rec.gradcam_path else None
        try:
            concepts = vlm.discover_negative_features(
                rec.image, gc, result.class_name, rec.label, rec.confidence, idx,
            )
            rec.concepts.extend(concepts)
            result.concepts.extend(concepts)
            logger.debug("neg_%d (%s): %d features", idx, rec.label, len(concepts))
        except Exception as e:
            _tqdm_write(f"  [WARN] Discovery failed for neg_{idx}: {e}")
            logger.warning("Discovery failed for neg_%d: %s", idx, e)

    # =========================================================================
    # STEP 4: FEATURE DEDUPLICATION
    # =========================================================================

    def _step_deduplicate_features(self, result: ClassResultV2):
        """Merge duplicate concepts across images into unique list."""
        seen: dict[str, FeatureConcept] = {}
        for c in result.concepts:
            key = c.name.lower().strip()
            if key not in seen:
                seen[key] = c
        result.unique_concepts = list(seen.values())
        target = sum(1 for c in result.unique_concepts if c.type == "target")
        env = sum(1 for c in result.unique_concepts if c.type == "environmental")
        neg = sum(1 for c in result.unique_concepts if c.type == "negative")
        logger.debug("Dedup: %d total → %d unique (target=%d, env=%d, neg=%d)",
                     len(result.concepts), len(result.unique_concepts), target, env, neg)

    # =========================================================================
    # STEP 5-6: EDIT GENERATION + MAPPING
    # =========================================================================

    def _gen_edits(self, vlm, result, image_rec, concept, edit_type):
        """Generate edits for one feature and add to result."""
        target = "negative" if concept.type == "negative" or edit_type.startswith("negative") else "positive"
        try:
            edits = vlm.generate_edit_for_feature(
                image_rec.image, result.class_name, concept.name, edit_type,
            )
            for edit_str in edits:
                result.edit_plans.append(EditPlan(
                    feature_name=concept.name, feature_type=concept.type,
                    edit_type=edit_type, edit_instruction=edit_str, target=target,
                ))
            logger.debug("  %s/%s: %d edits", concept.name, edit_type, len(edits))
        except Exception as e:
            _tqdm_write(f"  [WARN] Edit gen failed: {concept.name}/{edit_type}: {e}")
            logger.warning("Edit gen failed for %s/%s: %s", concept.name, edit_type, e)

    def _first_image(self, result: ClassResultV2, target_type: str) -> ImageRecord | None:
        """Get first image of given type."""
        for r in result.images:
            if r.target_type == target_type:
                return r
        return None

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _judge_one(self, vlm, er: EditResultV2, class_name: str,
                   models: list[str] | None = None) -> dict:
        """Get VLM verdict for ambiguous models only."""
        ctx = VerdictContext(
            class_name=class_name, feature_name=er.edit.feature_name,
            feature_type=er.edit.feature_type, edit_type=er.edit.edit_type,
            edit_instruction=er.edit.edit_instruction,
            is_target=er.edit.target == "positive",
        )
        per_model_data = [
            {"model_name": n, "original_confidence": m.original_confidence,
             "edited_confidence": m.edited_confidence, "delta": m.delta}
            for n, m in er.per_model.items() if not models or n in models
        ]
        orig = Image.open(er.image_record.path).convert("RGB")
        edited = Image.open(er.edited_image_path).convert("RGB")
        verdict = vlm.judge_verdict(orig, edited, ctx, per_model_data)
        return verdict.get("per_model", {})

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _make_class_dir(self, class_name: str) -> Path:
        """Create and return image directory for a class."""
        class_dir = (self.cfg.images_dir / class_name.replace(" ", "_")).resolve()
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir

    def _save_checkpoint(self, result: ClassResultV2):
        """Save analysis result to checkpoint."""
        ckpt_dir = self.cfg.model_checkpoint_dir(self.cfg.classifier_model)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{result.class_name.replace(' ', '_')}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        logger.info("Saved checkpoint: %s", path)

    # =========================================================================
    # PHASE 7: FEATURE CATALOG (cheap aggregation; always runs)
    # =========================================================================

    def _phase_feature_catalog(self, results: list[ClassResultV2]):
        """Aggregate per-class real/bias features across all classifier models."""
        from src.feature_extractor import extract_catalog
        analysis = self._results_to_analysis(results)
        self._catalog = extract_catalog(analysis)
        self._save_feature_catalog()

    def _results_to_analysis(self, results: list[ClassResultV2]) -> dict:
        """Shape ClassResultV2 list into the analysis_results.json format."""
        return {
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"),
            "results": [r.to_dict() for r in results],
        }

    def _save_feature_catalog(self) -> None:
        """Write the feature catalog to disk."""
        path = self.cfg.feature_catalog_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._catalog, indent=2, default=str))
        logger.info("Saved feature catalog: %s", path)

    # =========================================================================
    # PHASE 8: PROBE GENERATION (VLM prompts + FLUX text-to-image)
    # =========================================================================

    def _phase_probe_generate(self, results: list[ClassResultV2]):
        """Build a probe image set per class using the catalog."""
        if self.cfg.skip_probes or not self._catalog:
            return
        from src.probe_generator import (
            ProbeConfig, _manifest_to_dict, generate_probes,
        )
        probe_cfg = ProbeConfig(
            n_per_variant=self.cfg.probe_n_per_variant,
            feature_source=self.cfg.probe_feature_source,
            mode=self.cfg.probe_mode,
        )
        manifest_obj = generate_probes(
            self._catalog, self.cfg.probes_dir, probe_cfg, self.models,
            source_catalog=str(self.cfg.feature_catalog_path),
        )
        self._manifest = _manifest_to_dict(manifest_obj)
        logger.info("Saved probe manifest: %s", self.cfg.probes_dir / "manifest.json")

    # =========================================================================
    # PHASE 9: PROBE EVALUATION (every classifier × every probe image)
    # =========================================================================

    def _phase_probe_evaluate(self, results: list[ClassResultV2]):
        """Run each classifier on the probe set; write 4-metric report + HTML."""
        if self.cfg.skip_probes or not self._manifest:
            return
        from src.probe_evaluator import evaluate_probes, write_report
        from src.probe_reporter import write_probe_report
        manifest_path = self.cfg.probes_dir / "manifest.json"
        model_metrics = evaluate_probes(
            self._manifest, self.cfg.classifier_models, self.models,
            manifest_dir=self.cfg.probes_dir,
        )
        write_report(model_metrics, self.cfg.probe_report_path,
                      manifest_path=str(manifest_path))
        logger.info("Saved probe evaluation: %s", self.cfg.probe_report_path)
        html_path = write_probe_report(
            manifest_path, self.cfg.probe_report_path,
            self.cfg.reports_dir / "probe_report.html",
        )
        logger.info("Saved probe HTML report: %s", html_path)
