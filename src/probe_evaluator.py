"""Evaluate probe images with every classifier and report four metrics.

Reads the manifest written by ``src.probe_generator``, runs each classifier
on every probe image, and outputs per-model metrics:
    - top1_accuracy_overall
    - top1_accuracy_bias_stripped
    - top1_accuracy_bias_heavy
    - bias_lift_score
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from .__version__ import __version__

if TYPE_CHECKING:  # pragma: no cover
    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


_VARIANTS: tuple[str, ...] = (
    "bias_heavy", "bias_stripped", "real_feature_only", "adversarial",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VariantStats:
    """Aggregated stats for one (model, class, variant) triple."""
    mean_conf: float = 0.0
    top1_accuracy: float = 0.0
    n: int = 0


@dataclass
class ClassMetrics:
    """Per-class metrics under one model."""
    class_name: str
    bias_heavy: VariantStats | None = None
    bias_stripped: VariantStats | None = None
    real_feature_only: VariantStats | None = None
    adversarial: VariantStats | None = None
    bias_lift: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ModelMetrics:
    """Top-level report for one classifier model."""
    model_name: str
    top1_accuracy_overall: float = 0.0
    mean_conf_overall: float = 0.0
    top1_accuracy_bias_stripped: float = 0.0
    top1_accuracy_bias_heavy: float = 0.0
    top1_accuracy_adversarial: float = 0.0
    bias_lift_score: float = 0.0
    n_classes_scored: int = 0
    n_classes_skipped: int = 0
    mean_conf_bias_heavy: float = 0.0
    mean_conf_bias_stripped: float = 0.0
    mean_conf_real_feature_only: float = 0.0
    mean_conf_adversarial: float = 0.0
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_probes(manifest: dict, classifier_names: list[str],
                     models: "ModelManager",
                     manifest_dir: Path | None = None) -> dict[str, ModelMetrics]:
    """Run every classifier over every probe image; return per-model metrics."""
    base_dir = Path(manifest_dir) if manifest_dir else Path.cwd()
    images_by_class = _collect_images(manifest, base_dir)
    return {
        name: _evaluate_one_model(name, images_by_class, manifest, models)
        for name in classifier_names
    }


def write_report(model_metrics: dict[str, ModelMetrics], path: Path | str,
                  manifest_path: str = "") -> None:
    """Serialize the per-model metrics to disk as JSON."""
    report = {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_manifest": manifest_path,
        "models_evaluated": list(model_metrics.keys()),
        "scores": {name: _metrics_to_dict(m) for name, m in model_metrics.items()},
    }
    Path(path).write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def _collect_images(manifest: dict, base_dir: Path) -> dict[str, list[dict]]:
    """Map class_name → list of {variant, path, target_class} dicts (existing only)."""
    out: dict[str, list[dict]] = {}
    for class_name, entry in manifest.get("classes", {}).items():
        out[class_name] = _existing_images_for_class(class_name, entry, base_dir)
    return out


def _existing_images_for_class(class_name: str, entry: dict,
                                base_dir: Path) -> list[dict]:
    """Return image metadata records whose file exists on disk."""
    records: list[dict] = []
    for image in entry.get("images", []):
        full_path = base_dir / image["image_path"]
        if full_path.exists():
            records.append({"path": full_path, "variant": image["variant"],
                             "target_class": class_name})
        else:
            logger.warning("Probe image missing: %s", full_path)
    return records


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_one_model(model_name: str,
                         images_by_class: dict[str, list[dict]],
                         manifest: dict, models: "ModelManager") -> ModelMetrics:
    """Classify every probe image with one model and aggregate metrics."""
    classifier = models.classifier(model_name)
    per_class = {
        class_name: _score_one_class(
            class_name, images, manifest["classes"].get(class_name, {}), classifier)
        for class_name, images in images_by_class.items()
    }
    models.offload_classifier(model_name)
    return _aggregate_model_metrics(model_name, per_class)


def _score_one_class(class_name: str, images: list[dict],
                      entry: dict, classifier) -> ClassMetrics:
    """Classify every image for one class; build ClassMetrics."""
    if _class_has_no_bias(entry):
        return ClassMetrics(class_name=class_name, skipped=True,
                             skip_reason="no_bias_features")
    per_variant = _classify_per_variant(images, class_name, classifier)
    if _any_variant_missing(per_variant):
        return ClassMetrics(class_name=class_name, skipped=True,
                             skip_reason="missing_variant",
                             **{v: per_variant[v] for v in _VARIANTS if per_variant[v]})
    return ClassMetrics(
        class_name=class_name,
        bias_heavy=per_variant["bias_heavy"],
        bias_stripped=per_variant["bias_stripped"],
        real_feature_only=per_variant["real_feature_only"],
        adversarial=per_variant["adversarial"],
        bias_lift=per_variant["bias_heavy"].mean_conf
            - per_variant["real_feature_only"].mean_conf,
    )


def _class_has_no_bias(entry: dict) -> bool:
    """True if the class has no biased features recorded in the manifest."""
    return not entry.get("bias_features")


def _classify_per_variant(images: list[dict], class_name: str,
                           classifier) -> dict[str, VariantStats | None]:
    """Group images by variant, classify, return variant → VariantStats."""
    by_variant: dict[str, list[dict]] = {v: [] for v in _VARIANTS}
    for rec in images:
        if rec["variant"] in by_variant:
            by_variant[rec["variant"]].append(rec)
    return {
        v: _stats_for_variant(by_variant[v], class_name, classifier)
        for v in _VARIANTS
    }


def _stats_for_variant(records: list[dict], class_name: str,
                        classifier) -> VariantStats | None:
    """Run classifier on one variant's images; return None if empty.

    Per-image classifier failures (e.g. label-set mismatch) are logged and
    skipped so one bad call does not kill the whole evaluation phase.
    """
    if not records:
        return None
    confidences, hits = [], 0
    for rec in records:
        hit, conf = _classify_one(rec, class_name, classifier)
        if conf is None:
            continue
        confidences.append(conf)
        if hit:
            hits += 1
    n = len(confidences)
    if n == 0:
        return None
    return VariantStats(
        mean_conf=sum(confidences) / n,
        top1_accuracy=hits / n,
        n=n,
    )


def _classify_one(rec: dict, class_name: str,
                    classifier) -> tuple[bool, float | None]:
    """Classify a single probe image; return (hit, confidence) or (False, None)."""
    try:
        img = Image.open(rec["path"]).convert("RGB")
        pred = classifier.predict(img, target_class_name=class_name,
                                   top_k=1, compute_gradcam=False)
        conf = classifier.get_class_confidence(img, class_name)
    except Exception as e:
        logger.warning("Probe classify failed on %s for %s: %s",
                       rec.get("path"), class_name, e)
        return False, None
    return _label_matches(pred.label_name, class_name), conf


def _label_matches(predicted: str, target: str) -> bool:
    """True if the two labels refer to the same ImageNet class.

    Classifiers may return a short label (e.g. 'goldfish') while the caller
    passes the full canonical name (e.g. 'goldfish, Carassius auratus').
    This matches the fuzzy logic already used by ``_resolve_label``.
    """
    p, t = predicted.lower().strip(), target.lower().strip()
    return p == t or p in t or t in p


def _any_variant_missing(per_variant: dict[str, VariantStats | None]) -> bool:
    """True if any of the three variants has no data."""
    return any(per_variant.get(v) is None for v in _VARIANTS)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_model_metrics(model_name: str,
                              per_class: dict[str, ClassMetrics]) -> ModelMetrics:
    """Roll per-class metrics up into per-model totals."""
    scored = [c for c in per_class.values() if not c.skipped]
    skipped = [c for c in per_class.values() if c.skipped]
    totals = _accumulate_totals(per_class)
    metrics = ModelMetrics(
        model_name=model_name,
        per_class=per_class,
        n_classes_scored=len(scored),
        n_classes_skipped=len(skipped),
    )
    _assign_aggregate_metrics(metrics, totals, scored)
    return metrics


def _accumulate_totals(per_class: dict[str, ClassMetrics]) -> dict:
    """Sum per-variant n + hits + confidence across classes (scored only)."""
    t = {v: {"n": 0, "hits": 0, "conf": 0.0} for v in _VARIANTS}
    for cm in per_class.values():
        if cm.skipped:
            continue
        for v in _VARIANTS:
            vs = getattr(cm, v)
            if vs is None:
                continue
            t[v]["n"] += vs.n
            t[v]["hits"] += round(vs.top1_accuracy * vs.n)
            t[v]["conf"] += vs.mean_conf * vs.n
    return t


def _assign_aggregate_metrics(metrics: ModelMetrics, totals: dict,
                                scored: list[ClassMetrics]) -> None:
    """Fill the headline metrics on ``metrics`` from accumulated totals."""
    metrics.top1_accuracy_bias_heavy = _safe_ratio(totals["bias_heavy"]["hits"],
                                                     totals["bias_heavy"]["n"])
    metrics.top1_accuracy_bias_stripped = _safe_ratio(totals["bias_stripped"]["hits"],
                                                        totals["bias_stripped"]["n"])
    metrics.top1_accuracy_adversarial = _safe_ratio(
        totals["adversarial"]["hits"], totals["adversarial"]["n"])
    total_n = sum(t["n"] for t in totals.values())
    total_hits = sum(t["hits"] for t in totals.values())
    total_conf = sum(t["conf"] for t in totals.values())
    metrics.top1_accuracy_overall = _safe_ratio(total_hits, total_n)
    metrics.mean_conf_overall = _safe_ratio(total_conf, total_n)
    metrics.mean_conf_bias_heavy = _safe_ratio(totals["bias_heavy"]["conf"],
                                                 totals["bias_heavy"]["n"])
    metrics.mean_conf_bias_stripped = _safe_ratio(totals["bias_stripped"]["conf"],
                                                    totals["bias_stripped"]["n"])
    metrics.mean_conf_real_feature_only = _safe_ratio(
        totals["real_feature_only"]["conf"], totals["real_feature_only"]["n"])
    metrics.mean_conf_adversarial = _safe_ratio(totals["adversarial"]["conf"],
                                                 totals["adversarial"]["n"])
    metrics.bias_lift_score = _mean([cm.bias_lift for cm in scored])


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator or 0.0 when denominator is 0."""
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 if the list is empty."""
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _metrics_to_dict(metrics: ModelMetrics) -> dict:
    """Convert ModelMetrics to a plain JSON-serializable dict."""
    return asdict(metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(description="Evaluate probe images across classifiers")
    p.add_argument("--manifest", required=True, help="Path to probes/manifest.json")
    p.add_argument("--models", nargs="+", required=True, help="Classifier names")
    p.add_argument("--output", required=True, help="Path to write probe_evaluation.json")
    p.add_argument("--low-vram", action="store_true")
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> None:
    from .config import Config
    from .model_manager import ModelManager
    args = _cli_parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    cfg = Config(low_vram=args.low_vram)
    models = ModelManager(cfg)
    metrics = evaluate_probes(manifest, args.models, models,
                               manifest_dir=manifest_path.parent)
    write_report(metrics, args.output, manifest_path=str(manifest_path))
    _print_summary(metrics)


def _print_summary(metrics: dict[str, ModelMetrics]) -> None:
    """Dump a short stdout summary table for human reading."""
    header = (
        f"{'model':25s} {'overall':>10s} {'bias_heavy':>12s} {'bias_strip':>12s}"
        f" {'adversarial':>12s} {'bias_lift':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:25s} {m.top1_accuracy_overall:>10.3f} "
              f"{m.top1_accuracy_bias_heavy:>12.3f} "
              f"{m.top1_accuracy_bias_stripped:>12.3f} "
              f"{m.top1_accuracy_adversarial:>12.3f} "
              f"{m.bias_lift_score:>10.3f}")


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
