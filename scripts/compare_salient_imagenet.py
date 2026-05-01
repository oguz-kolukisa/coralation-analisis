"""Compare Algorithm 1 discovered features against Salient ImageNet human labels.

Inputs:
    output/reports/feature_catalog.json   (Algorithm 1 output, per-class buckets)
    data/salient_imagenet/ground_truth.json (built by build_salient_imagenet_artifacts.py)
    src/salient_imagenet_classes.json     (synset_id -> class_name mapping)

Outputs:
    output/reports/salient_imagenet_comparison.json   (per-class metrics + matched pairs)
    output/reports/salient_imagenet_comparison.md     (summary tables)

Usage:
    uv run python scripts/compare_salient_imagenet.py --classifier resnet50
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "output" / "reports" / "feature_catalog.json"
DEFAULT_GT = PROJECT_ROOT / "data" / "salient_imagenet" / "ground_truth.json"
DEFAULT_CLASSES = PROJECT_ROOT / "src" / "salient_imagenet_classes.json"
DEFAULT_OUT_JSON = PROJECT_ROOT / "output" / "reports" / "salient_imagenet_comparison.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "output" / "reports" / "salient_imagenet_comparison.md"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIM_THRESHOLD = 0.45
SI_SPURIOUS_LABEL = "spurious"
SI_CORE_LABEL = "core"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("si_compare")


@dataclass
class DiscoveredFeature:
    name: str
    bucket: str  # "real" | "bias_spurious" | "bias_state" | "inconclusive"
    feature_type: str
    n_edits: int


@dataclass
class HumanFeature:
    feature_index: int
    rank: int
    label: str  # "core" | "spurious"
    description: str  # joined worker reasons (used for matching)
    confidence: float


@dataclass
class ClassComparison:
    wordnet_id: str
    class_name: str
    si_spurious_count: int
    si_core_count: int
    discovered_spurious: list[str] = field(default_factory=list)
    discovered_real: list[str] = field(default_factory=list)
    matched_spurious: list[dict] = field(default_factory=list)
    matched_core: list[dict] = field(default_factory=list)
    spurious_recall: float = 0.0
    spurious_precision: float = 0.0


def embed_texts(texts: list[str], device: str) -> np.ndarray:
    """Embed a list of strings with all-MiniLM-L6-v2 (mean-pooled, L2-normalized)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    tokenizer = embed_texts._tokenizer
    model = embed_texts._model
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = F.normalize(summed / counts, p=2, dim=1)
    return pooled.cpu().numpy()


def init_embedder(device: str) -> None:
    """Load the embedding model once and stash it on embed_texts."""
    log.info("Loading embedding model: %s", EMBED_MODEL)
    embed_texts._tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_texts._model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()


def load_discovered(catalog_path: Path, classifier: str) -> dict[str, list[DiscoveredFeature]]:
    """Read feature_catalog.json and pull the chosen classifier's per-class buckets."""
    catalog = json.loads(catalog_path.read_text())
    classes = catalog.get("classes", {})
    out: dict[str, list[DiscoveredFeature]] = {}
    for cls_name, cls_data in classes.items():
        per_model = cls_data.get("per_model", {})
        if classifier not in per_model:
            continue
        out[cls_name] = list(_iter_features(per_model[classifier]))
    return out


def _iter_features(per_model_entry: dict) -> Iterable[DiscoveredFeature]:
    """Flatten the four buckets into DiscoveredFeature records."""
    for bucket in ("real", "bias_spurious", "bias_state", "inconclusive"):
        for item in per_model_entry.get(bucket, []):
            yield DiscoveredFeature(
                name=item["feature"],
                bucket=bucket,
                feature_type=item.get("feature_type", ""),
                n_edits=int(item.get("n_edits", 0)),
            )


def load_ground_truth(gt_path: Path) -> dict[str, list[HumanFeature]]:
    """Load Salient ImageNet ground truth keyed by wordnet_id."""
    gt = json.loads(gt_path.read_text())
    return {wnid: list(_iter_human_features(payload)) for wnid, payload in gt.items()}


def _iter_human_features(payload: dict) -> Iterable[HumanFeature]:
    """Convert raw GT records into HumanFeature with a joined-reasons description."""
    for feat in payload.get("features", []):
        reasons = " ; ".join(r for r in feat.get("worker_reasons", []) if r)
        yield HumanFeature(
            feature_index=feat["feature_index"],
            rank=feat["feature_rank"],
            label=feat["majority_label"],
            description=reasons or f"feature {feat['feature_index']}",
            confidence=feat["mean_confidence"],
        )


def build_wnid_lookup(class_path: Path, ground_truth: dict) -> dict[str, str]:
    """Invert all-1000 SI metadata to class_name -> synset_id (fallback to class file)."""
    by_name = {payload["class_name"]: wnid for wnid, payload in ground_truth.items() if payload.get("class_name")}
    if not by_name:
        mapping = json.loads(class_path.read_text())
        by_name = {v: k for k, v in mapping.items()}
    return by_name


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine sim of two L2-normalized matrices (rows assumed unit-length)."""
    return a @ b.T if a.size and b.size else np.zeros((len(a), len(b)))


def match_one_class(
    discovered: list[DiscoveredFeature],
    human: list[HumanFeature],
    threshold: float,
    device: str,
) -> tuple[list[dict], list[dict]]:
    """Match discovered spurious + real features to their best human counterparts."""
    spurious_matches = _match_bucket(
        [d for d in discovered if d.bucket == "bias_spurious"],
        [h for h in human if h.label == SI_SPURIOUS_LABEL],
        threshold,
        device,
    )
    core_matches = _match_bucket(
        [d for d in discovered if d.bucket == "real"],
        [h for h in human if h.label == SI_CORE_LABEL],
        threshold,
        device,
    )
    return spurious_matches, core_matches


def _match_bucket(
    discovered: list[DiscoveredFeature],
    human: list[HumanFeature],
    threshold: float,
    device: str,
) -> list[dict]:
    """Cosine-match each discovered feature to its best human counterpart."""
    if not discovered or not human:
        return []
    disc_emb = embed_texts([d.name for d in discovered], device)
    human_emb = embed_texts([h.description for h in human], device)
    sims = cosine_sim_matrix(disc_emb, human_emb)
    return [_match_record(d, human, sims[i], threshold) for i, d in enumerate(discovered)]


def _match_record(d: DiscoveredFeature, human: list[HumanFeature], row: np.ndarray, threshold: float) -> dict:
    """Pick the highest-similarity human feature for this discovered feature."""
    best = int(np.argmax(row))
    best_sim = float(row[best])
    matched = best_sim >= threshold
    h = human[best]
    return {
        "discovered": d.name,
        "discovered_bucket": d.bucket,
        "best_human_feature_index": h.feature_index,
        "best_human_label": h.label,
        "similarity": round(best_sim, 4),
        "matched": matched,
        "human_description": h.description[:200],
    }


def compute_class_metrics(comp: ClassComparison) -> ClassComparison:
    """Fill in spurious recall/precision based on matched pairs."""
    matched_count = sum(1 for m in comp.matched_spurious if m["matched"])
    comp.spurious_precision = (matched_count / len(comp.matched_spurious)) if comp.matched_spurious else 0.0
    matched_human_indices = {m["best_human_feature_index"] for m in comp.matched_spurious if m["matched"]}
    comp.spurious_recall = (len(matched_human_indices) / comp.si_spurious_count) if comp.si_spurious_count else 0.0
    return comp


def compare_one_class(
    cls_name: str,
    wnid: str,
    discovered: list[DiscoveredFeature],
    human: list[HumanFeature],
    threshold: float,
    device: str,
) -> ClassComparison:
    """Run the matching for a single class and assemble the ClassComparison record."""
    comp = ClassComparison(
        wordnet_id=wnid,
        class_name=cls_name,
        si_spurious_count=sum(1 for h in human if h.label == SI_SPURIOUS_LABEL),
        si_core_count=sum(1 for h in human if h.label == SI_CORE_LABEL),
        discovered_spurious=[d.name for d in discovered if d.bucket == "bias_spurious"],
        discovered_real=[d.name for d in discovered if d.bucket == "real"],
    )
    comp.matched_spurious, comp.matched_core = match_one_class(discovered, human, threshold, device)
    return compute_class_metrics(comp)


def aggregate_metrics(comps: list[ClassComparison]) -> dict:
    """Macro-average per-class precision/recall and overall match count."""
    spurious_classes = [c for c in comps if c.si_spurious_count > 0]
    macro_recall = float(np.mean([c.spurious_recall for c in spurious_classes])) if spurious_classes else 0.0
    macro_precision = float(np.mean([c.spurious_precision for c in comps if c.matched_spurious])) if any(c.matched_spurious for c in comps) else 0.0
    total_si_spurious = sum(c.si_spurious_count for c in comps)
    total_recovered = sum(
        len({m["best_human_feature_index"] for m in c.matched_spurious if m["matched"]}) for c in comps
    )
    return {
        "n_classes_compared": len(comps),
        "n_classes_with_si_spurious": len(spurious_classes),
        "macro_spurious_recall": round(macro_recall, 4),
        "macro_spurious_precision": round(macro_precision, 4),
        "micro_spurious_recall": round(total_recovered / total_si_spurious, 4) if total_si_spurious else 0.0,
        "total_si_spurious_features": total_si_spurious,
        "total_recovered_si_spurious_features": total_recovered,
    }


def write_markdown_report(comps: list[ClassComparison], summary: dict, out_path: Path, classifier: str, threshold: float) -> None:
    """Emit a human-readable markdown summary."""
    lines = _md_header(classifier, threshold, summary)
    lines.extend(_md_per_class_table(comps))
    out_path.write_text("\n".join(lines))


def _md_header(classifier: str, threshold: float, summary: dict) -> list[str]:
    """Build the title + summary block of the markdown report."""
    return [
        f"# Salient ImageNet vs. Algorithm 1 ({classifier})",
        "",
        f"Sentence-embedding similarity threshold: **{threshold}** (cosine, all-MiniLM-L6-v2).",
        "",
        "## Aggregate metrics",
        "",
        f"- Classes compared: **{summary['n_classes_compared']}**",
        f"- Classes with ≥1 SI spurious feature: **{summary['n_classes_with_si_spurious']}**",
        f"- Macro spurious recall: **{summary['macro_spurious_recall']}**",
        f"- Macro spurious precision: **{summary['macro_spurious_precision']}**",
        f"- Micro spurious recall: **{summary['micro_spurious_recall']}** "
        f"({summary['total_recovered_si_spurious_features']}/{summary['total_si_spurious_features']} SI spurious features matched)",
        "",
    ]


def _md_per_class_table(comps: list[ClassComparison]) -> list[str]:
    """Per-class table sorted by descending SI spurious count."""
    rows = sorted(comps, key=lambda c: (-c.si_spurious_count, c.class_name))
    out = [
        "## Per-class breakdown (top 50 by SI spurious count)",
        "",
        "| Class | SI spurious | Discovered spurious | Recall | Precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in rows[:50]:
        out.append(
            f"| {c.class_name[:50]} | {c.si_spurious_count} | {len(c.discovered_spurious)} | "
            f"{c.spurious_recall:.2f} | {c.spurious_precision:.2f} |"
        )
    return out


def parse_args() -> argparse.Namespace:
    """CLI argument parsing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", default="resnet50")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--class-file", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_embedder(args.device)

    discovered_by_class = load_discovered(args.catalog, args.classifier)
    raw_gt = json.loads(args.ground_truth.read_text())
    ground_truth = load_ground_truth(args.ground_truth)
    wnid_lookup = build_wnid_lookup(args.class_file, raw_gt)

    comps = list(_compare_all_classes(discovered_by_class, ground_truth, wnid_lookup, args.threshold, args.device))
    summary = aggregate_metrics(comps)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classifier": args.classifier, "threshold": args.threshold, "summary": summary, "per_class": [c.__dict__ for c in comps]}
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_markdown_report(comps, summary, args.out_md, args.classifier, args.threshold)

    log.info("Wrote %s", args.out_json)
    log.info("Wrote %s", args.out_md)
    log.info("Macro spurious recall=%.3f precision=%.3f", summary["macro_spurious_recall"], summary["macro_spurious_precision"])


def _compare_all_classes(
    discovered_by_class: dict[str, list[DiscoveredFeature]],
    ground_truth: dict[str, list[HumanFeature]],
    wnid_lookup: dict[str, str],
    threshold: float,
    device: str,
) -> Iterable[ClassComparison]:
    """Iterate over discovered classes and yield a ClassComparison per matched class."""
    for cls_name, discovered in discovered_by_class.items():
        wnid = wnid_lookup.get(cls_name)
        if not wnid or wnid not in ground_truth:
            log.warning("Skip %s: no Salient ImageNet ground truth", cls_name)
            continue
        yield compare_one_class(cls_name, wnid, discovered, ground_truth[wnid], threshold, device)


if __name__ == "__main__":
    main()
