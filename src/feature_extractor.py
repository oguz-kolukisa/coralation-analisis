"""Aggregate pipeline verdicts into per-class feature catalogs.

Reads ``analysis_results.json`` and produces a per-class, per-model bucketing
of features into real (essential/intrinsic) vs biases (spurious, state),
plus a cross-model consensus view.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


_REAL_VERDICTS = {"essential", "intrinsic"}
_BIAS_SPURIOUS = {"spurious"}
_BIAS_STATE = {"state_bias"}
_SIGNAL_VERDICTS = _REAL_VERDICTS | _BIAS_SPURIOUS | _BIAS_STATE


def extract_catalog(analysis: dict) -> dict:
    """Build feature catalog from a full ``analysis_results.json`` payload."""
    models = _discover_models(analysis)
    classes = {
        r["class_name"]: _class_catalog(r, models)
        for r in analysis.get("results", [])
    }
    return {
        "version": analysis.get("version"),
        "source_generated_at": analysis.get("generated_at"),
        "source_models": models,
        "classes": classes,
    }


def load_and_extract(path: Path | str) -> dict:
    """Load analysis JSON from disk and return the catalog."""
    with open(path) as f:
        return extract_catalog(json.load(f))


def write_catalog(catalog: dict, path: Path | str) -> None:
    """Write catalog to JSON on disk."""
    Path(path).write_text(json.dumps(catalog, indent=2))


def _discover_models(analysis: dict) -> list[str]:
    """Collect every classifier model name seen in any verdict."""
    models: set[str] = set()
    for result in analysis.get("results", []):
        for er in result.get("edit_results", []):
            models.update(er.get("verdict", {}).keys())
    return sorted(models)


def _class_catalog(result: dict, models: list[str]) -> dict:
    """Build per-model buckets and consensus for one class result."""
    per_model = {m: _model_buckets(result, m) for m in models}
    return {
        "per_model": per_model,
        "consensus": _consensus(per_model, models),
        "confusing_classes": list(result.get("confusing_classes", [])),
    }


def _model_buckets(result: dict, model: str) -> dict:
    """Split a class's features into real/bias/inconclusive for one model."""
    entries = _feature_entries(result, model)
    return {
        "real": [e for e in entries if _has_signal(e, _REAL_VERDICTS)],
        "bias_spurious": [e for e in entries if _has_signal(e, _BIAS_SPURIOUS)],
        "bias_state": [e for e in entries if _has_signal(e, _BIAS_STATE)],
        "inconclusive": [e for e in entries if not _has_any_signal(e)],
    }


def _feature_entries(result: dict, model: str) -> list[dict]:
    """Group edit_results by feature name and shape them as entries."""
    groups = _group_by_feature(result, model)
    return [_to_entry(name, data) for name, data in groups.items()]


def _group_by_feature(result: dict, model: str) -> dict:
    """Collect signals and metadata keyed by feature name."""
    groups: dict = defaultdict(lambda: {"signals": [], "n": 0, "type": ""})
    for er in result.get("edit_results", []):
        name = er.get("feature_name", "")
        groups[name]["type"] = er.get("feature_type", "")
        groups[name]["n"] += 1
        _record_verdict(groups[name], er, model)
    return groups


def _record_verdict(bucket: dict, er: dict, model: str) -> None:
    """Append this edit's verdict for the given model, if present."""
    v = er.get("verdict", {}).get(model)
    if isinstance(v, dict) and v.get("verdict"):
        bucket["signals"].append(v["verdict"])


def _to_entry(name: str, data: dict) -> dict:
    """Shape a grouped bucket into a catalog entry."""
    return {
        "feature": name,
        "feature_type": data["type"],
        "signals": sorted(set(data["signals"])),
        "n_edits": data["n"],
    }


def _has_signal(entry: dict, verdicts: set[str]) -> bool:
    """True if entry carries at least one of the given verdicts."""
    return any(s in verdicts for s in entry["signals"])


def _has_any_signal(entry: dict) -> bool:
    """True if entry carries any real/bias signal (not just meta verdicts)."""
    return bool(set(entry["signals"]) & _SIGNAL_VERDICTS)


def _consensus(per_model: dict, models: list[str]) -> dict:
    """Build strict (all models) and any (union) consensus sets."""
    buckets = ("real", "bias_spurious", "bias_state")
    out: dict = {}
    for bucket in buckets:
        out[f"{bucket}_strict"] = _intersect(per_model, models, bucket)
        out[f"{bucket}_any"] = _union(per_model, models, bucket)
    return out


def _names_in(per_model: dict, model: str, bucket: str) -> set[str]:
    """Return the set of feature names in one model's bucket."""
    return {e["feature"] for e in per_model[model][bucket]}


def _intersect(per_model: dict, models: list[str], bucket: str) -> list[str]:
    """Names present in this bucket for every model."""
    if not models:
        return []
    sets = [_names_in(per_model, m, bucket) for m in models]
    return sorted(set.intersection(*sets))


def _union(per_model: dict, models: list[str], bucket: str) -> list[str]:
    """Names present in this bucket for any model."""
    if not models:
        return []
    sets = [_names_in(per_model, m, bucket) for m in models]
    return sorted(set.union(*sets))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract per-class feature catalog")
    ap.add_argument("--input", default="output/reports/analysis_results.json")
    ap.add_argument("--output", default="output/reports/feature_catalog.json")
    args = ap.parse_args()
    catalog = load_and_extract(args.input)
    write_catalog(catalog, args.output)
    n_classes = len(catalog["classes"])
    n_models = len(catalog["source_models"])
    print(f"Wrote {args.output}: {n_classes} classes × {n_models} models")
