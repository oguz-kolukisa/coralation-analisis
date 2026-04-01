"""
Dataset Export — Save results as JSON + HuggingFace dataset format.

The JSON dataset includes all features, edits, results, and relative image paths
so the entire output folder is self-contained and reproducible.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _rel(abs_path: str, base: Path) -> str:
    """Convert absolute path to relative."""
    try:
        return str(os.path.relpath(Path(abs_path).resolve(), base.resolve()))
    except ValueError:
        return abs_path


def export_dataset_json(
    results: list[dict], config: dict, output_dir: Path,
) -> Path:
    """Export complete dataset as JSON with relative paths."""
    dataset = {
        "config": _extract_config(config),
        "classes": [_build_class_data(cls, output_dir) for cls in results],
    }
    path = output_dir / "dataset.json"
    path.write_text(json.dumps(dataset, indent=2, default=str))
    logger.info("Saved dataset JSON: %s", path)
    return path


def _extract_config(config: dict) -> dict:
    """Extract relevant config fields for dataset metadata."""
    return {
        "classifiers": config.get("classifier_models", []),
        "vlm": config.get("vlm_model", ""),
        "editor": config.get("editor_model", ""),
        "samples_per_class": config.get("samples_per_class", 0),
    }


def _build_class_data(cls: dict, output_dir: Path) -> dict:
    """Build dataset entry for one class."""
    return {
        "class_name": cls["class_name"],
        "features": [
            {"name": c["name"], "type": c["type"], "attention": c.get("attention", "")}
            for c in cls.get("unique_concepts", [])
        ],
        "confusing_classes": cls.get("confusing_classes", []),
        "edits": [_build_edit_data(er, output_dir) for er in cls.get("edit_results", [])],
    }


def _build_edit_data(er: dict, output_dir: Path) -> dict:
    """Build dataset entry for one edit result."""
    return {
        "feature_name": er["feature_name"],
        "feature_type": er["feature_type"],
        "edit_type": er["edit_type"],
        "edit_instruction": er["edit_instruction"],
        "target": er["target"],
        "original_image": _rel(er.get("original_image", ""), output_dir),
        "edited_image": _rel(er.get("edited_image", ""), output_dir),
        "results": {
            model: {
                "original": m["original_confidence"],
                "edited": m["edited_confidence"],
                "delta": m["delta"],
                "verdict": er.get("verdict", {}).get(model, {}).get("verdict", ""),
            }
            for model, m in er.get("per_model", {}).items()
        },
    }


def export_huggingface(results: list[dict], output_dir: Path) -> Path:
    """Export as HuggingFace dataset format."""
    try:
        from datasets import Dataset
    except ImportError:
        logger.warning("datasets library not installed, skipping HF export")
        return output_dir / "dataset_hf"

    rows = _build_hf_rows(results)
    if not rows:
        logger.warning("No edit results to export as HF dataset")
        return output_dir / "dataset_hf"

    ds = Dataset.from_list(rows)
    hf_dir = output_dir / "dataset_hf"
    ds.save_to_disk(str(hf_dir))
    logger.info("Saved HuggingFace dataset: %s", hf_dir)
    return hf_dir


def _build_hf_rows(results: list[dict]) -> list[dict]:
    """Build flat rows for HuggingFace dataset."""
    rows = []
    for cls in results:
        for er in cls.get("edit_results", []):
            row = _build_hf_row(cls["class_name"], er)
            rows.append(row)
    return rows


def _build_hf_row(class_name: str, er: dict) -> dict:
    """Build one row for HuggingFace dataset."""
    row = {
        "class_name": class_name,
        "feature_name": er["feature_name"],
        "feature_type": er["feature_type"],
        "edit_type": er["edit_type"],
        "edit_instruction": er["edit_instruction"],
        "target": er["target"],
        "original_image": er.get("original_image", ""),
        "edited_image": er.get("edited_image", ""),
    }
    for model, m in er.get("per_model", {}).items():
        row[f"{model}_original"] = m["original_confidence"]
        row[f"{model}_edited"] = m["edited_confidence"]
        row[f"{model}_delta"] = m["delta"]
        verdict = er.get("verdict", {}).get(model, {})
        row[f"{model}_verdict"] = verdict.get("verdict", "") if isinstance(verdict, dict) else ""
    return row
