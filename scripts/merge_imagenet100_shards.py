"""Merge 4 ImageNet-100 shards into a single set of reports.

Each shard directory looks like:

    /workspace/imagenet100/shard_<lo>_<hi>/output/reports/
        <model>_analysis_results.json   # per-model class results (25 classes)
        <model>_report.html / .md
        comparison_report.html / .md
        comparison_results.json

The shards are disjoint by class, so the merge is a simple list concat per
model name. Output goes to /workspace/imagenet100/merged/output/reports/.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reporter import Reporter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/workspace/imagenet100")
SHARD_DIRS = [
    ROOT / "shard_0_25" / "output",
    ROOT / "shard_25_50" / "output",
    ROOT / "shard_50_75" / "output",
    ROOT / "shard_75_100" / "output",
]
MERGED = ROOT / "merged" / "output"


def _shard_reports(shard: Path) -> Path:
    reports = shard / "reports"
    if not reports.exists():
        raise FileNotFoundError(f"Shard reports missing: {reports}")
    return reports


def _model_names_in(reports_dir: Path) -> list[str]:
    """Return the set of model names with per-model JSON in this shard."""
    return sorted(
        p.stem.removesuffix("_analysis_results")
        for p in reports_dir.glob("*_analysis_results.json")
        if not p.name.startswith("comparison")
    )


def _load_per_model(reports_dir: Path, model: str) -> list[dict]:
    path = reports_dir / f"{model}_analysis_results.json"
    return json.loads(path.read_text())


def _shared_config(reports_dir: Path) -> dict:
    """Pull the config block from any per-shard JSON so the merged report has it."""
    candidate = next(reports_dir.glob("*_analysis_results.json"), None)
    if candidate is None:
        return {}
    data = json.loads(candidate.read_text())
    return data[0].get("config", {}) if data else {}


def main():
    merged_reports = MERGED / "reports"
    merged_reports.mkdir(parents=True, exist_ok=True)

    # Collect per-model results across shards
    all_model_dicts: dict[str, list[dict]] = {}
    config = {}
    for shard in SHARD_DIRS:
        reports_dir = _shard_reports(shard)
        if not config:
            config = _shared_config(reports_dir)
        for model in _model_names_in(reports_dir):
            all_model_dicts.setdefault(model, []).extend(
                _load_per_model(reports_dir, model)
            )
        logger.info("merged shard %s (models=%d)", shard.name, len(all_model_dicts))

    # Per-model reports
    for model, model_dicts in all_model_dicts.items():
        reporter = Reporter(merged_reports, config=config, prefix=model)
        reporter.generate_all(model_dicts)

    # Cross-model comparison
    Reporter(merged_reports, config=config).generate_comparison(
        all_model_dicts, sorted(all_model_dicts.keys())
    )
    logger.info("merged %d models, output: %s", len(all_model_dicts), merged_reports)


if __name__ == "__main__":
    main()
