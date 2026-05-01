"""Build Salient ImageNet class list + ground-truth JSON from raw MTurk CSVs.

Reads:
    data/salient_imagenet/discover_spurious_features.csv  (MTurk worker votes)
    data/salient_imagenet/class_metadata.csv              (synset_id -> class_name)

Writes:
    src/salient_imagenet_classes.json   (synset_id -> class_name, default 357 classes)
    data/salient_imagenet/ground_truth.json  (per-class human feature labels, all 1000 classes)

Run once on the machine you'll execute the experiment on (no GPU needed):
    uv run python scripts/build_salient_imagenet_artifacts.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SI_DIR = PROJECT_ROOT / "data" / "salient_imagenet"
SPURIOUS_LABELS = {"background", "separate_object"}


def majority_spurious_features(votes: pd.DataFrame, min_votes: int) -> pd.DataFrame:
    """Aggregate worker votes per (class, feature) into a single label row."""
    votes = votes.copy()
    votes["is_spurious"] = votes["Answer.main"].isin(SPURIOUS_LABELS)
    grouped = votes.groupby(
        ["Input.class_index", "Input.wordnet_id", "Input.feature_index", "Input.feature_rank"]
    ).agg(
        n_spurious=("is_spurious", "sum"),
        n_main=("Answer.main", lambda x: (x == "main_object").sum()),
        n_background=("Answer.main", lambda x: (x == "background").sum()),
        n_separate=("Answer.main", lambda x: (x == "separate_object").sum()),
        mean_confidence=("Answer.confidence", "mean"),
    ).reset_index()
    grouped["majority_spurious"] = grouped["n_spurious"] >= min_votes
    return grouped


def select_spurious_classes(features: pd.DataFrame) -> list[int]:
    """Return class indices that have at least one majority-spurious feature."""
    flagged = features[features["majority_spurious"]]["Input.class_index"].unique()
    return sorted(int(c) for c in flagged)


def write_class_file(class_indices: list[int], metadata: pd.DataFrame, output: Path) -> int:
    """Write synset_id -> class_name JSON for the chosen class indices."""
    selected = metadata[metadata["class_index"].isin(class_indices)]
    selected = selected.sort_values("class_index")
    mapping = {row["wordnet_id"]: row["synset"] for _, row in selected.iterrows()}
    output.write_text(json.dumps(mapping, indent=2))
    return len(mapping)


def collect_worker_reasons(raw: pd.DataFrame, cls_idx: int, feat_idx: int) -> list[str]:
    """Return up-to-5 worker reason strings for a (class, feature) pair."""
    rows = raw[(raw["Input.class_index"] == cls_idx) & (raw["Input.feature_index"] == feat_idx)]
    reasons = rows["Answer.reasons"].dropna().astype(str).tolist()
    return reasons[:5]


def build_ground_truth(features: pd.DataFrame, raw: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Build per-class human-feature ground truth for all 1000 classes."""
    name_lookup = dict(zip(metadata["wordnet_id"], metadata["synset"]))
    gt: dict[str, dict] = {}
    for cls_idx in sorted(features["Input.class_index"].unique()):
        sub = features[features["Input.class_index"] == cls_idx]
        wnid = sub["Input.wordnet_id"].iloc[0]
        gt[wnid] = {
            "class_index": int(cls_idx),
            "class_name": name_lookup.get(wnid, ""),
            "features": [_feature_record(row, raw, cls_idx) for _, row in sub.iterrows()],
        }
    return gt


def _feature_record(row, raw: pd.DataFrame, cls_idx: int) -> dict:
    """Serialize one human-annotated feature with worker votes and reasons."""
    return {
        "feature_index": int(row["Input.feature_index"]),
        "feature_rank": int(row["Input.feature_rank"]),
        "majority_label": "spurious" if row["majority_spurious"] else "core",
        "n_main": int(row["n_main"]),
        "n_background": int(row["n_background"]),
        "n_separate": int(row["n_separate"]),
        "mean_confidence": float(row["mean_confidence"]),
        "worker_reasons": collect_worker_reasons(raw, cls_idx, int(row["Input.feature_index"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-votes", type=int, default=3, help="Workers (of 5) that must call a feature spurious.")
    parser.add_argument("--si-dir", type=Path, default=SI_DIR)
    parser.add_argument("--class-output", type=Path, default=PROJECT_ROOT / "src" / "salient_imagenet_classes.json")
    parser.add_argument("--gt-output", type=Path, default=SI_DIR / "ground_truth.json")
    args = parser.parse_args()

    raw = pd.read_csv(args.si_dir / "discover_spurious_features.csv")
    metadata = pd.read_csv(args.si_dir / "class_metadata.csv")

    features = majority_spurious_features(raw, args.min_votes)
    spurious_classes = select_spurious_classes(features)
    n_classes = write_class_file(spurious_classes, metadata, args.class_output)

    gt = build_ground_truth(features, raw, metadata)
    args.gt_output.write_text(json.dumps(gt, indent=2))

    n_spurious_feats = sum(
        1 for cls in gt.values() for f in cls["features"] if f["majority_label"] == "spurious"
    )
    print(f"Wrote {n_classes} classes -> {args.class_output}")
    print(f"Wrote ground truth ({len(gt)} classes, {n_spurious_feats} spurious features) -> {args.gt_output}")


if __name__ == "__main__":
    main()
