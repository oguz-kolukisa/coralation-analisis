"""Build feature_catalog.json from Salient ImageNet ground truth.

Output schema matches what src/probe_generator.py expects:
    {classes: {<class_name>: {consensus: {bias_spurious_any, real_any}, confusing_classes: [...]}}}

The bias_spurious_any list is populated from SI's `spurious`-labeled features and
real_any from SI's `core`-labeled features. Each SI feature has 5 worker reasons; we
distill them to a short noun phrase via regex heuristics so the prompt VLM gets clean
input instead of full sentences.

Confusing classes are reused from our existing per-class checkpoints (Algorithm 1
already computed them); falls back to [] if a checkpoint is missing.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/coralation-analisis")
GT_PATH = ROOT / "data/salient_imagenet/ground_truth.json"
SUBSET_PATH = ROOT / "src/salient_imagenet_classes.json"
CKPT_DIR = Path("/workspace/coralation/output/salient_imagenet_run/checkpoints/resnet50")
OUT_PATH = Path("/workspace/coralation/output/si_as_input_run/reports/feature_catalog.json")

_PREFIX_PATTERNS = [
    r"^the\s+focus\s+(?:is\s+)?(?:mainly\s+)?on\s+",
    r"^focus(?:es|ed|ing)?\s+(?:is\s+)?(?:mainly\s+)?on\s+",
    r"^(?:the\s+)?(?:red\s+)?(?:focus|region|highlighted\s+visual\s+attributes?|attention)\s+(?:is\s+)?(?:on|focuses\s+on)\s+",
    r"^red\s+focus\s+(?:is\s+)?on\s+",
    r"^image[s]?\s+(?:are|is)\s+(?:mainly|fully)?\s*focused\s+on\s+",
    r"^main\s+(?:focus|object)\s+is\s+",
]
_SUFFIX_PATTERNS = [
    r"\s+which\s+is\s+(?:not\s+)?(?:a\s+)?part\s+of.*$",
    r"\s+(?:not\s+(?:on|part\s+of)\s+the?\s*).*$",
    r"[.;!?].*$",
]
# Words that mark the end of a useful noun phrase
_TRUNC_WORDS = {"of", "in", "on", "around", "near", "behind", "below", "above",
                "next", "across", "with", "from", "to", "at", "by", "that",
                "which", "and", "or", "but", "while", "where"}
# Stop words that aren't useful as standalone features
_STOP = {"the", "a", "an", "some", "its", "this", "that", "these", "those",
         "main", "object", "class", "background", "image", "images", "part",
         "parts", "area", "areas", "region", "regions", "thing", "things",
         "section", "side"}


def _strip(text: str) -> str:
    text = text.strip().lower()
    for pat in _PREFIX_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    for pat in _SUFFIX_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:the|a|an|some|its|main)\s+", "", text).strip()
    return text


def _truncate_at_filler(words: list[str]) -> list[str]:
    """Cut at first filler word (of/in/around/etc) — keeps the head noun phrase."""
    out = []
    for w in words:
        if w in _TRUNC_WORDS and out:
            break
        out.append(w)
    return out


def _distill(reasons: list[str], class_name: str) -> str:
    """Pick the most common short noun phrase across worker reasons."""
    cls_words = re.split(r"[ ,]+", class_name.lower())
    cls_set = {w.strip(",.") for w in cls_words if len(w.strip(",.")) > 2}
    candidates: list[str] = []
    for r in reasons:
        s = _strip(r)
        if not s:
            continue
        words = [w.strip(",.;:") for w in s.split() if w.strip(",.;:")]
        words = _truncate_at_filler(words)
        words = [w for w in words if w not in cls_set and w not in _STOP and len(w) > 1]
        if words:
            candidates.append(" ".join(words[:4]))
    if not candidates:
        return _strip(reasons[0])[:50] or "spurious context"
    return Counter(candidates).most_common(1)[0][0]


def _load_confusing(class_name: str) -> list[str]:
    safe = class_name.replace(" ", "_").replace("/", "_")
    p = CKPT_DIR / f"{safe}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return list(d.get("confusing_classes", []))


def _build_class_entry(wnid: str, gt: dict) -> tuple[str, dict]:
    info = gt[wnid]
    cls_name = info["class_name"]
    bias, real = [], []
    for f in info["features"]:
        phrase = _distill(f["worker_reasons"], cls_name)
        (bias if f["majority_label"] == "spurious" else real).append(phrase)
    return cls_name, {
        "consensus": {
            "bias_spurious_strict": bias,
            "bias_spurious_any": bias,
            "real_any": real,
        },
        "confusing_classes": _load_confusing(cls_name),
    }


def _build_catalog(gt: dict, subset: dict) -> dict:
    classes = {}
    for wnid in subset:
        if wnid not in gt:
            continue
        cls_name, entry = _build_class_entry(wnid, gt)
        classes[cls_name] = entry
    return {
        "version": "0.1.0-si",
        "source_generated_at": "salient-imagenet-2026-05-02",
        "source_models": ["si-mturk-human"],
        "classes": classes,
    }


def main():
    gt = json.loads(GT_PATH.read_text())
    subset = json.loads(SUBSET_PATH.read_text())
    catalog = _build_catalog(gt, subset)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(catalog, indent=2))
    n = len(catalog["classes"])
    n_bias = sum(len(c["consensus"]["bias_spurious_any"]) for c in catalog["classes"].values())
    n_real = sum(len(c["consensus"]["real_any"]) for c in catalog["classes"].values())
    n_conf = sum(1 for c in catalog["classes"].values() if c["confusing_classes"])
    print(f"Wrote {OUT_PATH}")
    print(f"  classes: {n}")
    print(f"  bias_spurious total: {n_bias}  (avg {n_bias/n:.2f}/class)")
    print(f"  real total: {n_real}  (avg {n_real/n:.2f}/class)")
    print(f"  classes with confusing_classes filled: {n_conf}/{n}")


if __name__ == "__main__":
    main()
