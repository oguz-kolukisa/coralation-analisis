"""Generate probe images per class from a feature catalog.

Reads ``feature_catalog.json`` (produced by ``src.feature_extractor``),
asks the VLM to write prompt triplets (bias_heavy / bias_stripped /
real_feature_only) per class, then renders each prompt with the FLUX
editor used in text-to-image mode (image=None, 4 steps, bfloat16).
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from .__version__ import __version__
from .vlm import ProbeFeatures, ProbePromptSet

if TYPE_CHECKING:  # pragma: no cover
    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


_VARIANTS: tuple[str, ...] = (
    "bias_heavy", "bias_stripped", "real_feature_only", "adversarial",
)
_DEFAULT_SEED_BASE = 42
_DEFAULT_N_PER_VARIANT = 4
_VALID_MODES = ("round_robin", "vlm_discretion")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """User-tunable knobs for probe generation."""
    n_per_variant: int | None = None  # None → auto: ceil(sqrt(k_bias_features))
    seed_base: int = _DEFAULT_SEED_BASE
    feature_source: str = "strict"    # "strict" → falls back to "any" per class
    image_size: tuple[int, int] = (768, 768)
    mode: str = "round_robin"         # {"round_robin", "vlm_discretion"}


@dataclass
class ProbeImage:
    """One generated probe image plus its metadata."""
    class_name: str
    variant: str
    prompt: str
    seed: int
    image_path: str
    index: int


@dataclass
class ClassProbeEntry:
    """All probe metadata for one class."""
    bias_features: list[str] = field(default_factory=list)
    real_features: list[str] = field(default_factory=list)
    confusing_classes: list[str] = field(default_factory=list)
    feature_source_used: str = "none"
    n_per_variant: int = 0                    # actual N used for this class
    bias_feature_groups: list[list[str]] = field(default_factory=list)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    images: list[ProbeImage] = field(default_factory=list)
    vlm_failed: bool = False
    synthesized_fallback: bool = False


@dataclass
class ProbeManifest:
    """Top-level manifest written to disk alongside the probe images."""
    version: str
    generated_at: str
    source_catalog: str
    editor_model: str
    vlm_model: str
    config: ProbeConfig
    classes: dict[str, ClassProbeEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_probes(catalog: dict, output_dir: Path, cfg: ProbeConfig,
                    models: "ModelManager", source_catalog: str = "") -> ProbeManifest:
    """Build a probe image set for every class in ``catalog``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_sets = _vlm_phase(catalog, cfg, models)
    manifest = _new_manifest(cfg, models, source_catalog)
    _editor_phase(prompt_sets, cfg, models, output_dir, manifest, catalog)
    _write_manifest(manifest, output_dir / "manifest.json")
    return manifest


def load_catalog(path: Path | str) -> dict:
    """Read a ``feature_catalog.json`` from disk."""
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Phase 1: VLM — build prompt sets for every class
# ---------------------------------------------------------------------------

def _vlm_phase(catalog: dict, cfg: ProbeConfig,
                models: "ModelManager") -> dict[str, ProbePromptSet]:
    """Call the VLM once per class, collect all prompt sets."""
    vlm = models.vlm()
    out: dict[str, ProbePromptSet] = {}
    for class_name, class_data in catalog.get("classes", {}).items():
        bias, real, _ = _select_features(class_data, cfg.feature_source)
        confusing = list(class_data.get("confusing_classes", []))
        n = resolve_n_per_variant(len(bias), cfg.n_per_variant)
        groups = partition_round_robin(bias, n) if cfg.mode == "round_robin" else [bias]
        feats = ProbeFeatures(class_name, bias, real, confusing)
        out[class_name] = vlm.generate_probe_prompts(
            feats, n, mode=cfg.mode, bias_groups=groups,
        )
    models.offload_vlm()
    return out


def resolve_n_per_variant(k: int, user_override: int | None) -> int:
    """Return N prompts per variant: user override if set, else ceil(sqrt(k))."""
    if user_override is not None and user_override > 0:
        return user_override
    return max(1, math.ceil(math.sqrt(max(1, k))))


def partition_round_robin(items: list[str], n: int) -> list[list[str]]:
    """Split ``items`` into ``n`` round-robin groups; returns n lists (possibly empty)."""
    groups: list[list[str]] = [[] for _ in range(max(1, n))]
    for i, item in enumerate(items):
        groups[i % len(groups)].append(item)
    return groups


def _select_features(class_data: dict, source: str) -> tuple[list[str], list[str], str]:
    """Pick bias + real feature lists from consensus, with strict→any fallback."""
    consensus = class_data.get("consensus", {})
    if source == "strict":
        bias = list(consensus.get("bias_spurious_strict", []))
        used = "strict"
        if not bias:
            bias = list(consensus.get("bias_spurious_any", []))
            used = "any" if bias else "none"
    else:
        bias = list(consensus.get("bias_spurious_any", []))
        used = "any" if bias else "none"
    real = list(consensus.get("real_any", []))
    return bias, real, used


# ---------------------------------------------------------------------------
# Phase 2: Editor — render every prompt
# ---------------------------------------------------------------------------

def _editor_phase(prompt_sets: dict[str, ProbePromptSet], cfg: ProbeConfig,
                   models: "ModelManager", output_dir: Path,
                   manifest: ProbeManifest, catalog: dict) -> None:
    """Load FLUX once, render every prompt across every class × variant."""
    editor = models.editor()
    for class_name, prompt_set in prompt_sets.items():
        class_data = catalog["classes"].get(class_name, {})
        entry = _build_class_entry(class_data, prompt_set, cfg)
        class_dir = output_dir / _slugify(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        _render_all_variants(editor, prompt_set, entry, cfg, class_dir, output_dir)
        manifest.classes[class_name] = entry
    models.offload_editor()


def _build_class_entry(class_data: dict, prompt_set: ProbePromptSet,
                        cfg: ProbeConfig) -> ClassProbeEntry:
    """Assemble a ClassProbeEntry with feature lists and prompt metadata."""
    bias, real, source_used = _select_features(class_data, cfg.feature_source)
    confusing = list(class_data.get("confusing_classes", []))
    n = resolve_n_per_variant(len(bias), cfg.n_per_variant)
    groups = partition_round_robin(bias, n) if cfg.mode == "round_robin" else [bias]
    fallback = not bias
    return ClassProbeEntry(
        bias_features=bias, real_features=real,
        confusing_classes=confusing,
        feature_source_used=source_used,
        n_per_variant=n,
        bias_feature_groups=groups,
        prompts={
            "bias_heavy": list(prompt_set.bias_heavy),
            "bias_stripped": list(prompt_set.bias_stripped),
            "real_feature_only": list(prompt_set.real_feature_only),
            "adversarial": list(prompt_set.adversarial),
        },
        synthesized_fallback=fallback,
    )


def _render_all_variants(editor, prompt_set: ProbePromptSet,
                          entry: ClassProbeEntry, cfg: ProbeConfig,
                          class_dir: Path, output_dir: Path) -> None:
    """Render every (variant, index) pair for one class."""
    for variant_idx, variant in enumerate(_VARIANTS):
        prompts = entry.prompts[variant]
        for image_idx, prompt in enumerate(prompts):
            probe = _render_one(
                editor, prompt_set.class_name, variant, variant_idx,
                prompt, image_idx, cfg, class_dir, output_dir,
            )
            if probe is not None:
                entry.images.append(probe)


def _render_one(editor, class_name: str, variant: str, variant_idx: int,
                 prompt: str, image_idx: int, cfg: ProbeConfig,
                 class_dir: Path, output_dir: Path) -> ProbeImage | None:
    """Render a single prompt; return a ProbeImage or None on failure."""
    seed = _seed_for(variant_idx, image_idx, cfg.seed_base)
    image_path = class_dir / f"{variant}_{image_idx}.jpg"
    try:
        img = editor.generate_from_text(prompt, seed=seed, size=cfg.image_size)
        _persist_image(img, image_path)
    except Exception as e:
        logger.warning("Probe render failed for %s/%s/%d: %s",
                       class_name, variant, image_idx, e)
        return None
    rel_path = image_path.relative_to(output_dir).as_posix()
    return ProbeImage(class_name=class_name, variant=variant, prompt=prompt,
                      seed=seed, image_path=rel_path, index=image_idx)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Lower-case, replace non-alnum runs with underscore, trim."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _seed_for(variant_idx: int, image_idx: int, base: int) -> int:
    """Deterministic per (variant_idx, image_idx, seed_base)."""
    return base + variant_idx * 1000 + image_idx


def _persist_image(img: Image.Image, path: Path) -> None:
    """Save a PIL image as JPEG quality 90."""
    img.convert("RGB").save(path, quality=90)


def _new_manifest(cfg: ProbeConfig, models: "ModelManager",
                   source_catalog: str) -> ProbeManifest:
    """Build an empty manifest with run metadata."""
    return ProbeManifest(
        version=__version__,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        source_catalog=source_catalog,
        editor_model=_safe_cfg_str(models, "editor_model"),
        vlm_model=_safe_cfg_str(models, "vlm_model"),
        config=cfg,
    )


def _safe_cfg_str(models: "ModelManager", attr: str) -> str:
    """Read a string config attribute, returning '' if absent or non-string."""
    value = getattr(getattr(models, "cfg", None), attr, "")
    return value if isinstance(value, str) else ""


def _write_manifest(manifest: ProbeManifest, path: Path) -> None:
    """Serialize a ProbeManifest to JSON on disk."""
    path.write_text(json.dumps(_manifest_to_dict(manifest), indent=2))


def _manifest_to_dict(manifest: ProbeManifest) -> dict:
    """Convert a ProbeManifest into a plain dict (JSON-ready)."""
    out = asdict(manifest)
    out["config"] = asdict(manifest.config)
    out["config"]["image_size"] = list(manifest.config.image_size)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(description="Generate probe image set from a feature catalog")
    p.add_argument("--catalog", required=True, help="Path to feature_catalog.json")
    p.add_argument("--output", required=True, help="Output directory for probe images + manifest")
    p.add_argument("--n-per-variant", type=int, default=_DEFAULT_N_PER_VARIANT)
    p.add_argument("--feature-source", choices=("strict", "any"), default="strict")
    p.add_argument("--seed-base", type=int, default=_DEFAULT_SEED_BASE)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--high-vram", action="store_true")
    p.add_argument("--no-8bit-editor", action="store_true")
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> None:
    from .config import Config
    from .model_manager import ModelManager
    args = _cli_parse_args(argv)
    cfg = Config(
        low_vram=args.low_vram and not args.high_vram,
        use_8bit_editor=not args.no_8bit_editor,
    )
    models = ModelManager(cfg)
    catalog = load_catalog(args.catalog)
    probe_cfg = ProbeConfig(
        n_per_variant=args.n_per_variant,
        feature_source=args.feature_source,
        seed_base=args.seed_base,
    )
    manifest = generate_probes(catalog, Path(args.output), probe_cfg, models,
                                source_catalog=str(args.catalog))
    n_images = sum(len(c.images) for c in manifest.classes.values())
    print(f"Probe manifest: {len(manifest.classes)} classes, {n_images} images")


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
