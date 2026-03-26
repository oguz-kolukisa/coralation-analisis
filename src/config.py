"""
Configuration for the Coralation analysis pipeline.
"""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, Field

# Default location of the token file (project root)
_TOKEN_FILE = Path(__file__).parent.parent / ".token"


def load_hf_token(token_file: Path = _TOKEN_FILE) -> str | None:
    """
    Read the HuggingFace token from .token file.
    Ignores comment lines (starting with #) and blank lines.
    Falls back to the HF_TOKEN environment variable if the file is missing
    or contains only the placeholder.
    """
    import os

    if token_file.exists():
        for line in token_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if not line.startswith("hf_REPLACE"):
                    return line

    return os.environ.get("HF_TOKEN")


# ImageNet-1k class name → HuggingFace label index mapping is resolved at runtime.

_DEFAULT_CLASS_FILE = Path(__file__).parent / "imagenet100_classes.json"


def load_classes_from_file(path: Path) -> list[str]:
    """Load class names from a JSON file (synset_id → label mapping)."""
    with open(path) as f:
        mapping = json.load(f)
    return list(mapping.values())


class Config(BaseModel):
    # --- Model settings ---
    classifier_model: str = "resnet50"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    editor_model: str = "black-forest-labs/FLUX.2-klein-9b-kv"  # Fast, high-quality (4 steps)

    # --- Dataset settings ---
    hf_dataset: str = "ILSVRC/imagenet-1k"
    hf_dataset_split: str = "validation"
    class_source: str = "json"          # "json" = read from file, "dataset" = query dataset labels
    class_file: Path = _DEFAULT_CLASS_FILE  # path to JSON class list (used when class_source="json")
    samples_per_class: int = 100        # positive examples per class (for EDITING)
    inspect_samples: int = 10           # images to INSPECT for feature discovery (separate from editing)
    top_negative_classes: int = 5       # how many confusing classes to select
    negative_samples_per_class: int = 5  # negative images to sample from each confusing class
    max_scan: int = 10000               # max samples to scan when searching for a class
    random_seed: int | None = None      # None = random each run, set for reproducibility
    random_classes: bool = False        # if True, randomly select classes from dataset

    # --- Analysis settings ---
    confidence_delta_threshold: float = 0.15   # min delta to "confirm" a hypothesis
    min_negative_confidence: float = 0.05      # min confidence for negative samples (filter out noise)
    min_meaningful_delta: float = 0.01         # deltas below this are considered failed edits
    top_k_classes: int = 5                      # how many top predictions to record
    max_hypotheses_per_image: int = 5           # VLM edit instructions per image
    max_edits_per_hypothesis: int = 3           # images to edit per hypothesis

    # --- Robustness settings ---
    generations_per_edit: int = 1               # images generated per edit instruction

    # --- Attention map settings ---
    attention_method: str = "scorecam"          # "gradcam", "gradcam++", or "scorecam"
    compute_edit_gradcam: bool = True           # Grad-CAM on edited images (slower, enables attention diff)

    # --- Statistical validation settings ---
    statistical_alpha: float = 0.05             # significance level for t-tests
    min_effect_size: float = 0.5                # minimum Cohen's d for practical significance
    use_statistical_validation: bool = True     # use t-tests instead of simple threshold

    # --- Risk classification thresholds ---
    risk_high_threshold: float = 0.30           # >30% confirmation rate = HIGH risk
    risk_medium_threshold: float = 0.10         # >10% = MEDIUM, else LOW

    # --- Processing thresholds ---
    dedup_similarity_threshold: float = 0.70    # edit deduplication similarity threshold
    confusing_class_min_conf: float = 0.01      # min confidence to consider class "confusing"
    spurious_positive_delta: float = 0.10       # positive delta threshold to flag as spurious

    # --- Iterative analysis ---
    iterations: int = 2                         # number of VLM analysis iterations (1 = no iteration)
    verify_edits: bool = False                  # VLM edit verification disabled by default (slower, often unnecessary)

    # --- Paths ---
    output_dir: Path = Path("output")
    resume: bool = True                 # resume from checkpoint if available

    # --- Runtime ---
    device: str = "cuda"
    vlm_dtype: str = "bfloat16"
    diffusion_dtype: str = "float16"    # float16 works best with 8-bit quantization
    use_8bit_editor: bool = True        # use 8-bit quantization for image editor (saves ~50% VRAM)
    hf_token: str | None = None         # set via HF_TOKEN env var if needed
    low_vram: bool = True               # if True, load/offload models one at a time (for <32GB VRAM)

    model_config = {"arbitrary_types_allowed": True}


def get_config(**overrides) -> Config:
    return Config(**overrides)
