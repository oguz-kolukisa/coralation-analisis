# Coralation Architecture

> **IMPORTANT FOR CLAUDE**: When modifying the pipeline, you MUST also update:
> 1. This ARCHITECTURE.md file
> 2. The HTML report template in `reporter.py` (Methodology tab)
> 3. The research description if methodology changes significantly
> See `CLAUDE.md` for detailed instructions.

## Overview

Coralation is an automated, **domain-agnostic** bias/shortcut discovery tool for image classifiers. It uses a pipeline approach to identify features that models incorrectly rely upon for classification. The system works for ANY image classification domain (animals, vehicles, fashion, medical, furniture, etc.) without requiring domain-specific configuration.

## Pipeline Flow

```
1. Sample Collection
   └── dataset.py (ImageNetSampler)
       - Fetches positive samples (correctly classified as target class)
       - Fetches negative samples (VLM-selected from other classes)

2. Baseline Classification
   └── classifier.py (ImageNetClassifier)
       - Classifies all samples
       - Generates attention maps (Score-CAM/Grad-CAM/Grad-CAM++)

3. Feature Discovery (Iterative)
   └── vlm.py (QwenVLAnalyzer)
       - Analyzes images + attention maps
       - Identifies intrinsic vs contextual features
       - Generates SPECIFIC edit hypotheses (no alternatives like "or")
       - Classifies features as intrinsic (object parts) or contextual (background)

4. Counterfactual Editing
   └── editor.py (ImageEditor)
       - Applies edit instructions using Qwen-Image-Edit (default) or InstructPix2Pix
       - Creates multiple generations per edit for robustness
       - Supports both Qwen and InstructPix2Pix backends

5. Edit Verification (NEW)
   └── vlm.py (QwenVLAnalyzer.verify_edit)
       - VLM compares original vs edited image
       - Confirms edit was actually applied
       - Flags failed edits for review

6. Impact Measurement + Attention Diff
   └── classifier.py + models/statistical_validator.py + models/attention_maps.py
       - Re-classifies edited images
       - Optionally computes Grad-CAM on edited images (compute_edit_gradcam)
       - Generates attention diff heatmaps (blue=lost focus, red=gained focus)
       - Measures confidence delta
       - Statistical validation (t-test, Cohen's d)
       - Determines if change is statistically & practically significant

7. Report Generation
   └── reporter.py (Reporter)
       - Generates HTML/Markdown/JSON reports
       - Tabbed interface: Overview, Feature Analysis, Class Details, Methodology
       - Shows shortcut evidence with images
       - Includes pipeline configuration
```

## Module Responsibilities

### Core Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `config.py` | `Config` | Pydantic configuration model |
| `pipeline.py` | `AnalysisPipeline` | Orchestrates the full analysis |
| `model_manager.py` | `ModelManager` | Model lifecycle (load/offload) for VRAM efficiency |
| `classifier.py` | `ImageNetClassifier` | Image classification + attention maps |
| `vlm.py` | `QwenVLAnalyzer` | Vision-language model for hypothesis generation + edit verification |
| `editor.py` | `ImageEditor` | Image editing (Qwen-Image-Edit or InstructPix2Pix) |
| `dataset.py` | `ImageNetSampler` | ImageNet dataset sampling |
| `reporter.py` | `Reporter` | Report generation (HTML/MD/JSON) with tabbed interface |

### Models Subpackage (`src/models/`)

| Module | Purpose |
|--------|---------|
| `attention_maps.py` | Grad-CAM, Grad-CAM++, Score-CAM generators |
| `statistical_validator.py` | T-test and effect size calculations |
| `feature_analyzer.py` | Feature importance analysis |

## Key Data Structures

### ClassAnalysisResult (pipeline.py)
```python
@dataclass
class ClassAnalysisResult:
    class_name: str
    baseline_results: list[dict]               # Original sample classifications
    edit_results: list[EditResult]             # All edits tested
    confirmed_hypotheses: list[EditResult]     # Statistically confirmed biases (derived)
    detected_features: list[dict]             # VLM-identified features
    ...
    # Methods: apply_final_analysis(), finalize(), to_dict(), from_dict()
```

### ImageSet (pipeline.py)
```python
@dataclass
class ImageSet:
    """Groups all sampled images for one class analysis."""
    inspect: list[tuple[Image, str]]       # For feature discovery
    edit: list[tuple[Image, str]]          # For counterfactual editing
    negative: list[tuple[Image, str]]      # From confusing classes
    confusing_classes: list[str]
    annotated_inspect: list                # Populated during baseline
    annotated_negatives: list              # Populated during baseline
```

### EditResult (pipeline.py)
```python
@dataclass
class EditResult:
    instruction: str        # "Remove the wooden background"
    hypothesis: str         # "Background affects classification"
    generations: list       # Multiple edited image results
    mean_delta: float       # Average confidence change
    p_value: float          # Statistical significance
    cohens_d: float         # Effect size
    confirmed: bool         # Whether edit had significant impact
    # Methods: from_dict()
```

### ModelManager (model_manager.py)
```python
class ModelManager:
    """Loads/offloads models for VRAM efficiency."""
    def _ensure_only(model_name)             # Auto-offloads all others in low_vram
    def classifier() -> ImageNetClassifier   # Calls _ensure_only("classifier")
    def vlm() -> QwenVLAnalyzer              # Calls _ensure_only("vlm")
    def editor() -> ImageEditor              # Calls _ensure_only("editor")
    def sampler() -> ImageNetSampler
    def offload_all()
```

### DetectedFeature (vlm.py)
```python
@dataclass
class DetectedFeature:
    name: str               # "wooden background"
    category: str           # object_part, texture, color, context
    feature_type: str       # intrinsic or contextual
    gradcam_attention: str  # high, medium, low
    is_shortcut: bool       # True if contextual feature affects classification
```

## Feature Classification

The pipeline is **domain-agnostic** - it works for ANY image classification domain, not just animals or vehicles. Feature classification is performed by the VLM semantically, not by keyword matching.

**Intrinsic Features** (green in reports):
- Parts of the object itself (defining characteristics)
- Features that would remain if you isolated just the object
- Model SHOULD rely on these
- High impact = expected behavior

**Contextual Features** (red in reports):
- Background, environment, lighting, co-occurring objects
- Features that describe WHERE/HOW the object is photographed, not WHAT it is
- Model should NOT rely on these
- High impact = SHORTCUT/BIAS

**VLM-Based Classification**:
- All feature classification is performed by the VLM using semantic understanding
- No hardcoded keywords or domain-specific patterns
- Works equally well for fashion, medical images, furniture, etc.

## Report Structure (HTML)

The HTML report uses a tabbed interface:

1. **Overview Tab**: Summary stats, class table, quick navigation
2. **Feature Analysis Tab**: Feature impact summary, intrinsic vs contextual
3. **Class Details Tab**: Per-class images, edits, confirmed shortcuts
4. **Methodology Tab**: Pipeline explanation, config, interpretation guide

## Configuration Options

Key config parameters in `config.py`:

```python
# Models
classifier_model: str = "resnet50"
vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
editor_model: str = "black-forest-labs/FLUX.2-klein-9b-kv"

# Dataset (domain-agnostic)
hf_dataset: str = "ILSVRC/imagenet-1k"
hf_dataset_split: str = "validation"
image_field: str = "image"           # Field name for image data
label_field: str = "label"           # Field name for labels
custom_classes: list[str] | None = None  # Override DEFAULT_CLASSES

# Analysis
samples_per_class: int = 5
negative_samples: int = 5
confidence_delta_threshold: float = 0.15
iterations: int = 2
generations_per_edit: int = 3

# Attention
attention_method: str = "scorecam"  # or gradcam, gradcam++
compute_edit_gradcam: bool = True   # Grad-CAM on edited images (enables attention diff)

# Statistical validation
use_statistical_validation: bool = True
statistical_alpha: float = 0.05
min_effect_size: float = 0.5

# Risk classification thresholds
risk_high_threshold: float = 0.30    # >30% confirmation rate = HIGH risk
risk_medium_threshold: float = 0.10  # >10% = MEDIUM, else LOW

# Processing thresholds
dedup_similarity_threshold: float = 0.70  # Edit deduplication
confusing_class_min_conf: float = 0.01    # Min confidence for confusing classes
spurious_positive_delta: float = 0.10     # Delta threshold for spurious correlations

# Edit verification
verify_edits: bool = False  # VLM verifies edits were applied correctly
```

## VRAM Management

The `ModelManager` (`model_manager.py`) handles VRAM management automatically.
Each `model.classifier()` / `model.vlm()` / `model.editor()` call ensures
the requested model is loaded and offloads conflicting models in low_vram mode.

Low-VRAM mode (`--low-vram`, default):
- Models are loaded/offloaded sequentially via ModelManager
- Only one large model in VRAM at a time
- Works with 16-24GB VRAM

Batch mode (`--batch`):
- Phase-first pipeline: runs each phase for ALL classes before moving to the next
- Only 6 model swaps total regardless of class count (vs ~10 per class in sequential mode)
- Recommended for multi-class analysis to avoid redundant model load/offload cycles
- Uses `BatchClassState` to track per-class progress across phases

High-VRAM mode (`--high-vram`):
- All models kept loaded
- Faster but requires 40GB+ VRAM

## Extension Points

### Adding a New Attention Method
1. Implement in `src/models/attention_maps.py`
2. Add to `get_attention_generator()` factory
3. Update config validation
4. **Update ARCHITECTURE.md and reporter.py Methodology tab**

### Adding a New Editor
1. Create new class or add backend in `src/editor.py`
2. Implement `edit(image, instruction)` method
3. Add offload/load_to_gpu for VRAM management
4. **Update ARCHITECTURE.md and reporter.py Methodology tab**

### Adding New Report Formats
1. Add template to `reporter.py`
2. Implement `generate_<format>()` method
3. Include in `generate_all()`

### Adding New Pipeline Steps
1. Implement the step in appropriate module
2. Integrate into `pipeline.py`
3. **MUST update:**
   - This ARCHITECTURE.md (Pipeline Flow section)
   - `reporter.py` Methodology tab (Pipeline Steps section)
   - Config if new parameters needed
