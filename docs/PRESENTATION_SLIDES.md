# CORALATION - Presentation Slides

## Quick Reference Diagrams for Technical Presentations

---

## Slide 1: Problem Statement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE SHORTCUT LEARNING PROBLEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Training Data                     What Model Learns                   │
│   ┌─────────────────┐              ┌─────────────────┐                 │
│   │   [COW ON       │              │                 │                 │
│   │    GRASS]       │    ───►      │  COW = GRASS?   │                 │
│   │                 │              │                 │                 │
│   │   label: "cow"  │              │  (SHORTCUT!)    │                 │
│   └─────────────────┘              └─────────────────┘                 │
│                                                                         │
│   Problem: Model uses BACKGROUND instead of actual COW features        │
│                                                                         │
│   Real-world failure:                                                   │
│   ┌─────────────────┐              ┌─────────────────┐                 │
│   │   [COW ON       │              │                 │                 │
│   │    BEACH]       │    ───►      │  "NOT A COW"    │  ✗ WRONG       │
│   │                 │              │                 │                 │
│   └─────────────────┘              └─────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 2: Solution Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CORALATION: OUR SOLUTION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   AUTOMATED DISCOVERY OF MODEL SHORTCUTS                                │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│   │ Analyze  │───►│ Discover │───►│   Edit   │───►│ Validate │        │
│   │ Attention│    │ Features │    │  Images  │    │  Impact  │        │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘        │
│        │               │               │               │               │
│        ▼               ▼               ▼               ▼               │
│   Score-CAM        VLM (7B)      Diffusion       Statistics           │
│   Heatmaps         Analysis       Model          (t-test)             │
│                                                                         │
│   KEY INSIGHT: If removing a CONTEXTUAL feature (like grass)          │
│   decreases confidence, that's a SHORTCUT we found!                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 3: Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE FLOW                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    INPUT: "tabby cat"                           │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: Sample images from ImageNet                          │   │
│  │  • 5 positive (tabby cats)                                      │   │
│  │  • 5 negative (similar: Egyptian cat, tiger cat, etc.)         │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: Run classifier + Generate attention maps             │   │
│  │  ResNet-50 → Score-CAM heatmap                                  │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3: VLM analyzes image + attention                        │   │
│  │  Identifies: ears, whiskers, fur, background, etc.             │   │
│  │  Generates: "Remove the pointed ears", "Change background"...  │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 4: Apply edits with diffusion model                     │   │
│  │  Measure confidence change (delta)                              │   │
│  │  Multiple generations for robustness                            │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 5: Statistical validation                                │   │
│  │  t-test (p < 0.05) + Cohen's d (|d| > 0.5)                     │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Risk assessment + confirmed shortcuts                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 4: Feature Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE CLASSIFICATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                    DETECTED FEATURES                          │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│              ┌───────────────┴───────────────┐                         │
│              │                               │                         │
│              ▼                               ▼                         │
│   ┌─────────────────────┐         ┌─────────────────────┐             │
│   │  INTRINSIC FEATURES │         │ CONTEXTUAL FEATURES │             │
│   │  (Part of object)   │         │ (Background/Setting)│             │
│   ├─────────────────────┤         ├─────────────────────┤             │
│   │ • Pointed ears      │         │ • Wooden floor      │             │
│   │ • Whiskers          │         │ • Indoor lighting   │             │
│   │ • Fur pattern       │         │ • Furniture         │             │
│   │ • Eye shape         │         │ • Window            │             │
│   │ • Body silhouette   │         │ • Color cast        │             │
│   └─────────────────────┘         └─────────────────────┘             │
│              │                               │                         │
│              ▼                               ▼                         │
│   If removing DECREASES          If removing DECREASES                 │
│   confidence:                    confidence:                           │
│                                                                         │
│   ┌─────────────────────┐         ┌─────────────────────┐             │
│   │     EXPECTED!       │         │  SHORTCUT FOUND!    │             │
│   │  Model is correct   │         │   Model is BIASED   │             │
│   │     (✓ Good)        │         │     (✗ Problem)     │             │
│   └─────────────────────┘         └─────────────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 5: Counterfactual Editing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COUNTERFACTUAL EDITING                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Edit: "Remove the pointed ears from the cat"                         │
│                                                                         │
│   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐   │
│   │  ORIGINAL   │          │   EDITED    │          │   RESULT    │   │
│   │             │  ──────► │             │  ──────► │             │   │
│   │   [CAT]     │  Qwen    │  [CAT w/o   │  ResNet  │  Confidence │   │
│   │   Conf:     │  Image   │   ears]     │    50    │   Change    │   │
│   │   0.847     │  Edit    │             │          │   -0.41     │   │
│   └─────────────┘          └─────────────┘          └─────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Interpretation:                                                 │  │
│   │                                                                 │  │
│   │ • Ears are INTRINSIC feature (part of cat)                     │  │
│   │ • Confidence DECREASED significantly (-0.41)                    │  │
│   │ • This is EXPECTED behavior                                     │  │
│   │ • Model correctly relies on ears → NOT a shortcut              │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Edit: "Replace wooden floor with white background"                   │
│                                                                         │
│   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐   │
│   │  ORIGINAL   │          │   EDITED    │          │   RESULT    │   │
│   │   [CAT ON   │  ──────► │   [CAT ON   │  ──────► │  Confidence │   │
│   │   WOOD]     │          │    WHITE]   │          │   Change    │   │
│   │   Conf:     │          │             │          │   -0.22     │   │
│   │   0.847     │          │             │          │  SHORTCUT!  │   │
│   └─────────────┘          └─────────────┘          └─────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Interpretation:                                                 │  │
│   │                                                                 │  │
│   │ • Floor is CONTEXTUAL (not part of cat)                        │  │
│   │ • Confidence DECREASED (-0.22)                                  │  │
│   │ • This is UNEXPECTED - model shouldn't care about floor!       │  │
│   │ • SHORTCUT DETECTED: Model relies on indoor floor texture     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 6: Statistical Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STATISTICAL VALIDATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Why multiple generations?                                             │
│   • Diffusion models are stochastic                                     │
│   • Same prompt → different results                                     │
│   • Need statistical confidence                                         │
│                                                                         │
│   Example: 3 generations with different seeds                           │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Seed 42:  0.85 → 0.42  =  -0.43                               │  │
│   │  Seed 52:  0.85 → 0.38  =  -0.47                               │  │
│   │  Seed 62:  0.85 → 0.51  =  -0.34                               │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                      VALIDATION TESTS                           │  │
│   ├─────────────────────────────────────────────────────────────────┤  │
│   │                                                                 │  │
│   │   1. ONE-SAMPLE T-TEST                                         │  │
│   │      ┌─────────────────────────────────────────────────────┐   │  │
│   │      │  H₀: mean(delta) = 0  (no effect)                   │   │  │
│   │      │  H₁: mean(delta) ≠ 0  (significant effect)          │   │  │
│   │      │                                                     │   │  │
│   │      │  Result: p = 0.004 < 0.05  ✓ SIGNIFICANT           │   │  │
│   │      └─────────────────────────────────────────────────────┘   │  │
│   │                                                                 │  │
│   │   2. COHEN'S D (Effect Size)                                   │  │
│   │      ┌─────────────────────────────────────────────────────┐   │  │
│   │      │  d = mean / std = -0.41 / 0.065 = -6.3              │   │  │
│   │      │                                                     │   │  │
│   │      │  |d| > 0.8 → LARGE effect  ✓ PRACTICAL             │   │  │
│   │      └─────────────────────────────────────────────────────┘   │  │
│   │                                                                 │  │
│   │   CONFIRMED = (p < 0.05) AND (|d| >= 0.5)  ✓                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 7: Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SYSTEM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         MODELS                                  │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │  ResNet-50  │  │ Qwen2.5-VL  │  │ Qwen-Image  │             │   │
│  │  │  Classifier │  │   7B VLM    │  │    Edit     │             │   │
│  │  ├─────────────┤  ├─────────────┤  ├─────────────┤             │   │
│  │  │ 25M params  │  │  7B params  │  │ 20B params  │             │   │
│  │  │   ~2GB      │  │   ~14GB     │  │  ~8GB (FP8) │             │   │
│  │  │             │  │             │  │             │             │   │
│  │  │ Inference   │  │ Feature     │  │ Counter-    │             │   │
│  │  │ + Attention │  │ Discovery   │  │ factual     │             │   │
│  │  │ Maps        │  │ + Analysis  │  │ Editing     │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY MANAGEMENT                            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  Low-VRAM Mode: Sequential loading (one model at a time)       │   │
│  │                                                                 │   │
│  │  Time ──────────────────────────────────────────────────────►  │   │
│  │                                                                 │   │
│  │  ████████              ResNet (2GB)                            │   │
│  │          ██████████████████████████  Qwen-VL (14GB)            │   │
│  │                                    ████████████  Editor (8GB)  │   │
│  │                                                 ███  ResNet    │   │
│  │                                                    ███████ VLM │   │
│  │                                                                 │   │
│  │  FP8 Quantization: 50% VRAM savings for image editor          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 8: Example Output

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXAMPLE OUTPUT                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Class: "tabby cat"                                                     │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Robustness Score: 6/10                                                 │
│  Risk Level: MEDIUM                                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ CONFIRMED SHORTCUTS (Spurious Correlations)                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  1. Indoor wooden floor texture                                 │   │
│  │     Delta: -0.22  |  p=0.012  |  d=1.8                         │   │
│  │     Severity: MEDIUM                                            │   │
│  │                                                                 │   │
│  │  2. Warm indoor lighting                                        │   │
│  │     Delta: -0.18  |  p=0.034  |  d=1.2                         │   │
│  │     Severity: LOW                                               │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LEGITIMATE FEATURES (Correctly Used)                            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ✓ Pointed ears (delta: -0.41)                                 │   │
│  │  ✓ Whiskers (delta: -0.38)                                     │   │
│  │  ✓ Fur pattern (delta: -0.35)                                  │   │
│  │  ✓ Eye shape (delta: -0.29)                                    │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ RECOMMENDATIONS                                                 │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  1. Augment training data with varied backgrounds               │   │
│  │  2. Apply background randomization during training              │   │
│  │  3. Consider adversarial training on discovered shortcuts      │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 9: Key Metrics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY METRICS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Performance (per class):                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Edits generated:          10-15                                │   │
│  │  Generations per edit:     3                                    │   │
│  │  Total images edited:      30-45                                │   │
│  │  Time per class:           ~15-20 minutes (24GB GPU)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Statistical Thresholds:                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Significance level (α):   0.05                                 │   │
│  │  Min effect size (d):      0.5                                  │   │
│  │  Min confidence delta:     0.15                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Hardware Requirements:                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Minimum VRAM:            16GB (with aggressive offloading)    │   │
│  │  Recommended VRAM:        24GB                                  │   │
│  │  High-performance:        40GB+ (all models loaded)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 10: Future Directions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       FUTURE DIRECTIONS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. EXPAND MODEL SUPPORT                                         │   │
│  │    • Vision Transformers (ViT, DeiT, Swin)                     │   │
│  │    • Object detection models (YOLO, Faster R-CNN)              │   │
│  │    • Segmentation models                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 2. DOMAIN-SPECIFIC ANALYSIS                                     │   │
│  │    • Medical imaging (X-ray, MRI)                               │   │
│  │    • Autonomous driving                                         │   │
│  │    • Satellite imagery                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 3. REMEDIATION PIPELINE                                         │   │
│  │    • Automatic data augmentation based on discovered shortcuts │   │
│  │    • Fine-tuning recipes                                        │   │
│  │    • Continuous monitoring in production                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 4. SCALE & INTEGRATION                                          │   │
│  │    • MLOps pipeline integration                                 │   │
│  │    • Web-based interactive dashboard                            │   │
│  │    • Batch processing for enterprise                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      QUICK REFERENCE CARD                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RUN COMMAND:                                                           │
│  $ uv run python main.py --classes 5 --samples 5 --iterations 2        │
│                                                                         │
│  MODELS:                                                                │
│  • Classifier: ResNet-50 (ImageNet pretrained)                         │
│  • VLM: Qwen2.5-VL-7B-Instruct                                         │
│  • Editor: Qwen-Image-Edit (FP8 quantized)                             │
│                                                                         │
│  KEY CONCEPTS:                                                          │
│  • Intrinsic feature → Part of object → Expected to affect confidence │
│  • Contextual feature → Background → Should NOT affect confidence     │
│  • Shortcut = Contextual feature that DOES affect confidence          │
│                                                                         │
│  VALIDATION:                                                            │
│  • p < 0.05 (statistical significance)                                 │
│  • |d| > 0.5 (practical significance)                                  │
│  • Both required for confirmation                                       │
│                                                                         │
│  OUTPUT:                                                                │
│  • output/report.html (visual report)                                  │
│  • output/report.md (text report)                                      │
│  • output/analysis_results.json (raw data)                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
