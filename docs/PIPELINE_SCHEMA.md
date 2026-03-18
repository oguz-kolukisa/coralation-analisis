# Coralation Pipeline - Single Class Analysis

This document shows the complete flow for analyzing ONE class (e.g., "tabby cat").

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CORALATION PIPELINE                                  │
│                     (Single Class Analysis)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Class name (e.g., "tabby cat")                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   PHASE 1     │           │   PHASE 2     │           │   PHASE 3     │
│  Knowledge    │           │    Sample     │           │   Baseline    │
│  Discovery    │           │    Images     │           │Classification │
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 4                                           │
│                    Feature Discovery (VLM)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 5                                           │
│                  Edit Generation & Application                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 6                                           │
│               Feature Classification & Final Analysis                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: ClassAnalysisResult + JSON checkpoint + HTML/MD reports            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Phase Breakdown

### PHASE 1: Knowledge-Based Feature Discovery

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Knowledge Discovery                             │
│                         (No images needed)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │           VLM                 │
                    │   (Qwen2.5-VL-7B-Instruct)   │
                    └───────────────────────────────┘
                                    │
                                    │ Prompt: "What objects/environments
                                    │ are commonly ASSOCIATED with
                                    │ '{class_name}' but NOT part of it?"
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Knowledge-Based Features    │
                    │                               │
                    │ Example for "tabby cat":      │
                    │ • yarn balls                  │
                    │ • milk bowls                  │
                    │ • scratching posts            │
                    │ • indoor living rooms         │
                    │ • human laps                  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                         [Potential shortcuts to test]
```

---

### PHASE 2: Sample Images

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: Sample Images                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  POSITIVE SAMPLES │           │  NEGATIVE SAMPLES │
        │   (ImageNet)      │           │  (Confusing classes)
        └───────────────────┘           └───────────────────┘
                    │                               │
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ • inspect_samples │           │ From classifier's │
        │   (e.g., 10)      │           │ top-k predictions │
        │   for discovery   │           │                   │
        │                   │           │ e.g., for cat:    │
        │ • samples_per_class│          │ • tiger cat       │
        │   (e.g., 5)       │           │ • Egyptian cat    │
        │   for editing     │           │ • lynx            │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Image Collections        │
                    │                               │
                    │ • annotated_inspect (10 imgs) │
                    │ • annotated_positives (5 imgs)│
                    │ • annotated_negatives (5 imgs)│
                    └───────────────────────────────┘
```

---

### PHASE 3: Baseline Classification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: Baseline Classification                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         Classifier            │
                    │     (ResNet-50 ImageNet)      │
                    └───────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │  Prediction   │       │   Top-K       │       │   Grad-CAM    │
    │  Confidence   │       │  Predictions  │       │  Attention    │
    │               │       │               │       │    Maps       │
    │  e.g., 0.87   │       │ tabby: 0.87   │       │               │
    │               │       │ tiger: 0.05   │       │  [Heatmap]    │
    │               │       │ lynx: 0.03    │       │  Red = focus  │
    └───────────────┘       └───────────────┘       └───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      baseline_results         │
                    │                               │
                    │ For each image:               │
                    │ • image_path                  │
                    │ • true_label                  │
                    │ • predicted_label             │
                    │ • class_confidence            │
                    │ • top_k predictions           │
                    │ • gradcam_image               │
                    └───────────────────────────────┘
```

---

### PHASE 4: Feature Discovery (VLM)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: Feature Discovery                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │                    FOR EACH IMAGE                        │
        │                (inspect_samples images)                  │
        └─────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  Original Image   │           │   Grad-CAM Map    │
        │                   │           │                   │
        │   [Cat photo]     │           │   [Heatmap]       │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │            VLM                │
                    │   "Analyze this image and    │
                    │    Grad-CAM. Identify ALL    │
                    │    visual features."          │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      DetectedFeature[]        │
                    │                               │
                    │ INTRINSIC (object parts):     │
                    │ • pointed ears (high attn)    │
                    │ • striped fur (high attn)     │
                    │ • whiskers (medium attn)      │
                    │ • tail shape (low attn)       │
                    │                               │
                    │ CONTEXTUAL (background):      │
                    │ • grassy background           │
                    │ • wooden floor                │
                    │ • indoor lighting             │
                    └───────────────────────────────┘
```

---

### PHASE 5: Edit Generation & Application

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                PHASE 5: Edit Generation & Application                       │
└─────────────────────────────────────────────────────────────────────────────┘

                           STEP 5A: Generate Edits
                    ┌───────────────────────────────┐
                    │            VLM                │
                    │                               │
                    │  For each detected feature,   │
                    │  generate edit instruction:   │
                    │                               │
                    │  "Remove the pointed ears     │
                    │   completely, blend smoothly  │
                    │   with surrounding fur"       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     EditInstruction[]         │
                    │                               │
                    │ POSITIVE (feature removal):   │
                    │ • Remove ears → test ears     │
                    │ • Remove stripes → test fur   │
                    │ • Replace background → test   │
                    │   background dependency       │
                    │                               │
                    │ NEGATIVE (feature addition):  │
                    │ • Add cat ears to dog image   │
                    │ • Add stripes to solid cat    │
                    └───────────────────────────────┘
                                    │
                                    ▼
                           STEP 5B: Deduplicate
                    ┌───────────────────────────────┐
                    │  Remove similar instructions  │
                    │  (similarity > 0.7 threshold) │
                    └───────────────────────────────┘
                                    │
                                    ▼
                           STEP 5C: Apply Edits
        ┌─────────────────────────────────────────────────────────┐
        │                 FOR EACH EDIT INSTRUCTION                │
        └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        Image Editor           │
                    │   (FLUX or InstructPix2Pix)   │
                    │                               │
                    │   Generate N versions         │
                    │   (generations_per_edit = 3)  │
                    └───────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │  Generation 1 │       │  Generation 2 │       │  Generation 3 │
    │   seed=100    │       │   seed=101    │       │   seed=102    │
    └───────────────┘       └───────────────┘       └───────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                           STEP 5D: Measure Impact
                    ┌───────────────────────────────┐
                    │         Classifier            │
                    │                               │
                    │  Re-classify edited images    │
                    │  Compute confidence delta     │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Statistical Validation   │
                    │                               │
                    │  • t-test (p-value)           │
                    │  • Cohen's d (effect size)    │
                    │  • confirmed = significant?   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        EditResult             │
                    │                               │
                    │ instruction: "Remove ears"    │
                    │ original_conf: 0.87           │
                    │ mean_edited_conf: 0.45        │
                    │ mean_delta: -0.42             │
                    │ p_value: 0.003                │
                    │ cohens_d: 2.1 (large)         │
                    │ confirmed: TRUE               │
                    └───────────────────────────────┘
```

---

### PHASE 6: Feature Classification & Final Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PHASE 6: Feature Classification & Final Analysis               │
└─────────────────────────────────────────────────────────────────────────────┘

                        STEP 6A: VLM Classification
                    ┌───────────────────────────────┐
                    │            VLM                │
                    │                               │
                    │  For each tested edit:        │
                    │  "Is this feature INTRINSIC   │
                    │   or CONTEXTUAL for           │
                    │   '{class_name}'?"            │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Feature Classification      │
                    │                               │
                    │ "Remove ears"                 │
                    │   → feature_type: intrinsic   │
                    │   → feature_name: "Ears"      │
                    │                               │
                    │ "Replace background"          │
                    │   → feature_type: contextual  │
                    │   → feature_name: "Background"│
                    └───────────────────────────────┘
                                    │
                                    ▼
                        STEP 6B: Determine Shortcuts
        ┌─────────────────────────────────────────────────────────┐
        │                   DECISION MATRIX                        │
        │                                                          │
        │   Feature Type    │  Delta Direction  │    Result        │
        │ ──────────────────┼───────────────────┼─────────────────│
        │   INTRINSIC       │  Negative (↓)     │  ✓ ESSENTIAL    │
        │   INTRINSIC       │  Positive (↑)     │  ⚠ UNEXPECTED   │
        │   CONTEXTUAL      │  Negative (↓)     │  🚨 SHORTCUT    │
        │   CONTEXTUAL      │  Positive (↑)     │  ✓ ROBUST       │
        │   MODIFICATION    │  Positive (↑)     │  ⚠ SPURIOUS     │
        └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        STEP 6C: Final VLM Analysis
                    ┌───────────────────────────────┐
                    │            VLM                │
                    │                               │
                    │  Summarize all findings:      │
                    │  • robustness_score (1-10)    │
                    │  • risk_level (LOW/MED/HIGH)  │
                    │  • vulnerabilities            │
                    │  • recommendations            │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     ClassAnalysisResult       │
                    │                               │
                    │ class_name: "tabby cat"       │
                    │ risk_level: "MEDIUM"          │
                    │ robustness_score: 6           │
                    │                               │
                    │ confirmed_hypotheses:         │
                    │ • Background (SHORTCUT)       │
                    │                               │
                    │ essential_features:           │
                    │ • Ears, Stripes, Whiskers     │
                    │                               │
                    │ spurious_features:            │
                    │ • Background, Lighting        │
                    └───────────────────────────────┘
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  "tabby cat"
       │
       ▼
  ┌─────────┐    ┌─────────────┐
  │   VLM   │───▶│ Knowledge   │ (yarn, bowls, scratching posts...)
  └─────────┘    │ Features    │
                 └─────────────┘
       │
       ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  ImageNet   │───▶│  Positive   │    │  Negative   │
  │   Dataset   │    │  Samples    │    │  Samples    │
  └─────────────┘    │  (10 imgs)  │    │  (5 imgs)   │
                     └─────────────┘    └─────────────┘
                            │                  │
                            ▼                  ▼
                     ┌─────────────────────────────┐
                     │        Classifier           │
                     │  (Confidence + Grad-CAM)    │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │           VLM               │
                     │   (Feature Discovery)       │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │   Detected Features         │
                     │   (intrinsic + contextual)  │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │           VLM               │
                     │   (Generate Edits)          │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │       Image Editor          │
                     │   (Apply Edits × N gens)    │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │        Classifier           │
                     │   (Re-classify edited)      │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │   Statistical Validator     │
                     │   (t-test, Cohen's d)       │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │           VLM               │
                     │  (Classify & Analyze)       │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │         Reporter            │
                     │   (HTML + MD + JSON)        │
                     └─────────────────────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │          OUTPUTS            │
                     │  • analysis.json            │
                     │  • report.html              │
                     │  • report.md                │
                     │  • edited images            │
                     │  • gradcam heatmaps         │
                     └─────────────────────────────┘
```

---

## Model Usage Summary

| Phase | Model | Purpose |
|-------|-------|---------|
| 1 | VLM | Knowledge-based feature discovery |
| 2 | - | Dataset sampling (no model) |
| 3 | Classifier | Baseline predictions + Grad-CAM |
| 4 | VLM | Image-based feature discovery |
| 5a | VLM | Generate edit instructions |
| 5b | Editor | Apply image edits |
| 5c | Classifier | Re-classify edited images |
| 6a | VLM | Classify features as intrinsic/contextual |
| 6b | VLM | Final analysis and summary |

---

## Key Insight: Shortcut Detection Logic

```
                    ┌─────────────────────────────────────┐
                    │        SHORTCUT DETECTION           │
                    └─────────────────────────────────────┘

    Original Image ───────────────────────────────▶ Confidence: 0.87
         │
         │  Edit: "Replace grassy background
         │         with white studio"
         │
         ▼
    Edited Image ─────────────────────────────────▶ Confidence: 0.52
                                                          │
                                                          ▼
                                                   Delta: -0.35
                                                          │
                    ┌─────────────────────────────────────┴─────────┐
                    │                                               │
                    ▼                                               ▼
            Feature Type?                               Impact Significant?
                    │                                               │
            ┌───────┴───────┐                               ┌───────┴───────┐
            │               │                               │               │
            ▼               ▼                               ▼               ▼
       INTRINSIC      CONTEXTUAL                          YES              NO
            │               │                               │               │
            │               │                               │               │
            ▼               ▼                               ▼               ▼
       ✓ Expected    🚨 SHORTCUT!                     Confirmed       Not confirmed
       (model uses   (model wrongly                   hypothesis      (edit failed
        object       relies on                                        or no effect)
        features)    background)
```

---

## Output Files Structure

```
output_dir/
├── tabby_cat/
│   ├── analysis.json              # Checkpoint for this class
│   ├── pos_0_original.jpg         # Original positive sample
│   ├── pos_0_gradcam.jpg          # Grad-CAM heatmap
│   ├── pos_0_iter0_edit_0_gen_0.jpg  # Edited image (gen 0)
│   ├── pos_0_iter0_edit_0_gen_1.jpg  # Edited image (gen 1)
│   ├── pos_0_iter0_edit_0_gen_2.jpg  # Edited image (gen 2)
│   ├── neg_0_original.jpg         # Original negative sample
│   └── ...
├── analysis_results.json          # Consolidated results
├── report.html                    # Interactive HTML report
└── report.md                      # Markdown report
```
