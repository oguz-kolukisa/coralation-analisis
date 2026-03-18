# Coralation Architecture Diagram

## System Overview

```
                         ┌──────────────────────┐
                         │      main.py         │
                         │   (CLI + argparse)   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │     Config           │
                         │  (Pydantic model)    │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     AnalysisPipeline          │
                    │       (pipeline.py)           │
                    │                               │
                    │  Orchestrates 7 phases        │
                    │  per class, manages           │
                    │  checkpoints & progress       │
                    └───────┬───────────┬───────────┘
                            │           │
               ┌────────────▼──┐   ┌────▼──────────────┐
               │ ModelManager  │   │ StatisticalValidator│
               │ (VRAM mgmt)  │   │ (t-test, Cohen's d)│
               └──┬──┬──┬──┬──┘   └────────────────────┘
                  │  │  │  │
         ┌────────┘  │  │  └──────────┐
         │           │  │             │
    ┌────▼───┐ ┌─────▼──┐ ┌───▼───┐ ┌▼────────┐
    │Classif.│ │  VLM   │ │Editor │ │ Sampler │
    │ResNet50│ │Qwen-VL │ │FLUX.2 │ │ImageNet │
    └────────┘ └────────┘ └───────┘ └─────────┘
```

## Pipeline Phases (per class)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AnalysisPipeline.run_class()                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─── Phase 1 ──────────────────────────────────────────────────┐  │
│  │ KNOWLEDGE DISCOVERY                                          │  │
│  │ VLM asks: "What shortcuts might exist for {class}?"          │  │
│  │ Input:  class_name, all_classes                              │  │
│  │ Output: knowledge_based_features (e.g. "yarn ball" for cat)  │  │
│  │ Model:  QwenVLAnalyzer                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Phase 2 ──────────────▼───────────────────────────────────┐  │
│  │ SAMPLE COLLECTION                                            │  │
│  │ Fetch positive images (target class)                         │  │
│  │ Identify confusing classes (from classifier top-k)           │  │
│  │ Fetch negative images (confusing classes)                    │  │
│  │ Output: ImageSet { inspect, edit, negative }                 │  │
│  │ Models: ImageNetSampler, ImageNetClassifier                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Phase 3 ──────────────▼───────────────────────────────────┐  │
│  │ BASELINE CLASSIFICATION + GRAD-CAM                           │  │
│  │ Classify all images, generate attention heatmaps             │  │
│  │ Save originals + grad-cam overlays to disk                   │  │
│  │ Output: baseline_results[], annotated images                 │  │
│  │ Model:  ImageNetClassifier (+ ScoreCAM/GradCAM)              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Phase 4 ──────────────▼───────────────────────────────────┐  │
│  │ IMAGE-BASED FEATURE DISCOVERY                                │  │
│  │ VLM analyzes each image + attention map                      │  │
│  │ Identifies intrinsic vs contextual features                  │  │
│  │ Output: DiscoveredFeatures { detected, essential, summary }  │  │
│  │ Model:  QwenVLAnalyzer                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Phase 5 ──────────────▼───────────────────────────────────┐  │
│  │ EDIT GENERATION                                              │  │
│  │ VLM creates specific edit instructions per feature:          │  │
│  │   Positive: "Remove the pointed ears" (test reliance)        │  │
│  │   Negative: "Add tabby stripes" (test feature addition)      │  │
│  │ Deduplicate similar edits (SequenceMatcher >= 0.7)           │  │
│  │ Output: list[EditInput]                                      │  │
│  │ Model:  QwenVLAnalyzer                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Phase 6 ──────────────▼───────────────────────────────────┐  │
│  │ COUNTERFACTUAL TESTING + IMPACT MEASUREMENT                  │  │
│  │                                                              │  │
│  │  For each edit instruction:                                  │  │
│  │    ┌──────────────────────────────────────────────┐          │  │
│  │    │ Generate N variants (different seeds)        │          │  │
│  │    │  Image ──► Editor.edit() ──► Edited Image    │          │  │
│  │    │  Edited ──► Classifier ──► new_confidence    │          │  │
│  │    │  delta = new_conf - orig_conf                │          │  │
│  │    └──────────────────────────────────────────────┘          │  │
│  │                                                              │  │
│  │  Aggregate deltas across generations:                        │  │
│  │    ┌──────────────────────────────────────────────┐          │  │
│  │    │ Statistical:  t-test + Cohen's d             │          │  │
│  │    │ Threshold:    count(direction_match) > N/2   │          │  │
│  │    │ confirmed = stat_sig AND practical_sig       │          │  │
│  │    └──────────────────────────────────────────────┘          │  │
│  │                                                              │  │
│  │ Output: list[EditResult] with confirmed/rejected status      │  │
│  │ Models: ImageEditor, ImageNetClassifier, StatisticalValidator │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Phase 7 ──────────────▼───────────────────────────────────┐  │
│  │ FINAL ANALYSIS + CLASSIFICATION                              │  │
│  │ VLM classifies each feature: intrinsic or contextual         │  │
│  │ VLM summarizes: robustness_score, risk_level, shortcuts      │  │
│  │ Output: FinalAnalysis → applied to ClassAnalysisResult       │  │
│  │ Model:  QwenVLAnalyzer                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                   ┌──────────▼──────────┐                          │
│                   │  result.finalize()  │                          │
│                   │  Save checkpoint    │                          │
│                   └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Edit Hypothesis → Confirmation

```
  EditInstruction: "Remove the wooden background"
  hypothesis: "Model relies on background texture"
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  Original Image (conf: 0.85 for "guitar")   │
  └──────┬──────────┬──────────┬────────────────┘
         │          │          │
   seed=100    seed=101    seed=102
         │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
    │ Editor │ │ Editor │ │ Editor │
    │ FLUX.2 │ │ FLUX.2 │ │ FLUX.2 │
    └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
    │Classify│ │Classify│ │Classify│
    │ResNet50│ │ResNet50│ │ResNet50│
    └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │
    conf=0.62  conf=0.71  conf=0.64
    Δ=-0.23    Δ=-0.14    Δ=-0.21
         │          │          │
         └──────────┼──────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ StatisticalValidator.validate()         │
  │   deltas = [-0.23, -0.14, -0.21]       │
  │   mean_Δ = -0.193                       │
  │   t-test → p = 0.008 (significant)     │
  │   Cohen's d = 1.2 (large effect)       │
  │   direction = negative ✓               │
  │                                         │
  │   confirmed = TRUE                      │
  │   → SHORTCUT DETECTED                   │
  │     (contextual feature affects model)  │
  └─────────────────────────────────────────┘
```

## Module Dependency Graph

```
  main.py
    │
    ├──► Config (config.py)
    ├──► AnalysisPipeline (pipeline.py)
    │       │
    │       ├──► ModelManager (model_manager.py)
    │       │       ├──► ImageNetClassifier (classifier.py)
    │       │       │       └──► AttentionMapGenerator (models/attention_maps.py)
    │       │       │               ├── GradCAM
    │       │       │               ├── GradCAM++
    │       │       │               └── ScoreCAM
    │       │       ├──► QwenVLAnalyzer (vlm.py)
    │       │       ├──► ImageEditor (editor.py)
    │       │       └──► ImageNetSampler (dataset.py)
    │       │
    │       └──► StatisticalValidator (analysis/statistics.py)
    │
    └──► Reporter (reporter.py)
             └──► Jinja2 templates (HTML/Markdown/JSON)
```

## VRAM Management (Low-VRAM Mode)

```
  Timeline ─────────────────────────────────────────────────►

  Phase:  1-Knowledge  │ 2-Sample  │ 3-Baseline │ 4-Features │ 5-Edits        │ 6-Edits     │ 7-Final
                       │           │            │            │  (generate)     │  (apply)    │
  ─────────────────────┼───────────┼────────────┼────────────┼─────────────────┼─────────────┼──────
  VLM:    ████████████ │           │            │ ████████████│ ████████████████│             │ ██████
  Classif:             │           │ ████████████│            │                 │ █████████████│
  Editor:              │           │            │            │                 │ █████████████│
  Sampler: ████████████│████████████│            │            │                 │             │

  Legend: ████ = loaded in VRAM
  ModelManager auto-offloads competing models before each load
```

## Key Data Structures

```
  ClassAnalysisResult (one per class)
    ├── class_name: "tabby cat"
    ├── knowledge_based_features: [{feature, why_shortcut, ...}]
    ├── baseline_results: [{image_path, confidence, top_k, ...}]
    ├── detected_features: [{name, category, feature_type, gradcam_attention}]
    ├── essential_features: ["ears", "whiskers", "stripes"]
    ├── edit_results: [EditResult, ...]
    │     └── EditResult
    │           ├── instruction: "Remove the wooden background"
    │           ├── hypothesis: "Background affects classification"
    │           ├── generations: [GenerationResult, ...]
    │           │     └── {seed, edited_confidence, delta, edited_image_path}
    │           ├── mean_delta: -0.193
    │           ├── p_value: 0.008
    │           ├── cohens_d: 1.2
    │           ├── confirmed: true
    │           └── feature_type: "contextual"
    ├── confirmed_hypotheses: [EditResult, ...] (derived)
    ├── spurious_features: ["wooden background"]
    ├── robustness_score: 4/10
    ├── risk_level: "HIGH"
    └── final_summary: "Model relies on background texture..."
```

## Report Output

```
  output/
  ├── tabby_cat/
  │   ├── analysis.json              ← checkpoint (ClassAnalysisResult)
  │   ├── pos_0_original.jpg         ← original image
  │   ├── pos_0_gradcam.jpg          ← attention heatmap overlay
  │   ├── pos_0_iter0_edit_0_gen_0.jpg  ← edited variant
  │   └── neg_0_original.jpg         ← negative sample
  │
  ├── golden_retriever/
  │   └── ...
  │
  ├── report.html                    ← interactive HTML report
  │   └── Tabs: Overview | Feature Analysis | Class Details | Methodology
  ├── report.md                      ← markdown report
  └── report.json                    ← machine-readable results
```
