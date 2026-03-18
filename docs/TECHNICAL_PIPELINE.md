# CORALATION: Automated Bias/Shortcut Discovery Pipeline

## Technical Architecture Documentation

**Version:** 1.0
**Target Audience:** Technical Team
**Purpose:** Automated discovery of spurious correlations and shortcuts in image classification models

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Pipeline Phases](#3-pipeline-phases)
4. [Component Details](#4-component-details)
5. [Data Flow](#5-data-flow)
6. [Statistical Validation](#6-statistical-validation)
7. [Memory Management](#7-memory-management)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Executive Summary

### Problem Statement
Deep learning classifiers often learn **spurious correlations** (shortcuts) instead of true semantic features. For example:
- Classifying "cow" based on grass background instead of cow features
- Identifying "hospital" based on blue tint instead of medical equipment

### Solution
CORALATION automatically discovers these biases by:
1. Analyzing model attention patterns (Grad-CAM/Score-CAM)
2. Using VLM (Vision-Language Model) to identify visual features
3. Generating counterfactual edits to test feature importance
4. Statistically validating which features are true shortcuts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORALATION PIPELINE OVERVIEW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ImageNet    ──►  Classifier   ──►   VLM        ──►  Image     ──►  Report │
│   Dataset          (ResNet-50)       (Qwen-VL)        Editor                │
│                         │                │               │                  │
│                         ▼                ▼               ▼                  │
│                    Attention        Feature         Counterfactual          │
│                    Maps             Discovery       Images                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. High-Level Architecture

### System Components

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │   DATA LAYER    │    │  MODEL LAYER    │    │  ANALYSIS LAYER │              │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤              │
│  │                 │    │                 │    │                 │              │
│  │ ImageNet-1K     │───►│ ResNet-50       │───►│ Statistical     │              │
│  │ (HuggingFace)   │    │ Classifier      │    │ Validator       │              │
│  │                 │    │                 │    │                 │              │
│  │ Sampler         │    │ Qwen2.5-VL-7B   │    │ Hypothesis      │              │
│  │ - Positive      │    │ VLM Analyzer    │    │ Tester          │              │
│  │ - Negative      │    │                 │    │                 │              │
│  │                 │    │ Qwen-Image-Edit │    │ Result          │              │
│  │                 │    │ Editor          │    │ Aggregator      │              │
│  │                 │    │                 │    │                 │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
│           │                     │                      │                        │
│           └─────────────────────┴──────────────────────┘                        │
│                                 │                                               │
│                                 ▼                                               │
│                    ┌─────────────────────────┐                                  │
│                    │     OUTPUT LAYER        │                                  │
│                    ├─────────────────────────┤                                  │
│                    │ • HTML Report           │                                  │
│                    │ • Markdown Report       │                                  │
│                    │ • JSON Data             │                                  │
│                    │ • Edited Images         │                                  │
│                    └─────────────────────────┘                                  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Model | Size | VRAM | Purpose |
|-----------|-------|------|------|---------|
| Classifier | ResNet-50 | 25M params | ~2GB | Target model being analyzed |
| VLM | Qwen2.5-VL-7B-Instruct | 7B params | ~14GB | Feature discovery & verification |
| Editor | Qwen-Image-Edit | 20B params | ~8GB (FP8) | Counterfactual image generation |

---

## 3. Pipeline Phases

### Complete Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE PIPELINE FLOW                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ PHASE 1: DATA COLLECTION                                                  ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  ImageNet-1K ──► Sample Positive ──► VLM Select ──► Sample Negative      ║  │
│  ║                  (target class)      Confusing       (similar classes)    ║  │
│  ║                                      Classes                              ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                    │                                             │
│                                    ▼                                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ PHASE 2: BASELINE ANALYSIS                                                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  Images ──► ResNet-50 ──► Predictions ──► Score-CAM ──► Attention Maps   ║  │
│  ║             Forward       + Confidence     Analysis      (Heatmaps)       ║  │
│  ║             Pass                                                          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                    │                                             │
│                                    ▼                                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ PHASE 3: FEATURE DISCOVERY (VLM)                                          ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  Image + ──► Qwen2.5-VL ──► Detected    ──► Feature      ──► Edit        ║  │
│  ║  Attention   Analysis       Features        Classification   Instructions ║  │
│  ║  Map                        (8-12)          (intrinsic/                   ║  │
│  ║                                              contextual)                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                    │                                             │
│                                    ▼                                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ PHASE 4: COUNTERFACTUAL EDITING                                           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  For each edit instruction:                                               ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │                                                                     │  ║  │
│  ║  │  Original ──► Qwen-Image-Edit ──► Edited ──► VLM Verify ──► Retry? │  ║  │
│  ║  │  Image        (50 steps)          Image      (success?)     (2x)   │  ║  │
│  ║  │                                      │                              │  ║  │
│  ║  │                                      ▼                              │  ║  │
│  ║  │                              ResNet-50 ──► New Confidence           │  ║  │
│  ║  │                              Inference     + Delta                  │  ║  │
│  ║  │                                                                     │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                    │                                             │
│                                    ▼                                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ PHASE 5: STATISTICAL VALIDATION                                           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  Multiple    ──► Calculate  ──► t-test      ──► Cohen's d  ──► Confirm   ║  │
│  ║  Generations     Deltas         (p < 0.05)      (d > 0.5)      Shortcut  ║  │
│  ║  (2-5 per edit)                                                          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                    │                                             │
│                                    ▼                                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ PHASE 6: FINAL ANALYSIS & REPORTING                                       ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                           ║  │
│  ║  VLM Semantic  ──► Classify      ──► Risk        ──► Generate Reports    ║  │
│  ║  Analysis          Shortcuts         Assessment      (HTML/MD/JSON)      ║  │
│  ║                    vs Essential      (1-10 score)                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Data Collection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: DATA COLLECTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Target class name (e.g., "tabby cat")                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1.1: Sample Positive Images                                    │   │
│  │                                                                     │   │
│  │   ImageNet-1K ─────► Filter by ─────► Random ─────► N positive     │   │
│  │   Validation        class label      sample        images          │   │
│  │   (50K images)                       (shuffle)     (default: 5)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1.2: VLM Selects Confusing Classes                             │   │
│  │                                                                     │   │
│  │   VLM Prompt: "Which classes are visually similar to {class}?"     │   │
│  │                                                                     │   │
│  │   Example for "tabby cat":                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ Selected: Egyptian cat, tiger cat, Persian cat, lynx       │   │   │
│  │   │ Reasoning: Similar fur patterns, ear shapes, body structure│   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1.3: Sample Negative Images                                    │   │
│  │                                                                     │   │
│  │   From confusing ─────► 1 image ─────► N negative images           │   │
│  │   classes               per class      (default: 5)                 │   │
│  │                                                                     │   │
│  │   Purpose: Test what features cause false positives                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: positive_images[], negative_images[], confusing_classes[]         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 2: Baseline Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: BASELINE ANALYSIS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: positive_images[], negative_images[]                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2.1: Classification Inference                                  │   │
│  │                                                                     │   │
│  │   Image ──► Preprocess ──► ResNet-50 ──► Softmax ──► Top-K Preds   │   │
│  │             (224x224)      Forward       Probabilities              │   │
│  │             Normalize      Pass                                     │   │
│  │                                                                     │   │
│  │   Output per image:                                                 │   │
│  │   {                                                                 │   │
│  │     "predicted_label": "tabby cat",                                 │   │
│  │     "confidence": 0.847,                                            │   │
│  │     "class_confidence": 0.847,  // for target class                 │   │
│  │     "top_k": [("tabby", 0.84), ("tiger_cat", 0.12), ...]           │   │
│  │   }                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2.2: Attention Map Generation (Score-CAM)                      │   │
│  │                                                                     │   │
│  │   ┌───────────────────────────────────────────────────────────┐     │   │
│  │   │              SCORE-CAM ALGORITHM                          │     │   │
│  │   ├───────────────────────────────────────────────────────────┤     │   │
│  │   │                                                           │     │   │
│  │   │  1. Extract feature maps from layer4 (2048 channels)     │     │   │
│  │   │                                                           │     │   │
│  │   │  2. For each channel k:                                   │     │   │
│  │   │     a. Upsample activation map to input size             │     │   │
│  │   │     b. Normalize to [0,1]                                │     │   │
│  │   │     c. Mask input: X_masked = X * A_k                    │     │   │
│  │   │     d. Forward pass: score_k = f(X_masked)[target_class] │     │   │
│  │   │                                                           │     │   │
│  │   │  3. Weight maps by scores:                                │     │   │
│  │   │     CAM = ReLU(Σ score_k * A_k)                          │     │   │
│  │   │                                                           │     │   │
│  │   │  4. Normalize and overlay on original image              │     │   │
│  │   │                                                           │     │   │
│  │   └───────────────────────────────────────────────────────────┘     │   │
│  │                                                                     │   │
│  │   Visualization:                                                    │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │   │   Original   │ +  │   Heatmap    │ =  │   Overlay    │         │   │
│  │   │    Image     │    │  (jet cmap)  │    │   Result     │         │   │
│  │   │              │    │  Red = High  │    │              │         │   │
│  │   │   [CAT]      │    │  Blue = Low  │    │  [CAT+HEAT]  │         │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: baseline_results[] with predictions and attention maps            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Attention Methods Comparison:**

| Method | Speed | Stability | Gradient-Free | Recommended |
|--------|-------|-----------|---------------|-------------|
| Grad-CAM | Fast | Medium | No | Development |
| Grad-CAM++ | Fast | Medium | No | Quick tests |
| Score-CAM | Slow | High | Yes | **Production** |

---

### Phase 3: Feature Discovery

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: FEATURE DISCOVERY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Original image + Score-CAM heatmap                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 3.1: VLM Feature Detection                                     │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                   VLM INPUT                                 │   │   │
│  │   ├─────────────────────────────────────────────────────────────┤   │   │
│  │   │  Image 1: Original photograph                              │   │   │
│  │   │  Image 2: Score-CAM heatmap overlay                        │   │   │
│  │   │  Prompt: "Identify ALL visual features..."                 │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                   VLM OUTPUT                                │   │   │
│  │   ├─────────────────────────────────────────────────────────────┤   │   │
│  │   │  {                                                          │   │   │
│  │   │    "detected_features": [                                   │   │   │
│  │   │      {                                                      │   │   │
│  │   │        "name": "pointed ears",                              │   │   │
│  │   │        "category": "object_part",                           │   │   │
│  │   │        "feature_type": "intrinsic",                         │   │   │
│  │   │        "location": "top center",                            │   │   │
│  │   │        "gradcam_attention": "high"                          │   │   │
│  │   │      },                                                     │   │   │
│  │   │      {                                                      │   │   │
│  │   │        "name": "wooden floor",                              │   │   │
│  │   │        "category": "context",                               │   │   │
│  │   │        "feature_type": "contextual",                        │   │   │
│  │   │        "location": "bottom",                                │   │   │
│  │   │        "gradcam_attention": "medium"                        │   │   │
│  │   │      }                                                      │   │   │
│  │   │    ],                                                       │   │   │
│  │   │    "intrinsic_features": ["ears", "eyes", "whiskers"...],  │   │   │
│  │   │    "contextual_features": ["wooden floor", "sunlight"...]  │   │   │
│  │   │  }                                                          │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 3.2: Feature Classification                                    │   │
│  │                                                                     │   │
│  │   ┌─────────────────────┐         ┌─────────────────────┐          │   │
│  │   │  INTRINSIC FEATURES │         │ CONTEXTUAL FEATURES │          │   │
│  │   │  (Part of object)   │         │ (Background/Context)│          │   │
│  │   ├─────────────────────┤         ├─────────────────────┤          │   │
│  │   │ • Ears              │         │ • Wooden floor      │          │   │
│  │   │ • Eyes              │         │ • Window light      │          │   │
│  │   │ • Whiskers          │         │ • Furniture         │          │   │
│  │   │ • Fur pattern       │         │ • Indoor setting    │          │   │
│  │   │ • Body shape        │         │ • Color cast        │          │   │
│  │   └─────────────────────┘         └─────────────────────┘          │   │
│  │            │                               │                        │   │
│  │            ▼                               ▼                        │   │
│  │   If removing DECREASES         If removing DECREASES              │   │
│  │   confidence: EXPECTED          confidence: SHORTCUT!              │   │
│  │   (model is correct)            (model is biased)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 3.3: Generate Edit Instructions                                │   │
│  │                                                                     │   │
│  │   For each detected feature, generate specific edit:                │   │
│  │                                                                     │   │
│  │   Feature: "pointed ears"                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ {                                                           │   │   │
│  │   │   "edit_instruction": "Remove the pointed ears from cat",  │   │   │
│  │   │   "edit_type": "removal",                                   │   │   │
│  │   │   "expected_impact": "high",                                │   │   │
│  │   │   "hypothesis": "Ears are intrinsic; removal should        │   │   │
│  │   │                  decrease confidence significantly"         │   │   │
│  │   │ }                                                           │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │   Typically generates 10-15 edit instructions per image            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: edit_instructions[] with hypothesis for each feature              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 4: Counterfactual Editing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: COUNTERFACTUAL EDITING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Original image + Edit instruction                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EDIT RETRY FLOW                                  │   │
│  │                                                                     │   │
│  │   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     │   │
│  │   │ Original│     │ Qwen    │     │ Edited  │     │   VLM   │     │   │
│  │   │  Image  │────►│ Image   │────►│  Image  │────►│ Verify  │     │   │
│  │   │         │     │  Edit   │     │         │     │         │     │   │
│  │   └─────────┘     └─────────┘     └─────────┘     └────┬────┘     │   │
│  │                                                        │          │   │
│  │                        ┌───────────────────────────────┘          │   │
│  │                        │                                          │   │
│  │                        ▼                                          │   │
│  │              ┌─────────────────────┐                              │   │
│  │              │  Verification OK?   │                              │   │
│  │              └─────────┬───────────┘                              │   │
│  │                        │                                          │   │
│  │           ┌────────────┴────────────┐                             │   │
│  │           │                         │                             │   │
│  │           ▼                         ▼                             │   │
│  │    ┌─────────────┐          ┌─────────────┐                       │   │
│  │    │     YES     │          │      NO     │                       │   │
│  │    │  Continue   │          │  Retry?     │                       │   │
│  │    └──────┬──────┘          └──────┬──────┘                       │   │
│  │           │                        │                              │   │
│  │           │               ┌────────┴────────┐                     │   │
│  │           │               │                 │                     │   │
│  │           │               ▼                 ▼                     │   │
│  │           │        ┌───────────┐     ┌───────────┐                │   │
│  │           │        │ Retries   │     │ Retries   │                │   │
│  │           │        │ < Max (2) │     │ Exhausted │                │   │
│  │           │        └─────┬─────┘     └─────┬─────┘                │   │
│  │           │              │                 │                      │   │
│  │           │              ▼                 ▼                      │   │
│  │           │        ┌───────────┐     ┌───────────┐                │   │
│  │           │        │  Refine   │     │ Use Last  │                │   │
│  │           │        │  Prompt   │     │  Image    │                │   │
│  │           │        │ & Retry   │     │  Anyway   │                │   │
│  │           │        └─────┬─────┘     └─────┬─────┘                │   │
│  │           │              │                 │                      │   │
│  │           │              └────────┬────────┘                      │   │
│  │           │                       │                               │   │
│  │           └───────────────────────┼───────────────────────────────│   │
│  │                                   │                               │   │
│  │                                   ▼                               │   │
│  │                    ┌──────────────────────────┐                   │   │
│  │                    │   Measure Confidence     │                   │   │
│  │                    │   on Edited Image        │                   │   │
│  │                    │                          │                   │   │
│  │                    │   delta = new - original │                   │   │
│  │                    └──────────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PROMPT REFINEMENT STRATEGY                       │   │
│  │                                                                     │   │
│  │   Original: "Remove the pointed ears from the cat"                  │   │
│  │                                                                     │   │
│  │   Retry 1:  "COMPLETELY Remove the pointed ears from the cat.      │   │
│  │              Make sure it is fully removed and not visible at all. │   │
│  │              The ears are still visible."                           │   │
│  │                                                                     │   │
│  │   Retry 2:  "Replace the pointed ears with smooth, plain surface   │   │
│  │              that matches the surrounding area. Erase all traces." │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    QWEN-IMAGE-EDIT PARAMETERS                       │   │
│  │                                                                     │   │
│  │   Model: Qwen/Qwen-Image-Edit (20B parameters)                     │   │
│  │   Inference steps: 50                                               │   │
│  │   CFG scale: 4.0                                                    │   │
│  │   Max resolution: 768x768                                           │   │
│  │   Quantization: FP8 with group offloading                          │   │
│  │   Memory usage: ~8GB VRAM                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: edited_image, confidence_delta, verification_status               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 5: Statistical Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 5: STATISTICAL VALIDATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Multiple generations per edit (default: 3)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Why Multiple Generations?                                           │   │
│  │                                                                     │   │
│  │   Diffusion models are stochastic - same prompt produces           │   │
│  │   different outputs with different seeds.                          │   │
│  │                                                                     │   │
│  │   Example for "Remove ears" with 3 seeds:                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ Seed 42:  Original: 0.85 → Edited: 0.42 → Delta: -0.43     │   │   │
│  │   │ Seed 52:  Original: 0.85 → Edited: 0.38 → Delta: -0.47     │   │   │
│  │   │ Seed 62:  Original: 0.85 → Edited: 0.51 → Delta: -0.34     │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Statistical Tests                                                   │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 1. ONE-SAMPLE T-TEST                                        │   │   │
│  │   │                                                             │   │   │
│  │   │    H₀: mean(delta) = 0  (no effect)                        │   │   │
│  │   │    H₁: mean(delta) ≠ 0  (significant effect)               │   │   │
│  │   │                                                             │   │   │
│  │   │    For directional hypothesis (feature removal):            │   │   │
│  │   │    H₁: mean(delta) < 0  (confidence decreased)             │   │   │
│  │   │                                                             │   │   │
│  │   │    Statistically significant if: p-value < 0.05            │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 2. COHEN'S D (Effect Size)                                  │   │   │
│  │   │                                                             │   │   │
│  │   │    d = mean(delta) / std(delta)                            │   │   │
│  │   │                                                             │   │   │
│  │   │    Interpretation:                                          │   │   │
│  │   │    ┌──────────────────────────────────────────────────┐     │   │   │
│  │   │    │  |d| < 0.2   : Negligible effect                 │     │   │   │
│  │   │    │  |d| 0.2-0.5 : Small effect                      │     │   │   │
│  │   │    │  |d| 0.5-0.8 : Medium effect                     │     │   │   │
│  │   │    │  |d| > 0.8   : Large effect                      │     │   │   │
│  │   │    └──────────────────────────────────────────────────┘     │   │   │
│  │   │                                                             │   │   │
│  │   │    Practically significant if: |d| >= 0.5                  │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Confirmation Logic                                                  │   │
│  │                                                                     │   │
│  │   confirmed = (p_value < 0.05) AND (|cohen_d| >= 0.5)              │   │
│  │                     │                      │                        │   │
│  │                     │                      │                        │   │
│  │          Statistically           Practically                        │   │
│  │          Significant             Significant                        │   │
│  │                                                                     │   │
│  │   Direction Validation:                                             │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ Edit Type        │ Target   │ Expected Delta │ Valid If     │   │   │
│  │   ├───────────────────┼──────────┼────────────────┼──────────────┤   │   │
│  │   │ feature_removal  │ positive │ negative       │ delta < -0.15│   │   │
│  │   │ feature_addition │ positive │ positive       │ delta > +0.15│   │   │
│  │   │ feature_addition │ negative │ positive       │ delta > +0.15│   │   │
│  │   │ context_change   │ any      │ any            │ |delta| > 0.15│  │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: EditResult with p_value, cohens_d, confirmed status               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 6: Final Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 6: FINAL ANALYSIS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: All edit results with deltas and confirmation status                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 6.1: VLM Semantic Classification                               │   │
│  │                                                                     │   │
│  │   VLM reviews all results and classifies each tested feature:      │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ Feature        │ Delta  │ Type       │ Classification       │   │   │
│  │   ├────────────────┼────────┼────────────┼──────────────────────┤   │   │
│  │   │ Pointed ears   │ -0.41  │ intrinsic  │ ESSENTIAL (correct) │   │   │
│  │   │ Whiskers       │ -0.38  │ intrinsic  │ ESSENTIAL (correct) │   │   │
│  │   │ Wooden floor   │ -0.22  │ contextual │ SHORTCUT (bias!)    │   │   │
│  │   │ Indoor setting │ -0.18  │ contextual │ SHORTCUT (bias!)    │   │   │
│  │   │ Fur pattern    │ -0.35  │ intrinsic  │ ESSENTIAL (correct) │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 6.2: Risk Assessment                                           │   │
│  │                                                                     │   │
│  │   Robustness Score (1-10):                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  10: No shortcuts found                                     │   │   │
│  │   │  7-9: Minor contextual dependencies                         │   │   │
│  │   │  4-6: Some significant shortcuts                            │   │   │
│  │   │  1-3: Major bias issues                                     │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │   Risk Level:                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  LOW:    0-1 shortcuts, robustness >= 7                     │   │   │
│  │   │  MEDIUM: 2-3 shortcuts, robustness 4-6                      │   │   │
│  │   │  HIGH:   4+ shortcuts, robustness < 4                       │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 6.3: Report Generation                                         │   │
│  │                                                                     │   │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │   │
│  │   │   HTML Report   │  │ Markdown Report │  │   JSON Data     │    │   │
│  │   ├─────────────────┤  ├─────────────────┤  ├─────────────────┤    │   │
│  │   │ • Visual layout │  │ • Text-based    │  │ • Machine-      │    │   │
│  │   │ • Image gallery │  │ • Git-friendly  │  │   readable      │    │   │
│  │   │ • Interactive   │  │ • Documentation │  │ • API-ready     │    │   │
│  │   │ • Charts        │  │                 │  │ • Full data     │    │   │
│  │   └─────────────────┘  └─────────────────┘  └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: Complete analysis with risk assessment and recommendations        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Details

### 4.1 ImageNet Sampler (`src/dataset.py`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMAGENET SAMPLER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Data Source: HuggingFace datasets (ILSVRC/imagenet-1k)                    │
│  Split: validation (50,000 images)                                          │
│  Classes: 1,000 ImageNet categories                                         │
│                                                                             │
│  Methods:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ sample_positive(class_name, n)                                      │   │
│  │   └─► Returns n images matching the class label                     │   │
│  │                                                                     │   │
│  │ sample_from_classes(class_names, n_per_class)                       │   │
│  │   └─► Returns images from multiple classes (for negative samples)  │   │
│  │                                                                     │   │
│  │ get_label_names()                                                   │   │
│  │   └─► Returns all 1,000 class names                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Sampling Strategy:                                                         │
│  • Shuffles dataset with configurable seed                                  │
│  • Scans up to max_scan (10,000) images to find matches                    │
│  • Caches results to avoid re-scanning                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Classifier (`src/classifier.py`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMAGENET CLASSIFIER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model: ResNet-50 (pretrained on ImageNet-1K)                              │
│  Input: 224x224 RGB image (normalized)                                      │
│  Output: 1,000-class probability distribution                               │
│                                                                             │
│  Preprocessing Pipeline:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PIL Image ──► Resize(256) ──► CenterCrop(224) ──► ToTensor ──►    │   │
│  │               Normalize(mean=[0.485,0.456,0.406],                   │   │
│  │                         std=[0.229,0.224,0.225])                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Attention Map Generation:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Target Layer: model.layer4 (final conv block)                      │   │
│  │  Output Size: 7x7 feature maps (2048 channels)                      │   │
│  │  Upsampled to: Original image size                                  │   │
│  │  Colormap: Jet (blue → green → yellow → red)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 VLM Analyzer (`src/vlm.py`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VLM ANALYZER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model: Qwen2.5-VL-7B-Instruct                                             │
│  Context: 32K tokens (supports multiple images)                             │
│  Dtype: bfloat16                                                            │
│                                                                             │
│  Capabilities:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. select_confusing_classes(target, all_classes, n)                 │   │
│  │    └─► Identifies visually similar classes for negative sampling   │   │
│  │                                                                     │   │
│  │ 2. discover_features(image, attention_map, class_name)              │   │
│  │    └─► Identifies 8-12 visual features with classification         │   │
│  │                                                                     │   │
│  │ 3. generate_feature_edits(image, features, class_name)              │   │
│  │    └─► Creates specific edit instructions for each feature          │   │
│  │                                                                     │   │
│  │ 4. verify_edit(original, edited, instruction)                       │   │
│  │    └─► Validates if edit was correctly applied                      │   │
│  │                                                                     │   │
│  │ 5. final_analysis(image, class_name, results)                       │   │
│  │    └─► Semantic classification of shortcuts vs essential features  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Prompt Engineering:                                                        │
│  • Structured JSON output format                                            │
│  • Chain-of-thought reasoning                                               │
│  • JSON repair for malformed responses                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Image Editor (`src/editor.py`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMAGE EDITOR                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model: Qwen/Qwen-Image-Edit                                               │
│  Architecture: Diffusion Transformer (DiT)                                  │
│  Parameters: ~20B                                                           │
│                                                                             │
│  Memory Optimization:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ FP8 Quantization:                                                   │   │
│  │   • Storage: float8_e4m3fn (8-bit)                                  │   │
│  │   • Compute: bfloat16 (16-bit)                                      │   │
│  │   • Savings: ~50% VRAM reduction                                    │   │
│  │                                                                     │   │
│  │ Group Offloading:                                                   │   │
│  │   • Transformer: leaf-level offload with streaming                  │   │
│  │   • Text Encoder: block-level offload (2 blocks/group)             │   │
│  │   • VAE: leaf-level offload                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Edit Pipeline:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input Image ──► Resize(768) ──► Encode ──► Denoise ──► Decode ──► │   │
│  │                  (preserve AR)    (VAE)     (50 steps)   (VAE)      │   │
│  │                                                  │                  │   │
│  │                                                  ▼                  │   │
│  │                                          CFG Scale: 4.0             │   │
│  │                                          + Text Prompt              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow

### Data Structures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA STRUCTURES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DetectedFeature                                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ {                                                                   │   │
│  │   name: str              # "pointed ears"                          │   │
│  │   category: str          # "object_part" | "texture" | "context"   │   │
│  │   feature_type: str      # "intrinsic" | "contextual"              │   │
│  │   location: str          # "top center"                            │   │
│  │   gradcam_attention: str # "high" | "medium" | "low"               │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EditInstruction                                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ {                                                                   │   │
│  │   edit: str              # "Remove the pointed ears from the cat"  │   │
│  │   hypothesis: str        # "Ears are intrinsic feature..."         │   │
│  │   type: str              # "feature_removal" | "feature_addition"  │   │
│  │   target: str            # "positive" | "negative"                 │   │
│  │   priority: int          # 1-5 (importance)                        │   │
│  │   image_index: int       # which image to edit                     │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GenerationResult                                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ {                                                                   │   │
│  │   seed: int              # random seed used                         │   │
│  │   edited_confidence: float                                          │   │
│  │   delta: float           # edited - original                        │   │
│  │   edited_image_path: str                                            │   │
│  │   edit_verified: bool    # VLM verification passed                  │   │
│  │   verification_confidence: float                                    │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EditResult                                                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ {                                                                   │   │
│  │   instruction: str                                                  │   │
│  │   hypothesis: str                                                   │   │
│  │   edit_type: str                                                    │   │
│  │   target_type: str       # "positive" | "negative"                 │   │
│  │   original_confidence: float                                        │   │
│  │   generations: List[GenerationResult]  # multiple seeds            │   │
│  │   mean_delta: float                                                 │   │
│  │   std_delta: float                                                  │   │
│  │   p_value: float         # from t-test                              │   │
│  │   cohens_d: float        # effect size                              │   │
│  │   confirmed: bool        # statistically + practically significant │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ClassAnalysisResult                                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ {                                                                   │   │
│  │   class_name: str                                                   │   │
│  │   detected_features: List[dict]                                     │   │
│  │   essential_features: List[str]                                     │   │
│  │   spurious_features: List[str]     # confirmed shortcuts           │   │
│  │   baseline_results: List[dict]                                      │   │
│  │   edit_results: List[EditResult]                                    │   │
│  │   confirmed_hypotheses: List[EditResult]                            │   │
│  │   robustness_score: int            # 1-10                           │   │
│  │   risk_level: str                  # "LOW" | "MEDIUM" | "HIGH"     │   │
│  │   vulnerabilities: List[str]                                        │   │
│  │   recommendations: List[str]                                        │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Statistical Validation

### Validation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STATISTICAL VALIDATION DETAIL                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Example: Testing "Remove pointed ears" edit                                │
│                                                                             │
│  Generation Results:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Gen 1 (seed=42): 0.847 → 0.421 = -0.426                           │   │
│  │  Gen 2 (seed=52): 0.847 → 0.385 = -0.462                           │   │
│  │  Gen 3 (seed=62): 0.847 → 0.512 = -0.335                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Step 1: Calculate Statistics                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  deltas = [-0.426, -0.462, -0.335]                                  │   │
│  │  mean_delta = -0.408                                                │   │
│  │  std_delta = 0.065                                                  │   │
│  │  n = 3                                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Step 2: One-Sample t-test                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  H₀: μ = 0 (no effect)                                              │   │
│  │  H₁: μ < 0 (confidence decreased)                                   │   │
│  │                                                                     │   │
│  │  t = mean_delta / (std_delta / √n)                                  │   │
│  │  t = -0.408 / (0.065 / √3) = -10.87                                │   │
│  │                                                                     │   │
│  │  p-value = 0.0042 (one-tailed)                                      │   │
│  │  p < 0.05 ✓ Statistically significant                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Step 3: Cohen's d (Effect Size)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  d = mean_delta / std_delta                                         │   │
│  │  d = -0.408 / 0.065 = -6.28                                         │   │
│  │                                                                     │   │
│  │  |d| = 6.28 > 0.8 → "large" effect                                 │   │
│  │  |d| > 0.5 ✓ Practically significant                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Step 4: Direction Validation                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Edit type: feature_removal                                         │   │
│  │  Target: positive sample                                            │   │
│  │  Expected direction: negative (confidence should decrease)          │   │
│  │  Actual mean_delta: -0.408                                          │   │
│  │  Direction correct ✓                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Final Result:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  confirmed = True                                                   │   │
│  │  (p < 0.05) AND (|d| >= 0.5) AND (direction correct)               │   │
│  │                                                                     │   │
│  │  Interpretation: Removing ears significantly decreases confidence  │   │
│  │  This is EXPECTED behavior (ears are intrinsic to "cat")           │   │
│  │  → NOT a shortcut, this is a legitimate feature                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Memory Management

### VRAM Optimization Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGEMENT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Total Available: 24GB VRAM (typical)                                       │
│                                                                             │
│  Model VRAM Requirements:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ResNet-50:         ~2GB                                            │   │
│  │  Qwen2.5-VL-7B:     ~14GB (bfloat16)                               │   │
│  │  Qwen-Image-Edit:   ~8GB (FP8 + offload)                           │   │
│  │  ─────────────────────────────────                                  │   │
│  │  Total if loaded:   ~24GB (exactly at limit!)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Low-VRAM Mode (Sequential Loading):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Phase 1-2: Load Classifier (2GB)                                   │   │
│  │       │     ─────► Baseline inference + Grad-CAM                    │   │
│  │       ▼                                                             │   │
│  │  Offload Classifier                                                 │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Phase 3: Load VLM (14GB)                                           │   │
│  │       │   ─────► Feature discovery + Edit generation               │   │
│  │       ▼                                                             │   │
│  │  Offload VLM                                                        │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Phase 4: Load Editor (8GB) + Classifier (2GB) = 10GB              │   │
│  │       │   ─────► Edit + Measure confidence                          │   │
│  │       │                                                             │   │
│  │       ├─── Need verification? ───┐                                  │   │
│  │       │                          ▼                                  │   │
│  │       │                   Offload Editor                            │   │
│  │       │                   Load VLM (14GB)                           │   │
│  │       │                   Verify edit                               │   │
│  │       │                   Offload VLM                               │   │
│  │       │                   Load Editor (8GB)                         │   │
│  │       │                          │                                  │   │
│  │       ◄──────────────────────────┘                                  │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  Phase 6: Load VLM (14GB)                                           │   │
│  │       │   ─────► Final analysis                                     │   │
│  │       ▼                                                             │   │
│  │  Offload all, generate reports                                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FP8 Quantization for Image Editor:                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  # Transformer: FP8 storage, BF16 compute                          │   │
│  │  pipe.transformer.enable_layerwise_casting(                        │   │
│  │      storage_dtype=torch.float8_e4m3fn,                            │   │
│  │      compute_dtype=torch.bfloat16                                  │   │
│  │  )                                                                  │   │
│  │                                                                     │   │
│  │  # Group offloading: move layers CPU↔GPU as needed                 │   │
│  │  pipe.transformer.enable_group_offload(                            │   │
│  │      onload_device="cuda",                                         │   │
│  │      offload_device="cpu",                                         │   │
│  │      offload_type="leaf_level",                                    │   │
│  │      use_stream=True  # async transfer                             │   │
│  │  )                                                                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration Reference

### Full Configuration Options

```python
class Config:
    # --- Model settings ---
    classifier_model: str = "resnet50"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    editor_model: str = "Qwen/Qwen-Image-Edit"

    # --- Dataset settings ---
    hf_dataset: str = "ILSVRC/imagenet-1k"
    samples_per_class: int = 5          # positive examples
    negative_samples: int = 5           # negative examples
    max_scan: int = 10000               # max images to scan
    random_seed: int | None = None      # reproducibility

    # --- Analysis settings ---
    confidence_delta_threshold: float = 0.15  # min delta for confirmation
    max_hypotheses_per_image: int = 5         # edits per image
    generations_per_edit: int = 3             # seeds per edit
    iterations: int = 2                       # VLM refinement iterations

    # --- Attention map ---
    attention_method: str = "scorecam"  # gradcam | gradcam++ | scorecam

    # --- Statistical validation ---
    statistical_alpha: float = 0.05     # significance level
    min_effect_size: float = 0.5        # minimum Cohen's d

    # --- Verification & retry ---
    verify_edits: bool = True           # VLM verification
    edit_retry_attempts: int = 2        # retries on failed verification

    # --- Memory management ---
    low_vram: bool = True               # sequential model loading
    use_8bit_editor: bool = True        # FP8 quantization

    # --- Output ---
    output_dir: Path = Path("output")
    resume: bool = True                 # checkpoint support
```

### CLI Usage

```bash
# Default run (5 classes)
uv run python main.py

# Quick test with debug logging
uv run python main.py --classes 1 --samples 1 --iterations 1 \
    --debug --log-file debug.log

# Full analysis
uv run python main.py --classes 10 --samples 5 --iterations 2 \
    --generations 3 --attention scorecam

# High-VRAM mode (40GB+)
uv run python main.py --high-vram --classes 10
```

---

## Summary

### Key Innovation Points

1. **Automated Shortcut Discovery**: No manual labeling required
2. **VLM-Guided Analysis**: Semantic understanding of features
3. **Statistical Rigor**: t-tests + effect size for validation
4. **Counterfactual Testing**: Direct causal inference via editing
5. **Memory Efficient**: FP8 + offloading enables 24GB operation

### Limitations

1. Edit quality depends on diffusion model capabilities
2. VLM may misclassify some features
3. Statistical power limited with few generations
4. Score-CAM is slow (but most accurate)

### Future Improvements

1. Support for other classifier architectures
2. Batch processing for efficiency
3. Interactive web interface
4. Integration with MLOps pipelines
