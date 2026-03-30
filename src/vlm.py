"""
Vision-Language Model (VLM) - Feature discovery and hypothesis generation.

This module wraps Qwen2.5-VL for analyzing images and generating edit hypotheses.

Key Capabilities:
    1. Feature Discovery: Identify visual features in images (intrinsic vs contextual)
    2. Attention Analysis: Interpret Score-CAM/Grad-CAM attention maps
    3. Hypothesis Generation: Create testable edit instructions
    4. Semantic Verification: Validate which features are true shortcuts

Key Classes:
    - QwenVLAnalyzer: Main VLM interface
    - DetectedFeature: Single identified feature with metadata
    - FeatureDiscovery: Complete feature analysis for an image
    - EditInstruction: Actionable edit with hypothesis and priority

Feature Types:
    - intrinsic: Part of the object (ears, eyes, fur) - expected to affect classification
    - contextual: Background/environment - if affects classification, it's a shortcut

Usage:
    vlm = QwenVLAnalyzer(model_name, device)
    discovery = vlm.discover_features(image, attention_map, class_name)
    edits = vlm.generate_edit_plan(discovery, negative_samples)
"""
from __future__ import annotations
import base64
import contextlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr and tqdm during model loading."""
    import tqdm

    # Save original
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_tqdm_disable = getattr(tqdm.tqdm, '__init__', None)

    # Disable tqdm globally
    original_tqdm_init = tqdm.tqdm.__init__
    def silent_tqdm_init(self, *args, **kwargs):
        kwargs['disable'] = True
        return original_tqdm_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = silent_tqdm_init

    # Redirect to devnull
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            tqdm.tqdm.__init__ = original_tqdm_init

# =============================================================================
# PHASE 1: Feature Discovery from Grad-CAM
# =============================================================================

_FEATURE_DISCOVERY_PROMPT = """\
You are an expert in computer vision and model interpretability.

I am analyzing an image classifier.
Target class: **{class_name}**
Confidence: {confidence:.1%}

I will show you:
1. The original image
2. A Grad-CAM heatmap (red = high model attention, blue = low)

## Your Task: Identify ALL Visual Features

Analyze the image and Grad-CAM carefully. List EVERY visual feature that could influence classification.

### Categories to Consider:

**A) Object Parts** (structural features that DEFINE what a "{class_name}" is):
   - Parts, components, or anatomy specific to this class
   - Distinctive shapes or structures

**B) Texture & Patterns**:
   - Surface textures, material appearance
   - Patterns (stripes, spots, gradients, etc.)

**C) Colors**:
   - Dominant colors, color combinations
   - Specific hues characteristic of this class

**D) Shape & Silhouette**:
   - Overall outline, proportions
   - Distinctive contours

**E) Context & Background** (PAY SPECIAL ATTENTION to these):
   - Environment (indoor/outdoor, natural/urban)
   - Co-occurring objects
   - Typical settings where this class appears
   - Look for environmental elements that repeat across training images
   - These are the most likely sources of dataset bias

**F) Grad-CAM Analysis**:
   - What areas have HIGH attention (red)? Why?
   - What areas have LOW attention (blue)?
   - Is the model focusing on the RIGHT features?

**IMPORTANT DISTINCTION:**
- "intrinsic" = Part of the "{class_name}" itself (defining parts, inherent textures, characteristic colors)
- "contextual" = NOT part of the object (background, environment, co-occurring objects, lighting)

We need to test ALL features to see if the model relies on them.
After experiments, we'll determine which are essential vs spurious based on:
- If removing an INTRINSIC feature decreases confidence → that's EXPECTED (model is correct)
- If removing a CONTEXTUAL feature decreases confidence → that's a SHORTCUT/BIAS (model is wrong)

Respond with JSON:
{{
  "detected_features": [
    {{
      "name": "feature name",
      "category": "object_part | texture | color | shape | context",
      "feature_type": "intrinsic | contextual",
      "location": "where in image (e.g., 'top center')",
      "gradcam_attention": "high | medium | low",
      "reasoning": "brief description of this feature"
    }}
  ],
  "gradcam_summary": "what the heatmap reveals about model focus",
  "intrinsic_features": ["features that are part of the {class_name} itself"],
  "contextual_features": ["background, environment, co-occurring objects"]
}}

List 6-8 key features. Include both intrinsic (object parts) AND contextual (background) features.
Keep descriptions brief (under 20 words each).
"""

# =============================================================================
# PHASE 2: Generate Edits for Each Feature
# =============================================================================

_FEATURE_EDIT_PROMPT = """\
You are an expert in adversarial machine learning and image editing.

Target class: **{class_name}**

I have identified these features in the image:
{feature_list}

## Your Task: Generate DETAILED Edit Instructions

For EACH feature, create a highly detailed edit instruction optimized for AI image editing.

### CRITICAL RULES FOR EFFECTIVE PROMPTS:

1. **BE EXTREMELY SPECIFIC AND DETAILED** (50-200 characters ideal):
   - BAD: "Remove the feature"
   - GOOD: "Remove the [specific feature] completely, blend the area smoothly with surrounding texture"

   - BAD: "Change the background"
   - GOOD: "Replace the entire background with a plain white studio backdrop, maintain sharp edges around the subject"

2. **NEVER REFERENCE THE CLASS NAME IN EDIT INSTRUCTIONS**:
   - BAD: "Make it look more like a {class_name}"
   - BAD: "Modify to resemble a {class_name}"
   - BAD: "Add {class_name} characteristics"
   - GOOD: "Add bright red breast feathers with white wing bars"
   - GOOD: "Replace texture with smooth dark gray rubbery skin"
   - Every edit MUST name a concrete visual property (color, shape, texture, pattern, size)

3. **DESCRIBE THE EXACT OUTCOME**:
   - Specify colors: "bright red", "pure white", "dark gray"
   - Specify textures: "smooth matte surface", "natural texture"
   - Specify what to preserve: "maintain the original pose", "keep the lighting consistent"

4. **USE NATURAL LANGUAGE DESCRIPTIONS**:
   - Describe the specific change with concrete details
   - Include colors, textures, and what should remain unchanged

5. **FOR REMOVALS - Describe what replaces it**:
   - BAD: "Remove the feature"
   - GOOD: "Remove [feature] completely, blend the area smoothly with the surrounding surface"

6. **FOR BACKGROUND CHANGES - Be vivid**:
   - "Replace background with a plain neutral gray backdrop"
   - "Change to a professional studio setting with soft diffused lighting"

### Edit Types:
- **removal**: Remove and blend/replace with surrounding texture
- **modification**: Change appearance significantly (color, texture, size)
- **replacement**: Replace with something completely different

### IMPORTANT: For contextual features (background, environment, co-occurring objects):
- Generate REMOVAL edits that NEUTRALIZE the environment
- "Replace the underwater background with a plain white studio backdrop"
- "Remove the fishing net entirely, fill with clean neutral gray background"
- "Remove all co-occurring objects, leaving only the main subject on neutral background"
- These test whether the model relies on environment rather than the object itself

Respond with JSON:
{{
  "feature_edits": [
    {{
      "feature_name": "the feature being tested",
      "edit_instruction": "DETAILED 50-200 char instruction describing exact changes, colors, textures, and what to maintain",
      "edit_type": "removal | modification | replacement",
      "expected_impact": "high | medium | low",
      "hypothesis": "what we expect to happen and why"
    }}
  ],
  "compound_edits": [
    {{
      "features": ["feature1", "feature2"],
      "edit_instruction": "detailed instruction combining multiple changes",
      "hypothesis": "why testing these together matters"
    }}
  ]
}}

Generate one edit per feature (max 8 edits). Keep edit instructions under 100 characters.
Skip compound_edits to keep response short.
"""

# =============================================================================
# PHASE 3: Final Analysis with Semantic Verification
# =============================================================================

_FINAL_ANALYSIS_PROMPT = """\
You are an expert in machine learning model analysis and bias detection.

Target class: **{class_name}**

## Feature Analysis Results

I tested how removing/modifying various features affects the model's confidence.
Here are the results:

{results_summary}

## Your Task: Semantic Classification of Features

**CRITICAL DISTINCTION:**
- **Essential features**: Features that are SEMANTICALLY PART OF what defines "{class_name}".
  These are the defining characteristics, parts, or attributes of a "{class_name}".
  If removing these decreases confidence, that is CORRECT model behavior.

- **Spurious features (shortcuts)**: Features that are NOT semantically related to "{class_name}"
  but the model incorrectly relies on them.
  These include: background, environment, co-occurring objects, lighting, image quality.
  If removing these decreases confidence, that indicates a MODEL BIAS/SHORTCUT.

### Classification Rules:
1. For each feature that affected classification, ask: "Is this feature part of what DEFINES a {class_name}?"
2. Parts, textures, and colors inherent to the class = ESSENTIAL
3. Background, context, co-occurring objects = SPURIOUS if they affect classification
4. The goal is to find features the model SHOULDN'T rely on but does

### Analysis Tasks:

1. **Feature Classification**: For each tested feature, classify as:
   - "essential" = semantically part of the class definition
   - "spurious" = NOT part of class definition (indicates bias if it affects classification)

2. **Shortcut Detection**: List features that are BOTH:
   - Spurious (not semantically related to class)
   - High impact (removing them significantly changed confidence)

3. **Robustness Assessment**: Based on how many shortcuts were found

Respond with JSON:
{{
  "feature_importance": [
    {{"feature": "name", "impact": "delta value", "is_semantic": true/false, "classification": "essential | spurious"}}
  ],
  "confirmed_shortcuts": [
    {{"feature": "name", "evidence": "why it's a shortcut (not semantically related to {class_name})", "severity": "high | medium | low"}}
  ],
  "legitimate_features": ["features the model correctly relies on"],
  "robustness_score": 1-10,
  "risk_level": "LOW | MEDIUM | HIGH",
  "vulnerabilities": ["list of main weaknesses"],
  "recommendations": ["list of suggestions to improve"],
  "summary": "2-3 sentence overall assessment"
}}
"""

_KNOWLEDGE_BASED_FEATURES_PROMPT = """\
You are an expert in image classification, model biases, and visual associations.

Target class: **{class_name}**

## Your Task: Use Your General Knowledge to Identify Potential Shortcut Features

Based on your world knowledge (NOT by looking at any specific image), list visual features that
are commonly ASSOCIATED with "{class_name}" but are NOT actually part of what defines it.

CRITICAL: When describing features or edit instructions, NEVER reference "{class_name}" by name.
Describe only concrete visual properties (colors, textures, shapes, patterns, sizes).
- BAD: "features that look like {class_name}"
- GOOD: "bright orange coloring with black vertical stripes"

These are features that an image classifier might incorrectly learn to rely on (shortcuts/biases):

### Categories to Consider:

**A) Co-occurring Objects** (things often seen WITH {class_name}):
   - What objects typically appear alongside or near a "{class_name}"?
   - Items commonly used with or by "{class_name}"

**B) Typical Environments/Settings**:
   - Indoor vs outdoor
   - Specific rooms or locations
   - Natural vs urban settings
   - What backgrounds are "{class_name}" usually photographed in?

**C) Photographic Patterns**:
   - Common camera angles
   - Typical framing (close-up vs wide shot)
   - Lighting conditions
   - Photo quality patterns

**D) Human/Cultural Associations**:
   - Text or logos that often appear
   - Human presence patterns
   - Seasonal associations
   - Time of day patterns

**E) Dataset Biases**:
   - What biases might exist in how "{class_name}" images were collected?
   - Are there common watermarks, borders, or image quality issues?

### Output Format

For each potential shortcut feature, explain:
1. What the feature is
2. Why it's commonly associated with "{class_name}"
3. Why relying on it would be WRONG (it's not definitional)

Respond with JSON:
{{
  "target_class": "{class_name}",
  "knowledge_based_features": [
    {{
      "feature": "specific feature name",
      "category": "co_occurring_object | environment | photographic | cultural | dataset_bias",
      "association_reason": "why this is commonly associated with {class_name}",
      "why_shortcut": "why relying on this is incorrect (not part of class definition)",
      "test_hypothesis": "how to test if the model relies on this",
      "expected_impact": "high | medium | low"
    }}
  ],
  "potential_edit_instructions": [
    {{
      "edit": "detailed edit instruction naming specific colors/textures/shapes (NEVER mention {class_name} by name)",
      "feature": "which feature this tests",
      "hypothesis": "if adding this increases {class_name} confidence, the model has this shortcut"
    }}
  ]
}}

List 5-8 potential shortcut features. Keep descriptions brief (under 30 words each).
Skip potential_edit_instructions to keep response short.
"""

_CONFUSING_CLASSES_PROMPT = """\
You are an expert in image classification and model biases.

Target class: **{class_name}**

I need to find classes that a classifier might CONFUSE with "{class_name}".
These should be classes that:
1. Share visual features (shape, texture, color)
2. Appear in similar contexts
3. Are semantically related

From the following list of available classes, select the {num_classes} MOST LIKELY to be confused with "{class_name}":

Available classes:
{class_list}

Respond with JSON:
{{
  "confusing_classes": ["class1", "class2", ...],
  "reasoning": {{
    "class1": "why this class might be confused",
    "class2": "why this class might be confused"
  }}
}}

Select exactly {num_classes} classes that are most likely to cause false positives or misclassifications.
"""

_ANALYSIS_PROMPT = """\
You are an expert in computer vision bias analysis and AI image editing.

I am analyzing an image classifier to find SHORTCUTS and BIASES.
The target class is: **{class_name}**

I will show you:
1. The original image (classified as "{class_name}" with {confidence:.1%} confidence)
2. A Grad-CAM heatmap showing model attention (red = high focus)

## Your Analysis Tasks:

### 1. Feature Identification
- List ALL visual features relevant to "{class_name}" (shape, color, texture, parts, context)
- Rank features by importance: which are ESSENTIAL (intrinsic to the class) vs SPURIOUS (contextual)?
- What does the Grad-CAM reveal about model attention?

### 2. Generate DETAILED Edit Instructions

Create diverse edits to test model robustness. Each edit instruction must be HIGHLY DETAILED (50-200 characters).

**CRITICAL RULES FOR EFFECTIVE AI IMAGE EDITING:**

0. **NEVER REFERENCE THE CLASS NAME IN EDIT INSTRUCTIONS**:
   - BAD: "Make it look more like a {class_name}"
   - BAD: "Add {class_name} features"
   - GOOD: "Add bright red breast feathers along the chest area"
   - Each edit must describe concrete visual changes only

1. **BE EXTREMELY SPECIFIC AND DETAILED**:
   - BAD: "Remove the feature"
   - GOOD: "Remove [specific feature] completely, blend the area smoothly with surrounding texture"

   - BAD: "Change background"
   - GOOD: "Replace the entire background with a clean white studio backdrop, maintain sharp edges around the subject"

2. **DESCRIBE EXACT COLORS AND TEXTURES**:
   - Specify colors, textures, and details precisely

3. **FOR REMOVALS - Describe what replaces it**:
   - Always specify how to blend or fill the area

4. **FOR BACKGROUND CHANGES - Be vivid and specific**:
   - Describe the new background with concrete details

**Include these types of edits:**

A) **Single Feature Removal** (with detailed blending instructions)
B) **Compound Feature Removal** (multiple changes in one detailed instruction)
C) **Background/Context Changes** (vivid scene descriptions)
D) **Texture/Color Transformations** (specific color names and textures)

Respond with JSON:
{{
  "key_features": ["feature1", "feature2", ...],
  "essential_features": ["features the model SHOULD rely on (intrinsic to {class_name})"],
  "spurious_features": ["features that might be shortcuts (contextual, not intrinsic)"],
  "model_focus": "what Grad-CAM reveals",
  "edit_instructions": [
    {{
      "edit": "...",
      "hypothesis": "...",
      "type": "...",
      "priority": 5
    }},
    ...
  ]
}}

Generate {max_hypotheses} diverse edit instructions. Prioritize edits most likely to reveal shortcuts.

IMPORTANT: Be SPECIFIC. Never use "or", "such as", "for example". Each instruction = ONE concrete action.
"""

_ITERATIVE_REFINEMENT_PROMPT = """\
You are an expert in computer vision bias analysis. You are iteratively refining hypotheses about model shortcuts.

Target class: **{class_name}**

## Previous Edit Results:
{previous_results}

## Your Task:
Based on these results, generate NEW hypotheses to test. Consider:

1. **What worked**: Edits with large confidence changes reveal important features
2. **What didn't work**: Small changes suggest the feature isn't critical OR the edit wasn't effective
3. **Patterns**: Are there common themes in successful edits?
4. **Unexplored areas**: What features haven't been tested yet?

Generate {max_hypotheses} NEW edit instructions that:
- Build on successful edits (try variations, combinations)
- Avoid repeating failed approaches
- Test features not yet explored
- Try more aggressive/subtle versions of promising edits

### CRITICAL: Be SPECIFIC and DECISIVE
- NEVER use "or", "such as", "for example", "like" in edit instructions
- NEVER reference the target class name in edit instructions
- BAD: "Add {class_name} features" or "make it resemble a {class_name}"
- GOOD: "Add elongated curved beak with dark brown coloring"
- Each edit must be ONE concrete, specific action
- BAD: "Change background to a nature scene like forest or beach"
- GOOD: "Replace the background with a solid blue color"

For each edit provide:
- "edit": Clear, specific, single-action instruction (NO alternatives!)
- "hypothesis": Expected result based on previous findings
- "type": "feature_removal" | "feature_addition" | "background_change" | "compound"
- "priority": 1-5 (5 = most promising based on previous results)
- "rationale": Why this edit is worth trying given previous results

Respond with JSON:
{{
  "insights": ["key insight 1 from previous results", "insight 2", ...],
  "confirmed_shortcuts": ["features confirmed as shortcuts"],
  "needs_more_testing": ["features that need more investigation"],
  "edit_instructions": [
    {{
      "edit": "...",
      "hypothesis": "...",
      "type": "...",
      "priority": 5,
      "rationale": "..."
    }}
  ]
}}
"""

_NEGATIVE_ANALYSIS_PROMPT = """\
You are an expert in adversarial examples, model biases, and AI image editing.

Target class: **{class_name}**
This image is NOT a "{class_name}" — it is a "{true_label}".
Current confidence for "{class_name}": {confidence:.1%}

## Goal: Find SHORTCUTS that trick the model into misclassifying this as "{class_name}"

Test whether the model relies on SUPERFICIAL VISUAL CORRELATES (colors, textures, patterns)
or ENVIRONMENTAL CONTEXT instead of the actual defining features of "{class_name}".

### What is FORBIDDEN vs ALLOWED

**FORBIDDEN — Morphological/structural features (do NOT generate these):**
- Body parts and anatomy (fins, beaks, feathers, teeth, eyes, limbs, wings, tails)
- Body shape or silhouette changes
- Structural features that physically define "{class_name}"

**ALLOWED — Superficial visual correlates (colors, textures, patterns):**
- Colors that correlate with "{class_name}" but do not define it
- Textures and surface qualities associated with "{class_name}"
- Patterns commonly seen on "{class_name}" in training images
- These are properties that MANY different objects could have

**ALLOWED — Environmental/contextual elements:**
- Backgrounds and habitats where "{class_name}" is typically photographed
- Objects commonly found alongside "{class_name}" in training images
- Lighting and weather conditions typical of "{class_name}" photos
- Surfaces and materials common in "{class_name}" dataset images

### CRITICAL: Write HIGHLY DETAILED edit instructions (50-200 characters)

- NEVER reference the class name "{class_name}" in edit instructions
- Every instruction MUST name concrete visual properties only

### Types of Additions to Test:

A) **Color Correlates**: Colors associated with "{class_name}" in training data
   - Example for tench: "Change the subject's body color to olive green with golden undertones"
   - Example for robin: "Paint the chest area bright orange-red with sharp color boundary"

B) **Texture/Surface Correlates**: Surface properties associated with "{class_name}"
   - Example for shark: "Change skin texture to smooth dark gray rubbery surface"
   - Example for poodle: "Add fluffy white curly texture across the entire body surface"

C) **Pattern Correlates**: Visual patterns associated with "{class_name}"
   - Example for zebra: "Add bold black and white vertical stripes across the body"
   - Example for leopard: "Overlay dark rosette spots with tan centers across the surface"

D) **Habitat/Setting**: Typical backgrounds where "{class_name}" appears
   - Example for fish: "Replace background with murky green pond water with lily pads"
   - Example for bird: "Add wooden bird feeder with scattered seeds in foreground"

E) **Co-occurring Objects**: Things found alongside "{class_name}" in photos
   - Example for rooster: "Add wooden fence posts and scattered hay on the ground"
   - Example for shark: "Add a diving cage with metal bars in the foreground"

For each edit provide:
- "edit": DETAILED instruction (50-200 chars) with specific colors, textures, positions
- "hypothesis": Why this might fool the model into classifying as "{class_name}"
- "type": "correlate_addition" | "context_addition" | "environment_addition" | "co_object_addition"
- "priority": 1-5 (5 = most likely to trigger false positive)

Respond with JSON:
{{
  "common_mistake_triggers": ["visual shortcuts the model may use for {class_name}"],
  "edit_instructions": [
    {{
      "edit": "...",
      "hypothesis": "...",
      "type": "...",
      "priority": 5
    }}
  ]
}}

Generate {max_hypotheses} diverse edits. Include AT LEAST HALF as correlate_addition (color, texture, pattern).

CRITICAL: Be SPECIFIC. Never use "or", "such as", "for example". Each instruction = ONE concrete action.
Do NOT add structural body parts of "{class_name}" — test only superficial properties and context.
"""


_ENVIRONMENTAL_ANALYSIS_PROMPT = """\
You are an expert in dataset bias analysis and image classification.

Target class: **{class_name}**

I am showing you {num_images} sample images, all classified as "{class_name}".

## Your Task: Find RECURRING environmental patterns across these images

Analyze the ENVIRONMENTS and CONTEXTS across all {num_images} images.
Identify patterns that appear in MULTIPLE images — these indicate dataset bias.

### What to look for:
- **Backgrounds**: Water color, sky type, indoor/outdoor setting, studio vs natural
- **Settings**: Specific locations, habitats, terrain types
- **Lighting**: Time of day, artificial vs natural, flash patterns
- **Surfaces**: What the subject sits on, ground type, table material
- **Co-occurring objects**: Items that repeatedly appear alongside the subject
- **Framing**: Camera angle, distance, composition patterns

### Rules:
- Only report patterns found in 3 or more images
- Be SPECIFIC about what you observe (exact colors, materials, conditions)
- For each pattern, write a REMOVAL edit instruction (50-200 chars)
- NEVER reference "{class_name}" in edit instructions

Respond with JSON:
{{
  "environmental_patterns": [
    {{
      "pattern": "specific environmental element observed",
      "category": "background | setting | lighting | surface | co_object | framing",
      "frequency": "N out of {num_images}",
      "removal_edit": "detailed instruction to neutralize this element (50-200 chars)",
      "hypothesis": "why the model might rely on this environmental cue"
    }}
  ]
}}

List 4-8 patterns, ordered by frequency (most common first).
"""


@dataclass
class EnvironmentalPattern:
    """A recurring environmental pattern found across sample images."""

    pattern: str
    category: str  # background, setting, lighting, surface, co_object, framing
    frequency: str
    removal_edit: str
    hypothesis: str


@dataclass
class KnowledgeBasedFeature:
    """A potential shortcut feature identified from VLM's general knowledge."""
    feature: str
    category: str  # co_occurring_object, environment, photographic, cultural, dataset_bias
    association_reason: str
    why_shortcut: str
    test_hypothesis: str
    expected_impact: str  # high, medium, low
    edit_instruction: str = ""  # Generated edit to test this feature


@dataclass
class DetectedFeature:
    """A visual feature detected in the image."""
    name: str
    category: str  # object_part, texture, color, shape, context
    feature_type: str  # intrinsic (part of object) or contextual (background/environment)
    location: str
    gradcam_attention: str  # high, medium, low
    reasoning: str
    # These are determined AFTER experiments, not before
    is_shortcut: bool = False  # Set to True if contextual feature affects classification


@dataclass
class FeatureDiscovery:
    """Results from Phase 1: Feature Discovery."""
    class_name: str
    features: list[DetectedFeature] = field(default_factory=list)
    gradcam_summary: str = ""
    intrinsic_features: list[str] = field(default_factory=list)  # Part of the object
    contextual_features: list[str] = field(default_factory=list)  # Background/environment
    raw_response: str = ""

    # For backwards compatibility
    @property
    def potential_shortcuts(self) -> list[str]:
        return self.contextual_features

    @property
    def robust_features(self) -> list[str]:
        return self.intrinsic_features


@dataclass
class FeatureEditPlan:
    """Edit plan for a specific feature."""
    feature_name: str
    edit_instruction: str
    edit_type: str  # removal, modification, replacement
    expected_impact: str  # high, medium, low
    hypothesis: str


@dataclass
class FinalAnalysis:
    """Results from Phase 3: Final Analysis."""
    class_name: str
    feature_importance: list[dict] = field(default_factory=list)
    confirmed_shortcuts: list[dict] = field(default_factory=list)
    legitimate_features: list[str] = field(default_factory=list)
    robustness_score: int = 5
    risk_level: str = "MEDIUM"
    vulnerabilities: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""


@dataclass
class EditInstruction:
    edit: str
    hypothesis: str
    type: str
    target: str  # "positive" or "negative"
    priority: int = 3  # 1-5, higher = more likely to reveal bias
    image_index: int = 0
    source_class: str = ""  # original class of the image (for negatives)


@dataclass
class VLMAnalysis:
    class_name: str
    key_features: list[str] = field(default_factory=list)
    essential_features: list[str] = field(default_factory=list)
    spurious_features: list[str] = field(default_factory=list)
    common_mistake_triggers: list[str] = field(default_factory=list)
    model_focus: str = ""
    edit_instructions: list[EditInstruction] = field(default_factory=list)
    raw_response: str = ""
    # Iterative refinement fields
    insights: list[str] = field(default_factory=list)
    confirmed_shortcuts: list[str] = field(default_factory=list)
    needs_more_testing: list[str] = field(default_factory=list)


class QwenVLAnalyzer:
    """Loads Qwen2.5-VL-7B-Instruct and provides analysis methods."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import transformers
        import logging as _logging

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, dtype)
        self.model_name = model_name

        # Suppress all loading output
        transformers.logging.set_verbosity_error()
        _logging.getLogger("transformers").setLevel(_logging.ERROR)

        logger.debug("Loading VLM: %s on %s", model_name, self.device)
        with suppress_output():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.loaded = True
        logger.debug("VLM loaded")

    def offload(self):
        """Free VRAM by deleting the model."""
        import torch
        import gc
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        self.loaded = False

    def load_to_gpu(self):
        """Reload model to GPU if it was offloaded."""
        if not self.loaded:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import transformers
            import logging as _logging
            transformers.logging.set_verbosity_error()
            _logging.getLogger("transformers").setLevel(_logging.ERROR)

            with suppress_output():
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model.eval()
            self.loaded = True

    # ------------------------------------------------------------------
    # Class selection
    # ------------------------------------------------------------------

    def select_confusing_classes(
        self,
        target_class: str,
        available_classes: list[str],
        num_classes: int = 5,
    ) -> list[str]:
        """
        Use VLM to select classes most likely to be confused with the target.

        Args:
            target_class: The class we're analyzing
            available_classes: List of all available class names
            num_classes: How many confusing classes to select

        Returns:
            List of class names most likely to cause confusion
        """
        # Limit the list to avoid token limits (sample evenly)
        max_classes = 200
        if len(available_classes) > max_classes:
            step = len(available_classes) // max_classes
            sampled_classes = available_classes[::step][:max_classes]
        else:
            sampled_classes = available_classes

        # Remove the target class itself
        sampled_classes = [c for c in sampled_classes if c.lower() != target_class.lower()]

        prompt = _CONFUSING_CLASSES_PROMPT.format(
            class_name=target_class,
            num_classes=num_classes,
            class_list=", ".join(sampled_classes[:100]),  # Further limit for prompt
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            raw = self._run(messages, method_name="select_confusing_classes")
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                data = json.loads(self._repair_json(json_match.group()))
                confusing = data.get("confusing_classes", [])
                # Validate that returned classes are in the available list
                valid_classes = []
                available_lower = {c.lower(): c for c in available_classes}
                for c in confusing:
                    if c.lower() in available_lower:
                        valid_classes.append(available_lower[c.lower()])
                if valid_classes:
                    logger.debug("VLM selected confusing classes for %s: %s",
                               target_class, valid_classes)
                    return valid_classes[:num_classes]
        except Exception as e:
            logger.warning("Failed to get confusing classes from VLM: %s", e)

        # Fallback: return random classes
        import random
        fallback = random.sample(sampled_classes, min(num_classes, len(sampled_classes)))
        logger.debug("Using random fallback classes: %s", fallback)
        return fallback

    # ------------------------------------------------------------------
    # Knowledge-Based Feature Discovery (no image required)
    # ------------------------------------------------------------------

    def generate_knowledge_based_features(
        self,
        class_name: str,
        all_class_names: list[str] | None = None,
    ) -> dict:
        """
        Use VLM's general knowledge to identify potential shortcut features.

        This discovers features that are commonly ASSOCIATED with the class
        but are NOT definitional (co-occurring objects, typical environments, etc.).

        Args:
            class_name: Target class to analyze
            all_class_names: Optional list of all classifier classes for context

        Returns:
            dict with:
                - knowledge_based_features: list of potential shortcuts
                - potential_edit_instructions: edits to test these features
        """
        prompt = _KNOWLEDGE_BASED_FEATURES_PROMPT.format(class_name=class_name)

        # Add context about available classes if provided
        if all_class_names:
            # Sample a subset to avoid token limits
            sampled = all_class_names[::max(1, len(all_class_names) // 50)][:50]
            context = f"\n\nFor context, the classifier recognizes these classes (subset): {', '.join(sampled)}"
            prompt += context

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            logger.debug("generate_knowledge_based_features: Analyzing %s", class_name)
            raw = self._run(messages, method_name="generate_knowledge_based_features")

            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                json_str = json_match.group()
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = self._repair_json(json_str)
                    data = json.loads(json_str)

                result = {
                    "target_class": class_name,
                    "knowledge_based_features": data.get("knowledge_based_features", []),
                    "potential_edit_instructions": data.get("potential_edit_instructions", []),
                    "raw_response": raw,
                }

                logger.debug(
                    "generate_knowledge_based_features: Found %d potential shortcuts for %s",
                    len(result["knowledge_based_features"]), class_name
                )
                return result

        except Exception as e:
            logger.warning("Failed to generate knowledge-based features: %s", e)

        return {
            "target_class": class_name,
            "knowledge_based_features": [],
            "potential_edit_instructions": [],
            "raw_response": "",
        }

    # ------------------------------------------------------------------
    # Phase 1: Feature Discovery
    # ------------------------------------------------------------------

    def discover_features(
        self,
        image: Image.Image,
        gradcam: Image.Image,
        class_name: str,
        confidence: float,
    ) -> FeatureDiscovery:
        """
        Phase 1: Analyze image and Grad-CAM to discover all visual features.
        """
        prompt = _FEATURE_DISCOVERY_PROMPT.format(
            class_name=class_name,
            confidence=confidence,
        )

        content = [
            {"type": "image", "image": image},
            {"type": "image", "image": gradcam},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        logger.debug("discover_features: Analyzing %s (confidence=%.2f)", class_name, confidence)
        raw = self._run(messages, method_name="discover_features")
        return self._parse_feature_discovery(raw, class_name)

    def _parse_feature_discovery(self, raw: str, class_name: str) -> FeatureDiscovery:
        """Parse feature discovery response."""
        result = FeatureDiscovery(class_name=class_name, raw_response=raw)
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return result

            json_str = json_match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = self._repair_json(json_str)
                data = json.loads(json_str)
            result.gradcam_summary = data.get("gradcam_summary", "")
            result.intrinsic_features = data.get("intrinsic_features", [])
            result.contextual_features = data.get("contextual_features", [])

            for f in data.get("detected_features", []):
                result.features.append(DetectedFeature(
                    name=f.get("name", ""),
                    category=f.get("category", "unknown"),
                    feature_type=f.get("feature_type", "intrinsic"),
                    location=f.get("location", ""),
                    gradcam_attention=f.get("gradcam_attention", "medium"),
                    reasoning=f.get("reasoning", ""),
                ))
        except Exception as e:
            logger.warning("Failed to parse feature discovery: %s", e)

        return result

    # ------------------------------------------------------------------
    # Phase 2: Generate Feature-Based Edits
    # ------------------------------------------------------------------

    def generate_feature_edits(
        self,
        image: Image.Image,
        features: list[DetectedFeature],
        class_name: str,
    ) -> list[FeatureEditPlan]:
        """
        Phase 2: Generate specific edit instructions for each detected feature.
        """
        feature_list = "\n".join([
            f"- {f.name} ({f.category}): {f.reasoning} [attention: {f.gradcam_attention}]"
            for f in features
        ])

        prompt = _FEATURE_EDIT_PROMPT.format(
            class_name=class_name,
            feature_list=feature_list,
        )

        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        logger.debug("generate_feature_edits: Generating edits for %s (%d features)", class_name, len(features))
        raw = self._run(messages, method_name="generate_feature_edits")
        return self._parse_feature_edits(raw)

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON issues from LLM output.
        Handles: trailing commas, unquoted keys, truncated output, control chars.
        """
        # Remove markdown code fences if present
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)

        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

        # Remove any control characters (except newlines/tabs in strings)
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)

        # Handle truncated JSON by closing open brackets/braces
        # Count unclosed brackets
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')

        if open_braces > 0 or open_brackets > 0:
            # Truncated JSON detected - try to salvage it
            logger.debug("Repairing truncated JSON: %d unclosed braces, %d unclosed brackets",
                        open_braces, open_brackets)

            # Remove incomplete last item (truncated string/object)
            # Find last complete item by looking for last complete key-value or array element
            json_str = re.sub(r',\s*"[^"]*$', '', json_str)  # Remove incomplete string at end
            json_str = re.sub(r',\s*\{[^}]*$', '', json_str)  # Remove incomplete object at end
            json_str = re.sub(r',\s*\[[^\]]*$', '', json_str)  # Remove incomplete array at end
            json_str = re.sub(r':\s*"[^"]*$', ': ""', json_str)  # Fix incomplete string value
            json_str = re.sub(r':\s*-?\d+\.?\d*$', ': 0', json_str)  # Fix incomplete number
            json_str = re.sub(r':\s*$', ': null', json_str)  # Fix missing value entirely
            json_str = re.sub(r'"[^"]*:\s*$', '"": null', json_str)  # Fix incomplete key

            # Remove trailing commas again after cleanup
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

            # Close remaining brackets/braces
            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')

            json_str = json_str.rstrip()
            # Remove trailing comma before closing
            json_str = re.sub(r',\s*$', '', json_str)

            # Add closing brackets in correct order (inner first)
            # This is a simplification - assumes arrays close before objects
            json_str += ']' * open_brackets
            json_str += '}' * open_braces

        return json_str

    def _parse_feature_edits(self, raw: str) -> list[FeatureEditPlan]:
        """Parse feature edit response."""
        edits = []
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return edits

            json_str = json_match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to repair the JSON
                json_str = self._repair_json(json_str)
                data = json.loads(json_str)

            for e in data.get("feature_edits", []):
                edits.append(FeatureEditPlan(
                    feature_name=e.get("feature_name", ""),
                    edit_instruction=e.get("edit_instruction", ""),
                    edit_type=e.get("edit_type", "removal"),
                    expected_impact=e.get("expected_impact", "medium"),
                    hypothesis=e.get("hypothesis", ""),
                ))

            # Also add compound edits
            for e in data.get("compound_edits", []):
                edits.append(FeatureEditPlan(
                    feature_name="+".join(e.get("features", [])),
                    edit_instruction=e.get("edit_instruction", ""),
                    edit_type="compound",
                    expected_impact="high",
                    hypothesis=e.get("hypothesis", ""),
                ))
        except Exception as e:
            logger.warning("Failed to parse feature edits: %s", e)

        return edits

    # ------------------------------------------------------------------
    # Phase 3: Final Analysis
    # ------------------------------------------------------------------

    def final_analysis(
        self,
        image: Image.Image,
        class_name: str,
        results: list[dict],
    ) -> FinalAnalysis:
        """
        Phase 3: Analyze all results and provide comprehensive conclusions.

        Args:
            image: Original image
            class_name: Target class
            results: List of dicts with {feature, edit, delta, confirmed}
        """
        results_summary = "\n".join([
            f"- Feature: {r.get('feature', 'unknown')}\n"
            f"  Edit: {r.get('edit', '')}\n"
            f"  Confidence change: {r.get('delta', 0):+.1%}\n"
            f"  Confirmed shortcut: {'YES' if r.get('confirmed') else 'no'}"
            for r in results
        ])

        prompt = _FINAL_ANALYSIS_PROMPT.format(
            class_name=class_name,
            results_summary=results_summary,
        )

        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        logger.debug("final_analysis: Analyzing %s with %d results", class_name, len(results))
        raw = self._run(messages, method_name="final_analysis")
        return self._parse_final_analysis(raw, class_name)

    def _parse_final_analysis(self, raw: str, class_name: str) -> FinalAnalysis:
        """Parse final analysis response."""
        result = FinalAnalysis(class_name=class_name, raw_response=raw)
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return result

            data = json.loads(self._repair_json(json_match.group()))
            result.feature_importance = data.get("feature_importance", [])
            result.confirmed_shortcuts = data.get("confirmed_shortcuts", [])
            result.legitimate_features = data.get("legitimate_features", [])
            result.robustness_score = data.get("robustness_score", 5)
            result.risk_level = data.get("risk_level", "MEDIUM")
            result.vulnerabilities = data.get("vulnerabilities", [])
            result.recommendations = data.get("recommendations", [])
            result.summary = data.get("summary", "")
        except Exception as e:
            logger.warning("Failed to parse final analysis: %s", e)

        return result

    # ------------------------------------------------------------------
    # Legacy analysis methods
    # ------------------------------------------------------------------

    def analyze_positive(self, image: Image.Image, gradcam: Image.Image | None,
                          class_name: str, confidence: float,
                          max_hypotheses: int = 3) -> VLMAnalysis:
        """Analyze a positive example and generate edit instructions."""
        prompt = _ANALYSIS_PROMPT.format(
            class_name=class_name,
            confidence=confidence,
            max_hypotheses=max_hypotheses,
        )

        # Build content with images first, then text
        content = [{"type": "image", "image": image}]
        if gradcam is not None:
            content.append({"type": "image", "image": gradcam})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        logger.debug("analyze_positive: Analyzing %s (confidence=%.2f)", class_name, confidence)
        raw = self._run(messages, method_name="analyze_positive")
        return self._parse_analysis(raw, class_name, "positive")

    def analyze_negative(self, image: Image.Image, class_name: str,
                          true_label: str, confidence: float,
                          max_hypotheses: int = 3) -> VLMAnalysis:
        """Generate edit instructions for a negative example (false positive detection)."""
        prompt = _NEGATIVE_ANALYSIS_PROMPT.format(
            class_name=class_name,
            true_label=true_label,
            confidence=confidence,
            max_hypotheses=max_hypotheses,
        )

        # Image first, then text
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        logger.debug("analyze_negative: Analyzing %s vs %s (confidence=%.2f)", class_name, true_label, confidence)
        raw = self._run(messages, method_name="analyze_negative")
        return self._parse_analysis(raw, class_name, "negative")

    def analyze_environmental_patterns(
        self,
        images: list[Image.Image],
        class_name: str,
    ) -> list[EnvironmentalPattern]:
        """Find recurring environmental patterns across multiple sample images."""
        prompt = _ENVIRONMENTAL_ANALYSIS_PROMPT.format(
            class_name=class_name,
            num_images=len(images),
        )
        content = self._build_multi_image_content(images, prompt)
        messages = [{"role": "user", "content": content}]
        logger.debug("analyze_environmental_patterns: %s (%d images)", class_name, len(images))
        raw = self._run(messages, method_name="analyze_environmental_patterns")
        return self._parse_environmental_patterns(raw)

    def _build_multi_image_content(
        self,
        images: list[Image.Image],
        prompt: str,
    ) -> list[dict]:
        """Build VLM content list with multiple images followed by text."""
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        return content

    def _parse_environmental_patterns(self, raw: str) -> list[EnvironmentalPattern]:
        """Parse VLM response into EnvironmentalPattern objects."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                logger.warning("No JSON found in environmental analysis response")
                return []
            data = json.loads(self._repair_json(json_match.group()))
            return self._convert_pattern_dicts(data.get("environmental_patterns", []))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse environmental patterns: %s", e)
            return []

    def _convert_pattern_dicts(self, patterns: list[dict]) -> list[EnvironmentalPattern]:
        """Convert raw dicts to EnvironmentalPattern dataclass instances."""
        result = []
        for p in patterns:
            result.append(EnvironmentalPattern(
                pattern=p.get("pattern", ""),
                category=p.get("category", ""),
                frequency=p.get("frequency", ""),
                removal_edit=p.get("removal_edit", ""),
                hypothesis=p.get("hypothesis", ""),
            ))
        return result

    def analyze_iterative(
        self,
        image: Image.Image,
        edited_images: list[Image.Image],
        class_name: str,
        previous_results: list[dict],
        max_hypotheses: int = 5,
    ) -> VLMAnalysis:
        """
        Analyze previous edit results and generate refined hypotheses.

        Args:
            image: Original image
            edited_images: List of edited images from previous iteration
            class_name: Target class name
            previous_results: List of dicts with keys:
                - edit: str (the edit instruction)
                - original_confidence: float
                - edited_confidence: float
                - delta: float
                - confirmed: bool
            max_hypotheses: Number of new hypotheses to generate
        """
        # Format previous results for the prompt
        results_text = []
        for i, r in enumerate(previous_results):
            status = "CONFIRMED" if r.get("confirmed") else "not confirmed"
            delta = r.get("delta", 0)
            direction = "↓" if delta < 0 else "↑" if delta > 0 else "→"
            results_text.append(
                f"{i+1}. \"{r.get('edit', 'N/A')}\"\n"
                f"   Confidence: {r.get('original_confidence', 0):.1%} {direction} {r.get('edited_confidence', 0):.1%} "
                f"(Δ = {delta:+.1%}) [{status}]"
            )

        prompt = _ITERATIVE_REFINEMENT_PROMPT.format(
            class_name=class_name,
            previous_results="\n".join(results_text),
            max_hypotheses=max_hypotheses,
        )

        # Show original + some edited images for context
        content = [{"type": "image", "image": image}]
        # Add up to 4 edited images
        for img in edited_images[:4]:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        logger.debug("analyze_iterative: Analyzing %s with %d previous results, %d edited images",
                    class_name, len(previous_results), len(edited_images))
        raw = self._run(messages, method_name="analyze_iterative")
        return self._parse_iterative_analysis(raw, class_name)

    # ------------------------------------------------------------------

    def _run(self, messages: list[dict], method_name: str = "unknown") -> str:
        from qwen_vl_utils import process_vision_info
        import torch

        # Debug: Log the prompt being sent to VLM
        logger.debug("=" * 80)
        logger.debug("VLM._run called from: %s", method_name)
        logger.debug("-" * 40)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", [])
            if isinstance(content, str):
                logger.debug("VLM INPUT [%s]: %s", role, content[:500] + "..." if len(content) > 500 else content)
            else:
                for item in content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        logger.debug("VLM INPUT [%s/text]: %s", role, text[:1000] + "..." if len(text) > 1000 else text)
                    elif item.get("type") == "image":
                        logger.debug("VLM INPUT [%s/image]: <image provided>", role)
        logger.debug("-" * 40)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move all tensors to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        logger.debug("VLM generating response (max_new_tokens=2048, temp=0.1)...")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Reduced to save VRAM
                temperature=0.1,
                do_sample=False,
            )

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        # Debug: Log the full VLM response
        logger.debug("VLM OUTPUT (method=%s):", method_name)
        logger.debug("%s", response)
        logger.debug("=" * 80)

        return response

    def _parse_analysis(self, raw: str, class_name: str, default_target: str) -> VLMAnalysis:
        """Extract JSON from VLM response."""
        analysis = VLMAnalysis(class_name=class_name, raw_response=raw)
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                logger.warning("No JSON found in VLM response")
                return analysis

            data: dict[str, Any] = json.loads(self._repair_json(json_match.group()))
            analysis.key_features = data.get("key_features", [])
            analysis.essential_features = data.get("essential_features", [])
            analysis.spurious_features = data.get("spurious_features", [])
            analysis.common_mistake_triggers = data.get("common_mistake_triggers", [])
            analysis.model_focus = data.get("model_focus", "")

            # Parse edit instructions and sort by priority
            instructions = []
            for i, instr in enumerate(data.get("edit_instructions", [])):
                instructions.append(EditInstruction(
                    edit=instr.get("edit", ""),
                    hypothesis=instr.get("hypothesis", ""),
                    type=instr.get("type", "feature_addition"),
                    target=instr.get("target", default_target),
                    priority=instr.get("priority", 3),
                    image_index=i,
                ))
            # Sort by priority (highest first)
            analysis.edit_instructions = sorted(instructions, key=lambda x: x.priority, reverse=True)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse VLM response: %s", e)

        return analysis

    def _parse_iterative_analysis(self, raw: str, class_name: str) -> VLMAnalysis:
        """Extract JSON from iterative refinement response."""
        analysis = VLMAnalysis(class_name=class_name, raw_response=raw)
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                logger.warning("No JSON found in VLM iterative response")
                return analysis

            data: dict[str, Any] = json.loads(self._repair_json(json_match.group()))
            analysis.insights = data.get("insights", [])
            analysis.confirmed_shortcuts = data.get("confirmed_shortcuts", [])
            analysis.needs_more_testing = data.get("needs_more_testing", [])

            # Parse edit instructions
            instructions = []
            for i, instr in enumerate(data.get("edit_instructions", [])):
                instructions.append(EditInstruction(
                    edit=instr.get("edit", ""),
                    hypothesis=instr.get("hypothesis", ""),
                    type=instr.get("type", "feature_removal"),
                    target="positive",  # Iterative analysis focuses on positive samples
                    priority=instr.get("priority", 3),
                    image_index=0,
                ))
            analysis.edit_instructions = sorted(
                instructions, key=lambda x: x.priority, reverse=True
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse VLM iterative response: %s", e)

        return analysis

    def verify_edit(self, original: Image.Image, edited: Image.Image,
                    instruction: str) -> dict:
        """
        Verify if an edit was correctly applied by comparing original and edited images.

        Args:
            original: Original image before editing
            edited: Edited image
            instruction: The edit instruction that was applied

        Returns:
            dict with:
                - edit_applied: bool - whether the edit appears to have been applied
                - confidence: float - 0-1 confidence in the assessment
                - description: str - what changed in the image
                - issues: list[str] - any problems detected
        """
        prompt = f"""\
Compare these two images. The first is the original, the second should have this edit applied:
"{instruction}"

Analyze whether the edit was successfully applied.

Respond with JSON:
{{
  "edit_applied": true or false,
  "confidence": 0.0 to 1.0 (how confident are you),
  "description": "what changed between the images",
  "issues": ["any problems with the edit, or empty list if none"]
}}

Be strict: if the edit instruction says to remove something and it's still visible, that's a failure.
If the instruction says to change something and it looks the same, that's a failure.
"""

        content = [
            {"type": "image", "image": original},
            {"type": "image", "image": edited},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        try:
            logger.debug("verify_edit: Checking if edit was applied: %s", instruction[:50])
            raw = self._run(messages, method_name="verify_edit")
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                result = json.loads(self._repair_json(json_match.group()))
                return {
                    "edit_applied": bool(result.get("edit_applied", False)),
                    "confidence": float(result.get("confidence", 0.5)),
                    "description": str(result.get("description", "")),
                    "issues": list(result.get("issues", [])),
                }
        except Exception as e:
            logger.warning("Failed to verify edit: %s", e)

        return {
            "edit_applied": True,  # Assume success if verification fails
            "confidence": 0.5,
            "description": "Verification failed",
            "issues": ["Could not verify edit"],
        }

    def classify_features(self, class_name: str, features: list[dict]) -> list[dict]:
        """
        Use VLM to classify features as intrinsic or contextual.

        This is more accurate than keyword matching because the VLM understands
        semantic relationships (e.g., "wooden floor" is contextual for a cat,
        but might be intrinsic for furniture).

        Args:
            class_name: The target class (e.g., "tabby cat")
            features: List of dicts with 'instruction' and 'hypothesis' keys

        Returns:
            List of dicts with added 'feature_type' and 'feature_name' keys
        """
        if not features:
            return features

        # Build feature list for prompt
        feature_list = "\n".join([
            f"{i+1}. Edit: \"{f.get('instruction', '')}\"\n   Hypothesis: {f.get('hypothesis', '')}"
            for i, f in enumerate(features)
        ])

        prompt = f"""\
You are an expert in computer vision and semantic analysis.

Target class: **{class_name}**

I have tested these edits on images and measured their impact on classifier confidence.
For each edit, classify whether the feature being tested is INTRINSIC or CONTEXTUAL.

## Definitions:

**INTRINSIC features** = Features that are semantically PART OF the object itself.
- For "{class_name}": parts, inherent textures, colors, shapes that DEFINE what a "{class_name}" is
- Anything that would still be present if you isolated just the "{class_name}" from its environment

**CONTEXTUAL features** = Features that are NOT part of the object itself.
- Background, environment, setting, lighting, co-occurring objects
- Anything that describes WHERE or HOW the "{class_name}" is photographed, not WHAT it is

## Features to classify:

{feature_list}

## Your Task:

For EACH feature above, respond with JSON:
{{
  "classifications": [
    {{
      "index": 1,
      "feature_name": "simplified name (e.g., 'Main feature', 'Background', 'Color pattern')",
      "feature_type": "intrinsic" or "contextual",
      "reasoning": "brief explanation"
    }},
    ...
  ]
}}

Be accurate! The distinction matters:
- If removing an INTRINSIC feature drops confidence → model correctly uses it (GOOD)
- If removing a CONTEXTUAL feature drops confidence → model has a shortcut/bias (BAD)
"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            self._classify_via_vlm(messages, features)
        except Exception as e:
            logger.warning("Failed to classify features: %s", e)
            self._classify_via_keywords(features)

        return features

    def _classify_via_vlm(self, messages: list, features: list[dict]):
        """Run VLM classification and map results back to features."""
        logger.debug("classify_features: Classifying %d features", len(features))
        raw = self._run(messages, method_name="classify_features")
        logger.debug("classify_features: VLM response: %s", raw[:500])
        classifications = self._extract_classifications(raw)
        self._apply_classifications(classifications, features)
        self._fill_unclassified(features)

    def _apply_classifications(self, classifications: list[dict], features: list[dict]):
        """Map VLM classification results back to feature dicts."""
        for c in classifications:
            idx = c.get("index", 0) - 1
            if 0 <= idx < len(features):
                features[idx]["feature_type"] = self._normalize_feature_type(
                    c.get("feature_type", "unknown"),
                )
                features[idx]["feature_name"] = c.get("feature_name", "")

    def _classify_via_keywords(self, features: list[dict]):
        """Apply keyword-based fallback to all features."""
        for f in features:
            f["feature_type"] = self._fallback_classify(f.get("instruction", ""))
            f["feature_name"] = ""

    def _extract_classifications(self, raw: str) -> list[dict]:
        """Parse VLM classification response (object or bare array)."""
        # Try {"classifications": [...]} first
        match = re.search(r'\{\s*"classifications"\s*:\s*\[[\s\S]*?\]\s*\}', raw)
        if match:
            parsed = json.loads(self._repair_json(match.group()))
            return parsed.get("classifications", [])
        # Try bare JSON array: [{"index": 1, ...}, ...]
        arr_match = re.search(r'\[[\s\S]*\]', raw)
        if arr_match:
            parsed = json.loads(self._repair_json(arr_match.group()))
            if isinstance(parsed, list) and parsed and "index" in parsed[0]:
                return parsed
        # Last resort: any JSON object
        obj_match = re.search(r'\{[\s\S]*?\}', raw)
        if obj_match:
            parsed = json.loads(self._repair_json(obj_match.group()))
            return parsed.get("classifications", [])
        return []

    @staticmethod
    def _normalize_feature_type(raw: str) -> str:
        """Fix common VLM typos in feature_type values."""
        lower = raw.strip().lower()
        if lower.startswith("intrins"):
            return "intrinsic"
        if lower.startswith("context"):
            return "contextual"
        return raw

    def _fill_unclassified(self, features: list[dict]):
        """Apply keyword fallback to features the VLM did not classify."""
        for f in features:
            if not f.get("feature_type"):
                f["feature_type"] = self._fallback_classify(
                    f.get("instruction", ""),
                )

    _CONTEXTUAL_KEYWORDS = frozenset([
        "background", "environment", "lighting", "scene", "setting",
        "sky", "ground", "floor", "wall", "water", "grass", "snow",
        "forest", "field", "sand", "ocean", "sea", "landscape",
        "weather", "shadow", "reflection", "blur", "bokeh",
        "indoor", "outdoor", "habitat", "surrounding", "context",
    ])

    def _fallback_classify(self, instruction: str) -> str:
        """Keyword-based classification when VLM fails."""
        lower = instruction.lower()
        if any(kw in lower for kw in self._CONTEXTUAL_KEYWORDS):
            return "contextual"
        return "intrinsic"
