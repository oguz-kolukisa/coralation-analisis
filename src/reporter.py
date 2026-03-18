"""
Generates JSON + HTML/Markdown report from pipeline results.
"""
from __future__ import annotations
import copy
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from jinja2 import Template

logger = logging.getLogger(__name__)


def _get_feature_display_name(edit_result: dict) -> str:
    """
    Get the display name for a feature from VLM classification.
    Falls back to truncating the instruction if VLM didn't provide a name.
    """
    # Prefer VLM-provided feature_name
    if edit_result.get("feature_name"):
        return edit_result["feature_name"]
    # Fall back to instruction truncated
    instruction = edit_result.get("instruction", "")
    return instruction[:30] if instruction else "Unknown"


def _to_relative_path(abs_path: str, base_dir: Path) -> str:
    """Convert an absolute path to a relative path from base_dir."""
    try:
        # Resolve both paths to ensure consistent comparison
        path = Path(abs_path).resolve()
        base = base_dir.resolve()
        return str(path.relative_to(base))
    except ValueError:
        # If path is not relative to base_dir, return as-is
        return abs_path

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Classification Model Bias Analysis</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; max-width: 1800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 0; }
  h2 { color: #34495e; margin-top: 30px; background: #ecf0f1; padding: 10px; border-radius: 4px; }
  h3 { color: #555; margin-top: 20px; }
  h4 { color: #666; margin: 15px 0 10px 0; }

  /* Tab Navigation */
  .tab-nav { display: flex; gap: 0; margin: 20px 0 0 0; border-bottom: 2px solid #3498db; background: white; border-radius: 8px 8px 0 0; overflow: hidden; }
  .tab-btn { padding: 15px 30px; background: #ecf0f1; border: none; cursor: pointer; font-size: 1em; font-weight: 600; color: #555; transition: all 0.2s; border-right: 1px solid #ddd; }
  .tab-btn:last-child { border-right: none; }
  .tab-btn:hover { background: #d5dbdb; }
  .tab-btn.active { background: #3498db; color: white; }
  .tab-btn .badge { background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin-left: 8px; }
  .tab-btn.active .badge { background: white; color: #3498db; }
  .tab-content { display: none; background: white; padding: 25px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .tab-content.active { display: block; }

  /* Class selector sidebar for Classes tab */
  .classes-layout { display: grid; grid-template-columns: 220px 1fr; gap: 20px; }
  .class-nav { background: #f8f9fa; border-radius: 8px; padding: 15px; position: sticky; top: 20px; max-height: calc(100vh - 40px); overflow-y: auto; }
  .class-nav h4 { margin-top: 0; color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
  .class-nav-item { display: block; padding: 10px 12px; margin: 5px 0; background: white; border-radius: 4px; cursor: pointer; border-left: 3px solid transparent; transition: all 0.2s; }
  .class-nav-item:hover { background: #e8f4fd; border-left-color: #3498db; }
  .class-nav-item.active { background: #3498db; color: white; border-left-color: #2980b9; }
  .class-nav-item .risk { float: right; font-size: 0.75em; padding: 2px 6px; border-radius: 3px; }
  .class-nav-item .risk.high { background: #e74c3c; color: white; }
  .class-nav-item .risk.medium { background: #f39c12; color: white; }
  .class-nav-item .risk.low { background: #27ae60; color: white; }
  .class-detail { display: none; }
  .class-detail.active { display: block; }

  .class-section { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .summary-box { background: #3498db; color: white; border-radius: 8px; padding: 15px; margin: 10px 0; display: inline-block; }
  .confirmed { background: #27ae60; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }
  .not-confirmed { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }
  .priority-high { background: #9b59b6; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px; }
  .delta-pos { color: #27ae60; font-weight: bold; }
  .delta-neg { color: #e74c3c; font-weight: bold; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th { background: #2c3e50; color: white; padding: 10px; text-align: left; }
  td { padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
  tr:nth-child(even) { background: #f9f9f9; }
  tr:hover { background: #e8f4fd; }
  .img-grid { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; align-items: flex-start; }
  .img-cell { text-align: center; font-size: 0.8em; max-width: 180px; }
  .img-cell img { max-width: 160px; max-height: 160px; border-radius: 4px; border: 1px solid #ddd; }
  .img-cell.original img { border: 3px solid #3498db; }
  .img-cell.edited img { border: 2px solid #95a5a6; }
  .features { display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0; }
  .feature-tag { background: #3498db; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }
  .feature-tag.essential { background: #27ae60; }
  .feature-tag.spurious { background: #e74c3c; }
  .hypothesis-card { background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 0 4px 4px 0; }
  .hypothesis-card.confirmed-card { border-left-color: #27ae60; background: #f0fff4; }
  .hypothesis-card.shortcut-card { border-left-color: #e74c3c; background: #fff5f5; }
  .generation-row { display: flex; gap: 15px; margin-top: 10px; overflow-x: auto; padding: 10px 0; }
  .generation-item { text-align: center; min-width: 140px; }
  .generation-item img { width: 120px; height: 120px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }
  .generation-item .delta { font-weight: bold; font-size: 0.9em; }
  .stats-row { display: flex; flex-wrap: wrap; gap: 15px; margin: 10px 0; font-size: 0.9em; color: #666; }
  .stats-row span { background: #ecf0f1; padding: 3px 8px; border-radius: 4px; }
  .collapsible { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 8px; padding: 8px; border-radius: 4px; }
  .collapsible:hover { background: #e0e0e0; }
  .collapsible::before { content: '▶'; font-size: 0.8em; transition: transform 0.2s; }
  .collapsible.open::before { transform: rotate(90deg); }
  .collapse-content { display: none; padding-top: 10px; }
  .collapse-content.show { display: block; }
  /* Verdict strips */
  .verdict-strip { display: flex; align-items: center; gap: 12px; padding: 10px 15px; border-radius: 6px; margin: 8px 0; }
  .verdict-strip .verdict-badge { font-weight: 700; font-size: 0.95em; padding: 4px 12px; border-radius: 4px; color: white; white-space: nowrap; }
  .verdict-strip .confidence-change-bar { flex: 1; height: 10px; border-radius: 5px; background: #ecf0f1; position: relative; max-width: 200px; }
  .verdict-strip .confidence-change-bar .bar-fill { height: 100%; border-radius: 5px; }
  .verdict-shortcut { background: #fff5f5; border: 1px solid #e74c3c; }
  .verdict-shortcut .verdict-badge { background: #e74c3c; }
  .verdict-essential { background: #f0fff4; border: 1px solid #27ae60; }
  .verdict-essential .verdict-badge { background: #27ae60; }
  .verdict-robust { background: #f0f8ff; border: 1px solid #3498db; }
  .verdict-robust .verdict-badge { background: #3498db; }
  .verdict-neutral { background: #f8f9fa; border: 1px solid #95a5a6; }
  .verdict-neutral .verdict-badge { background: #95a5a6; }
  /* Grad-CAM comparison grid */
  .gradcam-comparison { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0; max-width: 540px; }
  .gradcam-comparison .img-cell { text-align: center; font-size: 0.8em; }
  .gradcam-comparison .img-cell img { max-width: 160px; max-height: 160px; border-radius: 4px; border: 1px solid #ddd; }
  /* Impact visualization */
  .impact-bar { height: 8px; border-radius: 4px; margin-top: 4px; }
  .impact-bar.negative { background: linear-gradient(to left, #e74c3c, #ffcccc); }
  .impact-bar.positive { background: linear-gradient(to right, #ccffcc, #27ae60); }
  .shortcut-warning { background: #fff3cd; border: 1px solid #ffc107; padding: 8px; border-radius: 4px; margin: 5px 0; }
  .expected-behavior { background: #d4edda; border: 1px solid #28a745; padding: 8px; border-radius: 4px; margin: 5px 0; }
  /* Feature summary table */
  .feature-summary { border-left: 4px solid #3498db; }
  .feature-intrinsic { color: #27ae60; }
  .feature-contextual { color: #e74c3c; font-weight: bold; }
  /* Stats cards */
  .stats-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
  .stat-card { background: white; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .stat-card .value { font-size: 2.5em; font-weight: bold; color: #2c3e50; }
  .stat-card .label { color: #666; font-size: 0.9em; margin-top: 5px; }
  .stat-card.danger .value { color: #e74c3c; }
  .stat-card.success .value { color: #27ae60; }
  .stat-card.warning .value { color: #f39c12; }
  /* Info boxes */
  .info-box { background: #e8f4fd; border: 1px solid #3498db; padding: 15px; border-radius: 8px; margin: 15px 0; }
  .info-box.warning { background: #fff3cd; border-color: #ffc107; }
  .info-box.success { background: #d4edda; border-color: #28a745; }
  .info-box.danger { background: #f8d7da; border-color: #dc3545; }
</style>
<script>
function showTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
  document.getElementById(tabId).classList.add('active');
  // Load images in the newly shown tab
  loadImages(tabId);
}

function showClass(className) {
  document.querySelectorAll('.class-nav-item').forEach(item => item.classList.remove('active'));
  document.querySelectorAll('.class-detail').forEach(detail => detail.classList.remove('active'));
  document.querySelector('[data-class="' + className + '"]').classList.add('active');
  document.getElementById('class-' + className).classList.add('active');
  // Load images for this class
  loadImages('class-' + className);
}

function toggleCollapse(id, btn) {
  var content = document.getElementById(id);
  content.classList.toggle('show');
  if (btn) btn.classList.toggle('open');
  // Load images when expanding
  if (content.classList.contains('show')) {
    loadImages(id);
  }
}

function loadImages(sectionId) {
  var section = document.getElementById(sectionId);
  if (!section) return;
  var imgs = section.querySelectorAll('img[data-src]');
  imgs.forEach(function(img) {
    if (img.dataset.src) {
      img.src = img.dataset.src;
      img.removeAttribute('data-src');
    }
  });
}

document.addEventListener('DOMContentLoaded', function() {
  // Show first tab and first class by default
  showTab('tab-overview');
});
</script>
</head>
<body>
<h1>Classification Model Bias Analysis Report</h1>

<!-- Tab Navigation -->
<div class="tab-nav">
  <button class="tab-btn active" data-tab="tab-overview" onclick="showTab('tab-overview')">
    Overview
  </button>
  <button class="tab-btn" data-tab="tab-features" onclick="showTab('tab-features')">
    Feature Analysis
    {% set total_shortcuts = namespace(count=0) %}
    {% for r in results %}
      {% for e in r.edit_results if e.confirmed %}
        {# Use VLM feature_type instead of keyword matching #}
        {% set is_contextual = e.feature_type == 'contextual' %}
        {# Only count as shortcut if contextual AND confidence dropped #}
        {% if is_contextual and e.mean_delta < -0.05 %}
          {% set total_shortcuts.count = total_shortcuts.count + 1 %}
        {% endif %}
      {% endfor %}
    {% endfor %}
    {% if total_shortcuts.count > 0 %}<span class="badge">{{ total_shortcuts.count }} shortcuts</span>{% endif %}
  </button>
  <button class="tab-btn" data-tab="tab-classes" onclick="showTab('tab-classes')">
    Class Details
  </button>
  <button class="tab-btn" data-tab="tab-methodology" onclick="showTab('tab-methodology')">
    Methodology
  </button>
</div>

<!-- ==================== OVERVIEW TAB ==================== -->
<div id="tab-overview" class="tab-content active">
  <h2>Analysis Summary</h2>

  <!-- Stats Cards -->
  {% set total_confirmed = namespace(count=0) %}
  {% set total_edits = namespace(count=0) %}
  {% set high_risk = namespace(count=0) %}
  {% for r in results %}
    {% set total_confirmed.count = total_confirmed.count + r.confirmed_hypotheses|length %}
    {% set total_edits.count = total_edits.count + r.edit_results|length %}
    {% if r.summary.confirmation_rate > 0.3 %}
      {% set high_risk.count = high_risk.count + 1 %}
    {% endif %}
  {% endfor %}

  <div class="stats-cards">
    <div class="stat-card">
      <div class="value">{{ results|length }}</div>
      <div class="label">Classes Analyzed</div>
    </div>
    <div class="stat-card">
      <div class="value">{{ total_edits.count }}</div>
      <div class="label">Feature Edits Tested</div>
    </div>
    <div class="stat-card danger">
      <div class="value">{{ total_confirmed.count }}</div>
      <div class="label">Confirmed Biases</div>
    </div>
    <div class="stat-card {{ 'danger' if high_risk.count > 0 else 'success' }}">
      <div class="value">{{ high_risk.count }}</div>
      <div class="label">High Risk Classes</div>
    </div>
  </div>

  <h3>Class Summary</h3>
  <table>
    <tr>
      <th>Class</th>
      <th>Total Edits</th>
      <th>Generations</th>
      <th>Confirmed</th>
      <th>Rate</th>
      <th>Risk</th>
      <th>Key Features</th>
    </tr>
    {% for r in results %}
    <tr onclick="showTab('tab-classes'); setTimeout(function(){ showClass('{{ r.class_name|replace(' ', '_') }}'); }, 100);" style="cursor: pointer;">
      <td><strong>{{ r.class_name }}</strong></td>
      <td>{{ r.summary.total_edits }}</td>
      <td>{{ r.summary.total_generations }}</td>
      <td>{{ r.summary.confirmed_count }}</td>
      <td>{{ "%.0f"|format(r.summary.confirmation_rate * 100) }}%</td>
      <td>
        {% set rate = r.summary.confirmation_rate %}
        {% if rate > (config.risk_high_threshold if config else 0.3) %}
          <span style="background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px;">HIGH</span>
        {% elif rate > (config.risk_medium_threshold if config else 0.1) %}
          <span style="background: #f39c12; color: white; padding: 3px 8px; border-radius: 4px;">MEDIUM</span>
        {% else %}
          <span style="background: #27ae60; color: white; padding: 3px 8px; border-radius: 4px;">LOW</span>
        {% endif %}
      </td>
      <td>{{ r.key_features[:3]|join(", ") }}</td>
    </tr>
    {% endfor %}
  </table>

  <div class="info-box">
    <strong>How to read this report:</strong>
    <ul style="margin: 10px 0 0 0;">
      <li><strong>Feature Analysis tab</strong> - See which features affect each class and identify shortcuts</li>
      <li><strong>Class Details tab</strong> - Explore individual classes with images and detailed edit results</li>
      <li><strong>Methodology tab</strong> - Understand the analysis pipeline and configuration</li>
    </ul>
  </div>
</div>

<!-- ==================== FEATURE ANALYSIS TAB ==================== -->
<div id="tab-features" class="tab-content">
  <h2>Feature Impact Summary</h2>
  <p>Quick overview of which features affect each class. <span class="feature-intrinsic">Green = intrinsic (expected)</span>, <span class="feature-contextual">Red = contextual (shortcut)</span>.</p>

  <div class="info-box warning">
    <strong>Understanding Feature Types and Biases:</strong>
    <ul style="margin: 10px 0 0 0;">
      <li><strong>Intrinsic Features</strong> - Parts of the object itself (ears, eyes, shape). Model SHOULD use these.</li>
      <li><strong>Contextual Features</strong> - Background, environment, lighting. If model relies on these, it's a <strong>shortcut/bias</strong>.</li>
      <li><strong>Spurious Correlations</strong> - When changing non-essential attributes (like color) <strong>increases</strong> confidence, the model learned a spurious correlation (e.g., "red = sports car").</li>
    </ul>
  </div>

  <table class="feature-summary">
  <tr>
    <th style="width: 15%;">Class</th>
    <th style="width: 30%;">Intrinsic Features (Expected)</th>
    <th style="width: 30%;">Contextual Features (Shortcuts)</th>
    <th style="width: 15%;">Impact</th>
    <th style="width: 10%;">Risk</th>
  </tr>
  {% for r in results %}
  <tr>
    <td><strong>{{ r.class_name }}</strong></td>
    <td>
      {# Collect intrinsic features with their impact - use VLM feature_type #}
      {% set ns = namespace(intrinsic=[]) %}
      {% for e in r.edit_results if e.confirmed %}
        {% set is_intrinsic = e.feature_type == 'intrinsic' %}
        {% if is_intrinsic %}
          {% set feat_name = e.feature_name if e.feature_name else e.instruction[:20] %}
          {% if feat_name not in ns.intrinsic %}
            {% set ns.intrinsic = ns.intrinsic + [feat_name] %}
          {% endif %}
        {% endif %}
      {% endfor %}
      {% if ns.intrinsic %}
        <span class="feature-intrinsic">{{ ns.intrinsic[:4]|unique|join(", ") }}{% if ns.intrinsic|length > 4 %} +{{ ns.intrinsic|length - 4 }}{% endif %}</span>
      {% else %}
        <em style="color: #999;">None confirmed</em>
      {% endif %}
    </td>
    <td>
      {# Collect shortcuts AND spurious correlations - use VLM feature_type #}
      {# Shortcuts: contextual features with NEGATIVE delta (model relied on them) #}
      {# Spurious correlations: modifications/replacements with POSITIVE delta (model learned wrong association) #}
      {% set ns = namespace(shortcuts=[], spurious=[], robust=[]) %}
      {% set spurious_threshold = config.spurious_positive_delta if config else 0.10 %}
      {% for e in r.edit_results if e.confirmed and not e.likely_failed %}
        {% set is_contextual = e.feature_type == 'contextual' %}
        {% set is_modification = e.edit_type == 'modification' or e.edit_type == 'replacement' %}
        {% set feat_name = e.feature_name if e.feature_name else e.instruction[:20] %}
        {% if is_contextual and e.mean_delta < -0.05 %}
          {# Contextual feature removal decreased confidence = shortcut #}
          {% if feat_name not in ns.shortcuts %}
            {% set ns.shortcuts = ns.shortcuts + [feat_name] %}
          {% endif %}
        {% elif is_modification and e.mean_delta > spurious_threshold and not is_contextual %}
          {# Non-contextual modification INCREASED confidence = spurious correlation #}
          {% if feat_name not in ns.spurious %}
            {% set ns.spurious = ns.spurious + [feat_name + ' (+' + "%.0f"|format(e.mean_delta * 100) + '%)'] %}
          {% endif %}
        {% elif is_contextual and e.mean_delta > 0.05 %}
          {% if feat_name not in ns.robust %}
            {% set ns.robust = ns.robust + [feat_name] %}
          {% endif %}
        {% endif %}
      {% endfor %}
      {% if ns.shortcuts or ns.spurious %}
        {% if ns.shortcuts %}
          <span class="feature-contextual">🚨 {{ ns.shortcuts[:2]|unique|join(", ") }}</span>
        {% endif %}
        {% if ns.spurious %}
          <span style="color: #e67e22; font-weight: bold;">⚠ Spurious: {{ ns.spurious[:2]|unique|join(", ") }}</span>
        {% endif %}
      {% elif ns.robust %}
        <span style="color: #3498db;">✓ Robust: {{ ns.robust[:2]|unique|join(", ") }}</span>
      {% elif r.spurious_features %}
        <span class="feature-contextual">{{ r.spurious_features[:3]|join(", ") }}</span>
      {% else %}
        <em style="color: #27ae60;">✓ None found</em>
      {% endif %}
    </td>
    <td>
      {# Show top impact #}
      {% if r.edit_results %}
        {% set sorted_edits = r.edit_results|sort(attribute='mean_delta') %}
        {% if sorted_edits %}
          {% set top = sorted_edits[0] %}
          <span class="delta-neg">{{ "%+.2f"|format(top.mean_delta) }}</span>
          <small>({{ top.simplified_feature if top.simplified_feature else top.instruction[:15] }})</small>
        {% endif %}
      {% else %}
        —
      {% endif %}
    </td>
    <td>
      {% set rate = r.summary.confirmation_rate %}
      {% if rate > (config.risk_high_threshold if config else 0.3) %}
        <span style="background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px;">HIGH</span>
      {% elif rate > (config.risk_medium_threshold if config else 0.1) %}
        <span style="background: #f39c12; color: white; padding: 3px 8px; border-radius: 4px;">MEDIUM</span>
      {% else %}
        <span style="background: #27ae60; color: white; padding: 3px 8px; border-radius: 4px;">LOW</span>
      {% endif %}
    </td>
  </tr>
  {% endfor %}
  </table>

  <h2>Feature Analysis with Evidence</h2>

  <div class="info-box">
    <strong>Understanding the results:</strong>
    <ul style="margin: 10px 0 0 0;">
      <li><strong>🚨 SHORTCUT</strong> (contextual + confidence DROP): Model wrongly relies on this feature - <span style="color: #e74c3c;">BAD</span></li>
      <li><strong>⚠ SPURIOUS</strong> (modification + confidence INCREASE): Model learned wrong correlation (e.g., "red = sports car") - <span style="color: #e67e22;">BAD</span></li>
      <li><strong>✓ ROBUST</strong> (contextual + confidence INCREASE): Model ignores irrelevant context - <span style="color: #27ae60;">GOOD</span></li>
      <li><strong>✓ ESSENTIAL</strong> (intrinsic + confidence DROP): Model correctly uses this feature - <span style="color: #27ae60;">GOOD</span></li>
      <li><strong>⚠ UNEXPECTED</strong> (intrinsic feature removal + confidence INCREASE): Removing object parts helped? - <span style="color: #f39c12;">INVESTIGATE</span></li>
    </ul>
    <p style="margin-top: 10px;"><em>Feature types are classified by VLM semantic analysis, not keyword matching.</em></p>
  </div>

  {% for r in results %}
  {# Use VLM feature_type for classification #}
  {% set has_shortcuts = namespace(found=false) %}
  {% set has_spurious = namespace(found=false) %}
  {% set has_robust = namespace(found=false) %}
  {% set has_essential = namespace(found=false) %}
  {% set has_unexpected = namespace(found=false) %}
  {% set spurious_threshold = config.spurious_positive_delta if config else 0.10 %}
  {% for e in r.edit_results if e.confirmed and not e.likely_failed %}
    {# Use VLM classification - feature_type is set by VLM #}
    {% set is_contextual = e.feature_type == 'contextual' %}
    {% set is_intrinsic = e.feature_type == 'intrinsic' %}
    {% set is_modification = e.edit_type == 'modification' or e.edit_type == 'replacement' %}
    {% set is_removal = e.edit_type == 'feature_removal' %}
    {% if is_contextual and e.mean_delta < -0.05 %}
      {% set has_shortcuts.found = true %}
    {% elif is_modification and e.mean_delta > spurious_threshold and not is_contextual %}
      {# Modification (like color change) that increases confidence = spurious correlation #}
      {% set has_spurious.found = true %}
    {% elif is_contextual and e.mean_delta > 0.05 %}
      {% set has_robust.found = true %}
    {% elif is_intrinsic and e.mean_delta < -0.05 %}
      {% set has_essential.found = true %}
    {% elif is_intrinsic and is_removal and e.mean_delta > 0.05 %}
      {% set has_unexpected.found = true %}
    {% endif %}
  {% endfor %}

  {# === SHORTCUTS (BAD) === #}
  {% if has_shortcuts.found %}
  <div class="class-section" style="margin-top: 20px;">
    <h3 style="color: #e74c3c;">🚨 {{ r.class_name }} - Contextual Shortcuts (Model Bias)</h3>
    <p style="color: #666;">These are features the model should NOT rely on, but removing them dropped confidence.</p>

    {% for e in r.edit_results if e.confirmed %}
    {% set is_contextual = e.feature_type == 'contextual' %}
    {% if is_contextual and e.mean_delta < -0.05 %}
    <div class="hypothesis-card shortcut-card">
      {# Verdict strip #}
      <div class="verdict-strip verdict-shortcut">
        <span class="verdict-badge">Shortcut: model relies on {{ e.feature_name if e.feature_name else 'this feature' }}</span>
        <span style="font-size: 0.9em; color: #666;">{{ "%.0f"|format(e.original_confidence * 100) }}% → {{ "%.0f"|format((e.original_confidence + e.mean_delta) * 100) }}%</span>
        <div class="confidence-change-bar">
          <div class="bar-fill" style="width: {{ [e.mean_delta|abs * 200, 100]|min }}%; background: #e74c3c;"></div>
        </div>
        <span class="delta-neg" style="font-size: 1.1em;">{{ "%+.0f"|format(e.mean_delta * 100) }}%</span>
      </div>

      <p style="margin: 8px 0 5px 0; color: #666;"><em>Edit: {{ e.instruction }}</em></p>
      <p style="margin: 5px 0; color: #888; font-size: 0.9em;">{{ e.hypothesis }}</p>

      <div class="generation-row" style="margin-top: 10px;">
        <div class="generation-item">
          <img src="{{ e.original_image_path }}" alt="original" style="border: 3px solid #3498db;">
          <div><strong>Original</strong></div>
          <div style="color: #27ae60;">{{ "%.0f"|format(e.original_confidence * 100) }}%</div>
        </div>
        {% for g in e.generations[:3] %}
        <div class="generation-item">
          <img src="{{ g.edited_image_path }}" alt="edited">
          <div>Edited</div>
          <div class="delta-neg">{{ "%.0f"|format(g.edited_confidence * 100) }}% ({{ "%+.0f"|format(g.delta * 100) }}%)</div>
        </div>
        {% endfor %}
      </div>

      {# Grad-CAM comparison row #}
      {% if e.original_gradcam_path and e.generations and e.generations[0].gradcam_image_path %}
      <h5 style="margin: 15px 0 5px 0; color: #555;">Attention Shift</h5>
      <div class="gradcam-comparison">
        <div class="img-cell">
          <img src="{{ e.original_gradcam_path }}" alt="original attention">
          <div>Original Attention</div>
        </div>
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_image_path }}" alt="edited attention">
          <div>Edited Attention</div>
        </div>
        {% if e.generations[0].gradcam_diff_path %}
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_diff_path }}" alt="attention diff">
          <div>Attention Shift<br><small style="color: #999;">blue=lost, red=gained</small></div>
        </div>
        {% endif %}
      </div>
      {% endif %}

      {# Collapsible statistical details #}
      {% if e.p_value is defined and e.p_value < 1.0 %}
      <h5 class="collapsible" onclick="toggleCollapse('stats-shortcut-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}', this)" style="margin: 10px 0 0 0; font-size: 0.85em; color: #888;">
        Statistical Details
      </h5>
      <div id="stats-shortcut-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}" class="collapse-content">
        <div class="stats-row">
          <span>p-value: {{ "%.4f"|format(e.p_value) }}</span>
          <span>Cohen's d: {{ "%.2f"|format(e.cohens_d) }} ({{ e.effect_size }})</span>
          <span>Mean Δ: {{ "%+.3f"|format(e.mean_delta) }}±{{ "%.3f"|format(e.std_delta|default(0)) }}</span>
          {% if e.statistically_significant %}<span style="background: #27ae60; color: white;">Statistically Significant</span>{% endif %}
        </div>
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {# === SPURIOUS CORRELATIONS (BAD) === #}
  {% if has_spurious.found %}
  <div class="class-section" style="margin-top: 20px;">
    <h3 style="color: #e67e22;">⚠ {{ r.class_name }} - Spurious Correlations (Learned Wrong Associations)</h3>
    <p style="color: #666;">Modifying these features <strong>increased</strong> confidence, suggesting the model learned spurious correlations.</p>

    {% for e in r.edit_results if e.confirmed and not e.likely_failed %}
    {% set is_contextual = e.feature_type == 'contextual' %}
    {% set is_modification = e.edit_type == 'modification' or e.edit_type == 'replacement' %}
    {% if is_modification and e.mean_delta > spurious_threshold and not is_contextual %}
    <div class="hypothesis-card" style="border-left-color: #e67e22; background: #fff8f0;">
      <div class="verdict-strip" style="background: #fff8f0; border: 1px solid #e67e22;">
        <span class="verdict-badge" style="background: #e67e22;">Spurious correlation: {{ e.feature_name if e.feature_name else 'this feature' }}</span>
        <span style="font-size: 0.9em; color: #666;">{{ "%.0f"|format(e.original_confidence * 100) }}% → {{ "%.0f"|format((e.original_confidence + e.mean_delta) * 100) }}%</span>
        <div class="confidence-change-bar">
          <div class="bar-fill" style="width: {{ [e.mean_delta|abs * 200, 100]|min }}%; background: #e67e22;"></div>
        </div>
        <span class="delta-pos" style="font-size: 1.1em;">{{ "%+.0f"|format(e.mean_delta * 100) }}%</span>
      </div>

      <p style="margin: 8px 0 5px 0; color: #666;"><em>Edit: {{ e.instruction }}</em></p>
      <p style="margin: 5px 0; color: #888; font-size: 0.9em;">{{ e.hypothesis }}</p>

      <div class="generation-row" style="margin-top: 10px;">
        <div class="generation-item">
          <img src="{{ e.original_image_path }}" alt="original" style="border: 3px solid #3498db;">
          <div><strong>Original</strong></div>
          <div>{{ "%.0f"|format(e.original_confidence * 100) }}%</div>
        </div>
        {% for g in e.generations[:3] %}
        <div class="generation-item">
          <img src="{{ g.edited_image_path }}" alt="edited">
          <div>Edited</div>
          <div class="delta-pos">{{ "%.0f"|format(g.edited_confidence * 100) }}% ({{ "%+.0f"|format(g.delta * 100) }}%)</div>
        </div>
        {% endfor %}
      </div>

      {% if e.original_gradcam_path and e.generations and e.generations[0].gradcam_image_path %}
      <h5 style="margin: 15px 0 5px 0; color: #555;">Attention Shift</h5>
      <div class="gradcam-comparison">
        <div class="img-cell">
          <img src="{{ e.original_gradcam_path }}" alt="original attention">
          <div>Original Attention</div>
        </div>
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_image_path }}" alt="edited attention">
          <div>Edited Attention</div>
        </div>
        {% if e.generations[0].gradcam_diff_path %}
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_diff_path }}" alt="attention diff">
          <div>Attention Shift<br><small style="color: #999;">blue=lost, red=gained</small></div>
        </div>
        {% endif %}
      </div>
      {% endif %}

      {% if e.p_value is defined and e.p_value < 1.0 %}
      <h5 class="collapsible" onclick="toggleCollapse('stats-spurious-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}', this)" style="margin: 10px 0 0 0; font-size: 0.85em; color: #888;">
        Statistical Details
      </h5>
      <div id="stats-spurious-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}" class="collapse-content">
        <div class="stats-row">
          <span>p-value: {{ "%.4f"|format(e.p_value) }}</span>
          <span>Cohen's d: {{ "%.2f"|format(e.cohens_d) }} ({{ e.effect_size }})</span>
          <span>Mean Δ: {{ "%+.3f"|format(e.mean_delta) }}±{{ "%.3f"|format(e.std_delta|default(0)) }}</span>
        </div>
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {# === ESSENTIAL FEATURES (GOOD) === #}
  {% if has_essential.found %}
  <div class="class-section" style="margin-top: 20px;">
    <h3 style="color: #27ae60;">✓ {{ r.class_name }} - Essential Features (Correct Behavior)</h3>
    <p style="color: #666;">The model correctly relies on these intrinsic features. Removing them dropped confidence as expected.</p>

    {% for e in r.edit_results if e.confirmed and not e.likely_failed %}
    {% set is_intrinsic = e.feature_type == 'intrinsic' %}
    {% if is_intrinsic and e.mean_delta < -0.05 %}
    <div class="hypothesis-card confirmed-card">
      <div class="verdict-strip verdict-essential">
        <span class="verdict-badge">Important feature for classification</span>
        <span style="font-size: 0.9em; color: #666;">{{ "%.0f"|format(e.original_confidence * 100) }}% → {{ "%.0f"|format((e.original_confidence + e.mean_delta) * 100) }}%</span>
        <div class="confidence-change-bar">
          <div class="bar-fill" style="width: {{ [e.mean_delta|abs * 200, 100]|min }}%; background: #27ae60;"></div>
        </div>
        <span class="delta-neg" style="font-size: 1.1em;">{{ "%+.0f"|format(e.mean_delta * 100) }}%</span>
      </div>

      <p style="margin: 8px 0 5px 0; color: #666;"><em>Edit: {{ e.instruction }}</em></p>
      <p style="margin: 5px 0; color: #888; font-size: 0.9em;">{{ e.hypothesis }}</p>

      <div class="generation-row" style="margin-top: 10px;">
        <div class="generation-item">
          <img src="{{ e.original_image_path }}" alt="original" style="border: 3px solid #3498db;">
          <div><strong>Original</strong></div>
          <div style="color: #27ae60;">{{ "%.0f"|format(e.original_confidence * 100) }}%</div>
        </div>
        {% for g in e.generations[:3] %}
        <div class="generation-item">
          <img src="{{ g.edited_image_path }}" alt="edited">
          <div>Edited</div>
          <div class="delta-neg">{{ "%.0f"|format(g.edited_confidence * 100) }}% ({{ "%+.0f"|format(g.delta * 100) }}%)</div>
        </div>
        {% endfor %}
      </div>

      {% if e.original_gradcam_path and e.generations and e.generations[0].gradcam_image_path %}
      <h5 style="margin: 15px 0 5px 0; color: #555;">Attention Shift</h5>
      <div class="gradcam-comparison">
        <div class="img-cell">
          <img src="{{ e.original_gradcam_path }}" alt="original attention">
          <div>Original Attention</div>
        </div>
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_image_path }}" alt="edited attention">
          <div>Edited Attention</div>
        </div>
        {% if e.generations[0].gradcam_diff_path %}
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_diff_path }}" alt="attention diff">
          <div>Attention Shift<br><small style="color: #999;">blue=lost, red=gained</small></div>
        </div>
        {% endif %}
      </div>
      {% endif %}

      {% if e.p_value is defined and e.p_value < 1.0 %}
      <h5 class="collapsible" onclick="toggleCollapse('stats-essential-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}', this)" style="margin: 10px 0 0 0; font-size: 0.85em; color: #888;">
        Statistical Details
      </h5>
      <div id="stats-essential-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}" class="collapse-content">
        <div class="stats-row">
          <span>p-value: {{ "%.4f"|format(e.p_value) }}</span>
          <span>Cohen's d: {{ "%.2f"|format(e.cohens_d) }} ({{ e.effect_size }})</span>
          <span>Mean Δ: {{ "%+.3f"|format(e.mean_delta) }}±{{ "%.3f"|format(e.std_delta|default(0)) }}</span>
        </div>
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {# === ROBUST TO CONTEXT (GOOD) === #}
  {% if has_robust.found %}
  <div class="class-section" style="margin-top: 20px;">
    <h3 style="color: #3498db;">✓ {{ r.class_name }} - Robust to Context (Good Behavior)</h3>
    <p style="color: #666;">Removing these contextual features <strong>increased</strong> confidence. The model is NOT relying on them as shortcuts.</p>

    {% for e in r.edit_results if e.confirmed %}
    {% set is_contextual = e.feature_type == 'contextual' %}
    {% if is_contextual and e.mean_delta > 0.05 %}
    <div class="hypothesis-card" style="border-left-color: #3498db; background: #f0f8ff;">
      <div class="verdict-strip verdict-robust">
        <span class="verdict-badge">Model robust to this context</span>
        <span style="font-size: 0.9em; color: #666;">{{ "%.0f"|format(e.original_confidence * 100) }}% → {{ "%.0f"|format((e.original_confidence + e.mean_delta) * 100) }}%</span>
        <div class="confidence-change-bar">
          <div class="bar-fill" style="width: {{ [e.mean_delta|abs * 200, 100]|min }}%; background: #3498db;"></div>
        </div>
        <span class="delta-pos" style="font-size: 1.1em;">{{ "%+.0f"|format(e.mean_delta * 100) }}%</span>
      </div>

      <p style="margin: 8px 0 5px 0; color: #666;"><em>Edit: {{ e.instruction }}</em></p>

      <div class="generation-row" style="margin-top: 10px;">
        <div class="generation-item">
          <img src="{{ e.original_image_path }}" alt="original" style="border: 3px solid #3498db;">
          <div><strong>Original</strong></div>
          <div>{{ "%.0f"|format(e.original_confidence * 100) }}%</div>
        </div>
        {% for g in e.generations[:3] %}
        <div class="generation-item">
          <img src="{{ g.edited_image_path }}" alt="edited">
          <div>Edited</div>
          <div class="delta-pos">{{ "%.0f"|format(g.edited_confidence * 100) }}% ({{ "%+.0f"|format(g.delta * 100) }}%)</div>
        </div>
        {% endfor %}
      </div>

      {% if e.original_gradcam_path and e.generations and e.generations[0].gradcam_image_path %}
      <h5 style="margin: 15px 0 5px 0; color: #555;">Attention Shift</h5>
      <div class="gradcam-comparison">
        <div class="img-cell">
          <img src="{{ e.original_gradcam_path }}" alt="original attention">
          <div>Original Attention</div>
        </div>
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_image_path }}" alt="edited attention">
          <div>Edited Attention</div>
        </div>
        {% if e.generations[0].gradcam_diff_path %}
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_diff_path }}" alt="attention diff">
          <div>Attention Shift<br><small style="color: #999;">blue=lost, red=gained</small></div>
        </div>
        {% endif %}
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {# === UNEXPECTED (INVESTIGATE) === #}
  {% if has_unexpected.found %}
  <div class="class-section" style="margin-top: 20px;">
    <h3 style="color: #f39c12;">⚠ {{ r.class_name }} - Unexpected Results (Investigate)</h3>
    <p style="color: #666;">Removing these intrinsic features <strong>increased</strong> confidence. This is unexpected - these features may be confusing the model.</p>

    {% for e in r.edit_results if e.confirmed %}
    {% set is_intrinsic = e.feature_type == 'intrinsic' %}
    {% if is_intrinsic and e.mean_delta > 0.05 %}
    <div class="hypothesis-card" style="border-left-color: #f39c12; background: #fffbf0;">
      <div class="verdict-strip verdict-neutral" style="border-color: #f39c12;">
        <span class="verdict-badge" style="background: #f39c12;">Unexpected: removing intrinsic feature helped</span>
        <span style="font-size: 0.9em; color: #666;">{{ "%.0f"|format(e.original_confidence * 100) }}% → {{ "%.0f"|format((e.original_confidence + e.mean_delta) * 100) }}%</span>
        <div class="confidence-change-bar">
          <div class="bar-fill" style="width: {{ [e.mean_delta|abs * 200, 100]|min }}%; background: #f39c12;"></div>
        </div>
        <span class="delta-pos" style="font-size: 1.1em;">{{ "%+.0f"|format(e.mean_delta * 100) }}%</span>
      </div>

      <p style="margin: 8px 0 5px 0; color: #666;"><em>Edit: {{ e.instruction }}</em></p>

      <div class="generation-row" style="margin-top: 10px;">
        <div class="generation-item">
          <img src="{{ e.original_image_path }}" alt="original" style="border: 3px solid #3498db;">
          <div><strong>Original</strong></div>
          <div>{{ "%.0f"|format(e.original_confidence * 100) }}%</div>
        </div>
        {% for g in e.generations[:3] %}
        <div class="generation-item">
          <img src="{{ g.edited_image_path }}" alt="edited">
          <div>Edited</div>
          <div class="delta-pos">{{ "%.0f"|format(g.edited_confidence * 100) }}% ({{ "%+.0f"|format(g.delta * 100) }}%)</div>
        </div>
        {% endfor %}
      </div>

      {% if e.original_gradcam_path and e.generations and e.generations[0].gradcam_image_path %}
      <h5 style="margin: 15px 0 5px 0; color: #555;">Attention Shift</h5>
      <div class="gradcam-comparison">
        <div class="img-cell">
          <img src="{{ e.original_gradcam_path }}" alt="original attention">
          <div>Original Attention</div>
        </div>
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_image_path }}" alt="edited attention">
          <div>Edited Attention</div>
        </div>
        {% if e.generations[0].gradcam_diff_path %}
        <div class="img-cell">
          <img src="{{ e.generations[0].gradcam_diff_path }}" alt="attention diff">
          <div>Attention Shift<br><small style="color: #999;">blue=lost, red=gained</small></div>
        </div>
        {% endif %}
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

</div>

<!-- ==================== CLASS DETAILS TAB ==================== -->
<div id="tab-classes" class="tab-content">
  <h2>Class-by-Class Analysis</h2>
  <p>Select a class from the sidebar to view detailed analysis, images, and edit results.</p>

  <div class="classes-layout">
    <!-- Class Navigation Sidebar -->
    <div class="class-nav">
      <h4>Classes ({{ results|length }})</h4>
      {% for r in results %}
      <div class="class-nav-item {{ 'active' if loop.first else '' }}" data-class="{{ r.class_name|replace(' ', '_') }}" onclick="showClass('{{ r.class_name|replace(' ', '_') }}')">
        {{ r.class_name }}
        {% set rate = r.summary.confirmation_rate %}
        {% set high_thresh = config.risk_high_threshold if config else 0.3 %}
        {% set med_thresh = config.risk_medium_threshold if config else 0.1 %}
        <span class="risk {{ 'high' if rate > high_thresh else ('medium' if rate > med_thresh else 'low') }}">
          {{ "%.0f"|format(rate * 100) }}%
        </span>
      </div>
      {% endfor %}
    </div>

    <!-- Class Details Content -->
    <div class="class-details-content">
      {% for r in results %}
      <div id="class-{{ r.class_name|replace(' ', '_') }}" class="class-detail {{ 'active' if loop.first else '' }}">
        <div class="class-section">
          <h2>{{ r.class_name }}</h2>

  {% if r.key_features %}
  <h4>Key Visual Features</h4>
  <div class="features">
    {% for f in r.key_features %}<span class="feature-tag">{{ f }}</span>{% endfor %}
  </div>
  {% endif %}

  {% if r.essential_features %}
  <h4>Essential Features (model SHOULD use)</h4>
  <div class="features">
    {% for f in r.essential_features %}<span class="feature-tag essential">{{ f }}</span>{% endfor %}
  </div>
  {% endif %}

  {% if r.spurious_features %}
  <h4>Spurious Features (potential shortcuts)</h4>
  <div class="features">
    {% for f in r.spurious_features %}<span class="feature-tag spurious">{{ f }}</span>{% endfor %}
  </div>
  {% endif %}

  {% if r.model_focus %}
  <p><strong>Model Attention (Grad-CAM):</strong> {{ r.model_focus }}</p>
  {% endif %}

  {% if r.vlm_insights %}
  <h4>VLM Insights (from iterative analysis)</h4>
  <ul>
    {% for insight in r.vlm_insights %}<li>{{ insight }}</li>{% endfor %}
  </ul>
  {% endif %}

  {% if r.confirmed_shortcuts %}
  <h4>VLM-Confirmed Shortcuts</h4>
  <div class="features">
    {% for s in r.confirmed_shortcuts %}<span class="feature-tag spurious">{{ s }}</span>{% endfor %}
  </div>
  {% endif %}

  {% if r.final_summary %}
  <div class="summary-box" style="background: {% if r.risk_level == 'HIGH' %}#e74c3c{% elif r.risk_level == 'MEDIUM' %}#f39c12{% else %}#27ae60{% endif %};">
    <strong>Risk Level: {{ r.risk_level }}</strong> | Robustness: {{ r.robustness_score }}/10
  </div>
  <p><strong>Summary:</strong> {{ r.final_summary }}</p>
  {% endif %}

  {% if r.vulnerabilities %}
  <h4>Identified Vulnerabilities</h4>
  <ul>
    {% for v in r.vulnerabilities %}<li>{{ v }}</li>{% endfor %}
  </ul>
  {% endif %}

  {% if r.recommendations %}
  <h4>Recommendations</h4>
  <ul>
    {% for rec in r.recommendations %}<li>{{ rec }}</li>{% endfor %}
  </ul>
  {% endif %}

  {% if r.detected_features %}
  <h4>Detected Features</h4>
  <p><em>Intrinsic = part of the object (expected to affect classification). Contextual = background/environment (if it affects classification, it's a shortcut).</em></p>
  <table>
    <tr>
      <th>Feature</th>
      <th>Category</th>
      <th>Type</th>
      <th>Model Attention</th>
    </tr>
    {% for f in r.detected_features[:10] %}
    <tr>
      <td>{{ f.name }}</td>
      <td>{{ f.category }}</td>
      <td>
        {% if f.feature_type == 'intrinsic' %}
          <span class="feature-tag essential">Intrinsic</span>
        {% else %}
          <span class="feature-tag spurious">Contextual</span>
        {% endif %}
      </td>
      <td>{{ f.gradcam_attention }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  <h3 class="collapsible" onclick="toggleCollapse('baseline-{{ loop.index }}', this); loadImages('baseline-{{ loop.index }}');">
    Baseline Samples ({{ r.baseline_results|length }})
  </h3>
  <div id="baseline-{{ loop.index }}" class="collapse-content">
    <div class="img-grid">
      {% for b in r.baseline_results[:8] %}
      <div class="img-cell">
        <img {% if loop.index <= 4 %}src="{{ b.image_path }}"{% else %}data-src="{{ b.image_path }}"{% endif %} alt="{{ b.true_label }}">
        <div>{{ b.true_label }}</div>
        <div>conf: <strong>{{ "%.3f"|format(b.class_confidence) }}</strong></div>
        <div><small>{{ b.type }}</small></div>
      </div>
      {% endfor %}
    </div>
  </div>

  <h3 class="collapsible open" onclick="toggleCollapse('confirmed-{{ loop.index }}', this); loadImages('confirmed-{{ loop.index }}');">
    Confirmed Shortcuts ({{ r.confirmed_hypotheses|length }})
  </h3>
  <div id="confirmed-{{ loop.index }}" class="collapse-content show">
  {% if r.confirmed_hypotheses %}
  {% for e in r.confirmed_hypotheses %}
  {# Use VLM feature_type instead of keyword matching #}
  {% set e_is_contextual = e.feature_type == 'contextual' %}
  {# TRUE shortcut = contextual AND negative delta (model relied on it) #}
  {% set e_is_shortcut = e_is_contextual and e.mean_delta < -0.05 %}
  {# ROBUST = contextual AND positive delta (model not relying on it) #}
  {% set e_is_robust = e_is_contextual and e.mean_delta > 0.05 %}
  <div class="hypothesis-card {{ 'shortcut-card' if e_is_shortcut else ('confirmed-card') }}">
    <strong>{{ e.instruction }}</strong>
    {% if e_is_shortcut %}
      <span style="background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;">🚨 SHORTCUT</span>
    {% elif e_is_robust %}
      <span style="background: #27ae60; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;">✓ ROBUST</span>
    {% endif %}
    {% if e.priority >= 4 %}<span class="priority-high">Priority {{ e.priority }}</span>{% endif %}
    <br>
    <em>{{ e.hypothesis }}</em>
    <div class="stats-row">
      <span>Mean Δ: <span class="{{ 'delta-pos' if e.mean_delta > 0 else 'delta-neg' }}">{{ "%+.3f"|format(e.mean_delta) }}±{{ "%.3f"|format(e.std_delta|default(0)) }}</span></span>
      <span>Range: {{ "%+.3f"|format(e.min_delta) }} to {{ "%+.3f"|format(e.max_delta) }}</span>
      <span>Confirmed: {{ e.confirmation_count }}/{{ e.generations|length }}</span>
      <span>Original: {{ "%.3f"|format(e.original_confidence) }}</span>
    </div>
    {% if e.p_value is defined and e.p_value < 1.0 %}
    <div class="stats-row">
      <span>p-value: {{ "%.4f"|format(e.p_value) }}{% if e.p_value < 0.05 %} ✓{% endif %}</span>
      <span>Cohen's d: {{ "%.2f"|format(e.cohens_d) }} ({{ e.effect_size }})</span>
      {% if e.statistically_significant %}<span style="background: #27ae60;">Stat. Significant</span>{% endif %}
      {% if e.practically_significant %}<span style="background: #3498db;">Pract. Significant</span>{% endif %}
    </div>
    {% endif %}
    <h4 class="collapsible" onclick="toggleCollapse('images-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}', this); loadImages('images-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}');">
      Original vs Generated Images
    </h4>
    <div id="images-{{ r.class_name|replace(' ', '_') }}-{{ loop.index0 }}" class="collapse-content">
      <div class="generation-row">
        <div class="generation-item">
          <img data-src="{{ e.original_image_path }}" alt="original">
          <div><strong>Original</strong></div>
          <div>{{ "%.3f"|format(e.original_confidence) }}</div>
        </div>
        {% for g in e.generations %}
        <div class="generation-item">
          <img data-src="{{ g.edited_image_path }}" alt="gen {{ loop.index }}">
          <div>Gen {{ loop.index }}</div>
          <div class="delta {{ 'delta-pos' if g.delta > 0 else 'delta-neg' }}">{{ "%+.3f"|format(g.delta) }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
  {% endfor %}
  {% else %}
  <p><em>No shortcuts confirmed for this class.</em></p>
  {% endif %}
  </div>

  <h3 class="collapsible" onclick="toggleCollapse('edits-{{ loop.index }}', this); loadImages('edits-{{ loop.index }}');">
    All Edit Results ({{ r.edit_results|length }}) - Click to expand
  </h3>
  <div id="edits-{{ loop.index }}" class="collapse-content">
  {% for e in r.edit_results %}
  {# Use VLM feature_type instead of keyword matching #}
  {% set all_is_ctx = e.feature_type == 'contextual' %}
  {% set all_is_modification = e.edit_type == 'modification' or e.edit_type == 'replacement' %}
  {% set spurious_thresh = config.spurious_positive_delta if config else 0.10 %}
  {# TRUE shortcut = contextual AND negative delta #}
  {% set all_is_shortcut = all_is_ctx and e.mean_delta < -0.05 %}
  {% set all_is_robust = all_is_ctx and e.mean_delta > 0.05 %}
  {# Spurious correlation = modification with positive delta on non-contextual feature #}
  {% set all_is_spurious = all_is_modification and e.mean_delta > spurious_thresh and not all_is_ctx %}
  <div class="hypothesis-card {{ 'shortcut-card' if (e.confirmed and all_is_shortcut) else ('confirmed-card' if e.confirmed else '') }}">
    <strong>{{ e.instruction }}</strong>
    {% if e.likely_failed %}
      <span style="background: #95a5a6; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em;">⚠ EDIT FAILED</span>
    {% elif e.confirmed %}
      {% if all_is_shortcut %}
        <span style="background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em;">🚨 SHORTCUT</span>
      {% elif all_is_spurious %}
        <span style="background: #e67e22; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em;">⚠ SPURIOUS</span>
      {% elif all_is_robust %}
        <span style="background: #27ae60; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em;">✓ ROBUST</span>
      {% else %}
        <span class="confirmed">CONFIRMED</span>
      {% endif %}
    {% else %}
      <span class="not-confirmed">not confirmed</span>
    {% endif %}
    <br>
    <em>{{ e.hypothesis }}</em>
    <div class="stats-row">
      <span>Mean Δ: <span class="{{ 'delta-pos' if e.mean_delta > 0 else 'delta-neg' }}">{{ "%+.3f"|format(e.mean_delta) }}±{{ "%.3f"|format(e.std_delta|default(0)) }}</span></span>
      <span>Range: {{ "%+.3f"|format(e.min_delta) }} to {{ "%+.3f"|format(e.max_delta) }}</span>
      <span>{{ e.confirmation_count }}/{{ e.generations|length }} confirmed</span>
      {% if e.p_value is defined and e.p_value < 1.0 %}
      <span>p={{ "%.3f"|format(e.p_value) }}</span>
      <span>d={{ "%.2f"|format(e.cohens_d) }}</span>
      {% endif %}
    </div>
    <div class="generation-row">
      <div class="generation-item">
        <img data-src="{{ e.original_image_path }}" alt="original">
        <div><strong>Original</strong></div>
      </div>
      {% for g in e.generations %}
      <div class="generation-item">
        <img data-src="{{ g.edited_image_path }}" alt="gen {{ loop.index }}">
        <div class="delta {{ 'delta-pos' if g.delta > 0 else 'delta-neg' }}">{{ "%+.3f"|format(g.delta) }}</div>
      </div>
      {% endfor %}
    </div>
    {# Grad-CAM comparison for all edits #}
    {% if e.original_gradcam_path and e.generations and e.generations[0].gradcam_image_path %}
    <div class="gradcam-comparison" style="margin-top: 8px;">
      <div class="img-cell">
        <img data-src="{{ e.original_gradcam_path }}" alt="original attention">
        <div>Original Attention</div>
      </div>
      <div class="img-cell">
        <img data-src="{{ e.generations[0].gradcam_image_path }}" alt="edited attention">
        <div>Edited Attention</div>
      </div>
      {% if e.generations[0].gradcam_diff_path %}
      <div class="img-cell">
        <img data-src="{{ e.generations[0].gradcam_diff_path }}" alt="attention diff">
        <div>Attention Shift</div>
      </div>
      {% endif %}
    </div>
    {% endif %}
  </div>
  {% endfor %}
  </div>
      </div>
    </div>
      {% endfor %}
    </div>
  </div>
</div>

<!-- Move Feature Dependency Rubric to Features Tab - append after the summary table -->
<!-- This section now appears in the Features tab above -->

<!-- ==================== METHODOLOGY TAB ==================== -->
<div id="tab-methodology" class="tab-content">
  <h2>Analysis Methodology</h2>

  <div class="info-box">
    <h3 style="margin-top: 0;">Pipeline Overview</h3>
    <p>This analysis uses an automated pipeline to discover biases and shortcuts in image classification models.</p>
  </div>

  <h3>Configuration</h3>
  <table>
    <tr><th style="width: 30%;">Parameter</th><th>Value</th></tr>
    <tr><td><strong>Classifier Model</strong></td><td>{{ config.classifier_model if config else 'ResNet-50 (ImageNet-1k)' }}</td></tr>
    <tr><td><strong>Vision-Language Model (VLM)</strong></td><td>{{ config.vlm_model if config else 'Qwen/Qwen2.5-VL-7B-Instruct' }}</td></tr>
    <tr><td><strong>Image Editor Model</strong></td><td>{{ config.editor_model if config else 'Qwen/Qwen-Image-Edit' }}</td></tr>
    <tr><td><strong>Attention Method</strong></td><td>{{ config.attention_method if config else 'Score-CAM' }}</td></tr>
    <tr><td><strong>Samples per Class</strong></td><td>{{ config.samples_per_class if config else '5' }} positive, {{ config.negative_samples if config else '5' }} negative</td></tr>
    <tr><td><strong>VLM Iterations</strong></td><td>{{ config.iterations if config else '2' }}</td></tr>
    <tr><td><strong>Generations per Edit</strong></td><td>{{ config.generations_per_edit if config else '3' }}</td></tr>
    <tr><td><strong>Confidence Delta Threshold</strong></td><td>{{ config.confidence_delta_threshold if config else '0.15' }}</td></tr>
    <tr><td><strong>Statistical Validation</strong></td><td>{{ "Enabled (t-test + Cohen's d)" if (config and config.use_statistical_validation) else "Threshold-based" }}</td></tr>
    <tr><td><strong>Edit Grad-CAM</strong></td><td>{{ "Enabled (attention diff on edited images)" if (config and config.compute_edit_gradcam) else "Disabled" }}</td></tr>
    <tr><td><strong>Edit Verification</strong></td><td>{{ "Enabled (VLM verifies edits)" if (config and config.verify_edits) else "Disabled" }}</td></tr>
    <tr><td><strong>Pipeline Mode</strong></td><td>Phase-first (6 model swaps total)</td></tr>
  </table>

  <h3>Pipeline Steps</h3>
  <div style="display: grid; gap: 15px; margin: 20px 0;">
    <div class="info-box" style="border-left: 4px solid #1abc9c;">
      <h4 style="margin: 0 0 10px 0;">1. Knowledge Discovery</h4>
      <p style="margin: 0;">VLM uses world knowledge to identify potential shortcuts for the target class (e.g., for "cat" it suggests "yarn ball", "milk bowl" as commonly associated features). No images needed.</p>
    </div>
    <div class="info-box" style="border-left: 4px solid #3498db;">
      <h4 style="margin: 0 0 10px 0;">2. Sample Collection</h4>
      <p style="margin: 0;">Collect positive samples and negative samples from confusing classes (identified from classifier top-k predictions) from ImageNet validation set.</p>
    </div>
    <div class="info-box" style="border-left: 4px solid #9b59b6;">
      <h4 style="margin: 0 0 10px 0;">3. Baseline Classification</h4>
      <p style="margin: 0;">Classify all samples with attention maps ({{ config.attention_method if config else 'Score-CAM' }}) to establish baseline confidence and visualize which image regions the classifier focuses on.</p>
    </div>
    <div class="info-box" style="border-left: 4px solid #e67e22;">
      <h4 style="margin: 0 0 10px 0;">4. Image-Based Feature Discovery</h4>
      <p style="margin: 0;">The VLM ({{ config.vlm_model if config else 'Qwen/Qwen2.5-VL-7B-Instruct' }}) analyzes each image + attention map to identify visual features. It classifies features as <strong>intrinsic</strong> (object parts) or <strong>contextual</strong> (background).</p>
    </div>
    <div class="info-box" style="border-left: 4px solid #27ae60;">
      <h4 style="margin: 0 0 10px 0;">5. Counterfactual Editing</h4>
      <p style="margin: 0;">The VLM generates specific edit instructions, then the image editor ({{ config.editor_model if config else 'Qwen/Qwen-Image-Edit' }}) applies each edit. Multiple generations per edit ensure robustness. Model lifecycle managed by ModelManager for VRAM efficiency.</p>
    </div>
    <div class="info-box" style="border-left: 4px solid #e74c3c;">
      <h4 style="margin: 0 0 10px 0;">6. Impact Measurement + Attention Diff</h4>
      <p style="margin: 0;">Re-classify edited images and measure confidence change (delta). When enabled, Grad-CAM is also computed on edited images to produce attention diff heatmaps showing where the model gained (red) or lost (blue) focus. Statistical tests (t-test, Cohen's d) validate significance.</p>
    </div>
    <div class="info-box" style="border-left: 4px solid #2c3e50;">
      <h4 style="margin: 0 0 10px 0;">7. Report Generation</h4>
      <p style="margin: 0;">Generate comprehensive reports showing shortcuts with evidence images, feature impact analysis, and risk assessment per class.</p>
    </div>
  </div>

  <h3>Interpretation Guide</h3>

  <div class="info-box" style="margin-bottom: 20px;">
    <strong>Key Insight:</strong> A <strong>shortcut</strong> is when the model wrongly relies on a contextual feature (background, environment).
    <ul style="margin: 10px 0 0 0;">
      <li><strong>Shortcut detected:</strong> Removing contextual feature → confidence <span class="delta-neg">DROPS</span> → model was using it (BAD)</li>
      <li><strong>Robust behavior:</strong> Removing contextual feature → confidence <span class="delta-pos">INCREASES</span> → model wasn't relying on it (GOOD)</li>
    </ul>
  </div>

  <table>
    <tr>
      <th>Impact (Δ)</th>
      <th>Meaning</th>
      <th>If Intrinsic Feature</th>
      <th>If Contextual Feature</th>
    </tr>
    <tr>
      <td><span class="delta-neg">-0.30 or lower</span></td>
      <td>Very high importance</td>
      <td style="background: #d4edda;">✓ Expected - critical feature</td>
      <td style="background: #f8d7da;">🚨 SHORTCUT - Major bias!</td>
    </tr>
    <tr>
      <td><span class="delta-neg">-0.15 to -0.30</span></td>
      <td>Significant importance</td>
      <td style="background: #d4edda;">✓ Good - model uses this</td>
      <td style="background: #fff3cd;">⚠ SHORTCUT - Bias concern</td>
    </tr>
    <tr>
      <td><span style="color: #666;">-0.05 to +0.05</span></td>
      <td>Minimal impact</td>
      <td style="background: #fff3cd;">May be underutilized</td>
      <td style="background: #d4edda;">✓ Good - not relied upon</td>
    </tr>
    <tr>
      <td><span class="delta-pos">+0.05 to +0.30</span></td>
      <td>Feature was distracting</td>
      <td style="background: #fff3cd;">Unexpected - investigate</td>
      <td style="background: #d4edda;">✓ ROBUST - Model handles noise well</td>
    </tr>
    <tr>
      <td><span class="delta-pos">+0.30 or higher</span></td>
      <td>Feature was hurting badly</td>
      <td style="background: #fff3cd;">Unexpected - investigate</td>
      <td style="background: #d4edda;">✓ Very ROBUST - Model ignores context</td>
    </tr>
  </table>

  <h3>Risk Levels</h3>
  {% set high_thresh = "%.0f"|format((config.risk_high_threshold if config else 0.30) * 100) %}
  {% set med_thresh = "%.0f"|format((config.risk_medium_threshold if config else 0.10) * 100) %}
  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
    <div class="info-box success">
      <h4 style="margin: 0;">🟢 LOW RISK</h4>
      <p style="margin: 10px 0 0 0;">&lt;{{ med_thresh }}% of hypotheses confirmed. Model appears robust and relies primarily on intrinsic features.</p>
    </div>
    <div class="info-box warning">
      <h4 style="margin: 0;">🟡 MEDIUM RISK</h4>
      <p style="margin: 10px 0 0 0;">{{ med_thresh }}-{{ high_thresh }}% confirmed. Some shortcuts present. May need attention for deployment in diverse contexts.</p>
    </div>
    <div class="info-box danger">
      <h4 style="margin: 0;">🔴 HIGH RISK</h4>
      <p style="margin: 10px 0 0 0;">&gt;{{ high_thresh }}% confirmed. Significant shortcuts/biases detected. Model may fail on out-of-distribution data.</p>
    </div>
  </div>
</div>

</body>
</html>
"""

_MD_TEMPLATE = """# Classification Model Bias Analysis Report

**Model**: ResNet-50 (ImageNet-1k) | **Classes analyzed**: {{ results|length }}

## Overall Summary

| Class | Edits | Generations | Confirmed | Rate | Key Features |
|---|---|---|---|---|---|
{% for r in results -%}
| {{ r.class_name }} | {{ r.summary.total_edits }} | {{ r.summary.total_generations }} | {{ r.summary.confirmed_count }} | {{ "%.0f"|format(r.summary.confirmation_rate * 100) }}% | {{ r.key_features[:3]|join(", ") }} |
{% endfor %}

{% for r in results %}
---

## {{ r.class_name }}

### Feature Analysis

**Key features**: {{ r.key_features|join(", ") }}

{% if r.essential_features %}**Essential features** (model should use): {{ r.essential_features|join(", ") }}{% endif %}

{% if r.spurious_features %}**Spurious features** (potential shortcuts): {{ r.spurious_features|join(", ") }}{% endif %}

**Model focus (Grad-CAM)**: {{ r.model_focus }}

{% if r.vlm_insights %}
**VLM Insights**: {{ r.vlm_insights|join("; ") }}
{% endif %}

{% if r.confirmed_shortcuts %}
**VLM-Confirmed Shortcuts**: {{ r.confirmed_shortcuts|join(", ") }}
{% endif %}

### Confirmed Shortcuts ({{ r.confirmed_hypotheses|length }})

{% for e in r.confirmed_hypotheses %}
#### {{ loop.index }}. {{ e.instruction }}

- **Hypothesis**: {{ e.hypothesis }}
- **Mean Δ**: {{ "%+.3f"|format(e.mean_delta) }} (range: {{ "%+.3f"|format(e.min_delta) }} to {{ "%+.3f"|format(e.max_delta) }})
- **Confirmations**: {{ e.confirmation_count }}/{{ e.generations|length }} generations
- **Original confidence**: {{ "%.3f"|format(e.original_confidence) }}

| Generation | Δ Confidence | Image |
|---|---|---|
| Original | - | ![Original]({{ e.original_image_path }}) |
{% for g in e.generations -%}
| Gen {{ loop.index }} | {{ "%+.3f"|format(g.delta) }} | ![Gen {{ loop.index }}]({{ g.edited_image_path }}) |
{% endfor %}

{% endfor %}

### All Edit Results

| Edit | Type | Priority | Mean Δ | Range | Confirmed |
|---|---|---|---|---|---|
{% for e in r.edit_results -%}
| {{ e.instruction[:50] }}{% if e.instruction|length > 50 %}...{% endif %} | {{ e.edit_type }} | {{ e.priority }} | {{ "%+.3f"|format(e.mean_delta) }} | {{ "%+.3f"|format(e.min_delta) }} to {{ "%+.3f"|format(e.max_delta) }} | {{ "✓ " + (e.confirmation_count|string) + "/" + (e.generations|length|string) if e.confirmed else "✗" }} |
{% endfor %}
{% endfor %}

---

## Feature Dependency Rubric

Summary of features, shortcuts, and correlations discovered across all classes.

| Class | Essential Features | Spurious Features | Confirmed Biases | Risk |
|---|---|---|---|---|
{% for r in results -%}
| **{{ r.class_name }}** | {{ r.essential_features|join(", ") if r.essential_features else "Not identified" }} | {{ r.spurious_features|join(", ") if r.spurious_features else (r.confirmed_shortcuts|join(", ") if r.confirmed_shortcuts else "None") }} | {{ r.confirmed_hypotheses|length }} confirmed | {% set rate = r.summary.confirmation_rate %}{% if rate > 0.3 %}🔴 HIGH{% elif rate > 0.1 %}🟡 MEDIUM{% else %}🟢 LOW{% endif %} |
{% endfor %}

### Risk Level Legend
- 🟢 **LOW**: <10% of hypotheses confirmed - model appears robust
- 🟡 **MEDIUM**: 10-30% confirmed - some shortcuts present, may need attention
- 🔴 **HIGH**: >30% confirmed - significant shortcuts/biases detected

### Top Confirmed Shortcuts

{% for r in results if r.confirmed_hypotheses %}
**{{ r.class_name }}:**
{% for h in r.confirmed_hypotheses[:3] %}
- {{ h.instruction }} (Δ={{ "%+.3f"|format(h.mean_delta) }})
{% endfor %}
{% endfor %}
"""


class Reporter:
    """
    Generates analysis reports in JSON, HTML, and Markdown formats.

    The HTML report features:
    - Tabbed interface (Overview, Feature Analysis, Class Details, Methodology)
    - Collapsible image sections with lazy loading for performance
    - Feature impact visualization with color-coded intrinsic vs contextual features
    - Statistical validation results (p-values, Cohen's d)

    Usage:
        reporter = Reporter(output_dir, config)
        paths = reporter.generate_all(results)
    """

    def __init__(self, output_dir: Path, config: dict | None = None):
        """
        Initialize the reporter.

        Args:
            output_dir: Directory to save reports
            config: Optional configuration dict to include in methodology section
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def _convert_paths_to_relative(self, results: list[dict]) -> list[dict]:
        """Convert all absolute image paths to relative paths."""
        results = copy.deepcopy(results)

        def convert_edit_paths(e: dict):
            """Convert paths in an edit result dict."""
            if "original_image_path" in e:
                e["original_image_path"] = _to_relative_path(
                    e["original_image_path"], self.output_dir
                )
            if e.get("original_gradcam_path"):
                e["original_gradcam_path"] = _to_relative_path(
                    e["original_gradcam_path"], self.output_dir
                )
            # Convert paths in all generations
            for g in e.get("generations", []):
                if "edited_image_path" in g:
                    g["edited_image_path"] = _to_relative_path(
                        g["edited_image_path"], self.output_dir
                    )
                if g.get("gradcam_image_path"):
                    g["gradcam_image_path"] = _to_relative_path(
                        g["gradcam_image_path"], self.output_dir
                    )
                if g.get("gradcam_diff_path"):
                    g["gradcam_diff_path"] = _to_relative_path(
                        g["gradcam_diff_path"], self.output_dir
                    )

        for r in results:
            # Convert baseline_results paths
            for b in r.get("baseline_results", []):
                if "image_path" in b:
                    b["image_path"] = _to_relative_path(b["image_path"], self.output_dir)

            # Convert edit_results paths
            for e in r.get("edit_results", []):
                convert_edit_paths(e)

            # Convert confirmed_hypotheses paths
            for e in r.get("confirmed_hypotheses", []):
                convert_edit_paths(e)

        return results

    def save_consolidated_json(self, results: list[dict]) -> Path:
        path = self.output_dir / "analysis_results.json"
        # Keep absolute paths in JSON for programmatic access
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved consolidated JSON: %s", path)
        return path

    def generate_html(self, results: list[dict]) -> Path:
        """Generate HTML report with tabs and interactive features."""
        path = self.output_dir / "report.html"
        # Use relative paths for HTML (so report is portable)
        relative_results = self._convert_paths_to_relative(results)
        html = Template(_HTML_TEMPLATE).render(results=relative_results, config=self.config)
        path.write_text(html, encoding="utf-8")
        logger.info("Saved HTML report: %s", path)
        return path

    def generate_markdown(self, results: list[dict]) -> Path:
        path = self.output_dir / "report.md"
        # Use relative paths for Markdown too
        relative_results = self._convert_paths_to_relative(results)
        md = Template(_MD_TEMPLATE).render(results=relative_results)
        path.write_text(md, encoding="utf-8")
        logger.info("Saved Markdown report: %s", path)
        return path

    def generate_all(self, results: list[dict]) -> dict[str, Path]:
        return {
            "json": self.save_consolidated_json(results),
            "html": self.generate_html(results),
            "markdown": self.generate_markdown(results),
        }
