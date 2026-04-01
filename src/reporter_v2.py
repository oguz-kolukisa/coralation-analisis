"""
Reporter V2 — Per-model tabbed reports + cross-model comparison.

Generates:
- Per-model HTML (tabs: Overview, Confirmed Features, All Images, Per-Class)
- Cross-model comparison HTML
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

from .__version__ import __version__

logger = logging.getLogger(__name__)


def _rel(abs_path: str, base: Path) -> str:
    try:
        return str(os.path.relpath(Path(abs_path).resolve(), base.resolve()))
    except ValueError:
        return abs_path


# =============================================================================
# PER-MODEL REPORT TEMPLATE
# =============================================================================

_MODEL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Coralation — {{ model_name }}</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f5f5; color: #333; }
  .header { background: #1a237e; color: white; padding: 20px 30px; }
  .header h1 { margin: 0; font-size: 1.8em; }
  .header p { margin: 5px 0 0; opacity: 0.8; }
  .tabs { display: flex; background: #283593; padding: 0 20px; }
  .tab { padding: 12px 24px; color: rgba(255,255,255,0.7); cursor: pointer; border-bottom: 3px solid transparent; font-weight: 500; }
  .tab.active { color: white; border-bottom-color: #ffab40; }
  .tab-content { display: none; padding: 20px 30px; }
  .tab-content.active { display: block; }
  .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
  .stat { background: white; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
  .stat-val { font-size: 2em; font-weight: bold; color: #1a237e; }
  .stat-label { color: #666; font-size: 0.85em; margin-top: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  th, td { border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left; font-size: 0.9em; }
  th { background: #eceff1; font-weight: 600; }
  tr:hover { background: #f5f5f5; }
  .v-essential { color: #2e7d32; font-weight: bold; }
  .v-spurious { color: #c62828; font-weight: bold; }
  .v-state_bias { color: #e65100; font-weight: bold; }
  .v-unknown { color: #9e9e9e; }
  .v-not_significant { color: #bdbdbd; font-style: italic; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; color: white; margin: 2px; }
  .tag-target { background: #1976d2; }
  .tag-negative { background: #7b1fa2; }
  .tag-environmental { background: #00796b; }
  .img-group { display: flex; gap: 12px; margin: 12px 0; flex-wrap: wrap; align-items: flex-start; }
  .img-item { text-align: center; }
  .img-item img { max-width: 280px; max-height: 220px; border-radius: 6px; border: 1px solid #ddd; display: block; }
  .img-item small { color: #666; font-size: 0.8em; }
  .delta-bar { height: 8px; border-radius: 4px; background: #ecf0f1; margin: 4px 0; max-width: 150px; display: inline-block; }
  .delta-fill { height: 100%; border-radius: 4px; }
  .delta-neg { color: #c62828; }
  .delta-pos { color: #2e7d32; }
  .feature-group { border-left: 4px solid #1976d2; padding-left: 15px; margin: 20px 0; }
  .feature-group.spurious { border-left-color: #c62828; }
  .feature-group.state_bias { border-left-color: #e65100; }
  .feature-group.essential { border-left-color: #2e7d32; }
  h3.feature-title { margin: 0 0 8px; }
</style>
<script>
function showTab(id) {
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(e => e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}
</script>
</head>
<body>

<div class="header">
  <h1>CORALATION — {{ model_name }}</h1>
  <p>Spurious Correlation Analysis Report — v{{ version }} — {{ generated_at }}</p>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('overview')">Overview</div>
  <div class="tab" onclick="showTab('confirmed')">Confirmed Features</div>
  <div class="tab" onclick="showTab('by-feature')">By Feature (Evidence)</div>
  <div class="tab" onclick="showTab('all-images')">All Images</div>
  <div class="tab" onclick="showTab('per-class')">Per-Class Detail</div>
</div>

<!-- ===== TAB 1: OVERVIEW ===== -->
<div id="overview" class="tab-content active">
  <div class="stats">
    <div class="stat"><div class="stat-val">{{ classes|length }}</div><div class="stat-label">Classes</div></div>
    <div class="stat"><div class="stat-val">{{ total_features }}</div><div class="stat-label">Features</div></div>
    <div class="stat"><div class="stat-val">{{ total_edits }}</div><div class="stat-label">Edits Tested</div></div>
    <div class="stat"><div class="stat-val" style="color:#2e7d32">{{ n_essential }}</div><div class="stat-label">Essential</div></div>
    <div class="stat"><div class="stat-val" style="color:#c62828">{{ n_spurious }}</div><div class="stat-label">Spurious</div></div>
    <div class="stat"><div class="stat-val" style="color:#e65100">{{ n_state_bias }}</div><div class="stat-label">State Bias</div></div>
    <div class="stat"><div class="stat-val" style="color:#888">{{ n_not_significant }}</div><div class="stat-label">Not Significant</div></div>
    <div class="stat"><div class="stat-val" style="color:#b71c1c">{{ n_edit_failed }}</div><div class="stat-label">Edit Failed</div></div>
  </div>

  <div class="card">
    <h3>Per-Class Summary</h3>
    <table>
      <tr><th>Class</th><th>Features</th><th>Edits</th><th>Essential</th><th>Spurious</th><th>State Bias</th><th>Not Significant</th><th>Edit Failed</th></tr>
      {% for cls in classes %}
      <tr>
        <td><strong>{{ cls.class_name }}</strong></td>
        <td>{{ cls.unique_concepts|length }}</td>
        <td>{{ cls.edit_results|length }}</td>
        <td class="v-essential">{{ cls._essential }}</td>
        <td class="v-spurious">{{ cls._spurious }}</td>
        <td class="v-state_bias">{{ cls._state_bias }}</td>
        <td style="color:#888">{{ cls._not_significant }}</td>
        <td style="color:#b71c1c">{{ cls._edit_failed }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
</div>

<!-- ===== TAB 2: CONFIRMED FEATURES ===== -->
<div id="confirmed" class="tab-content">
  {# Strong spurious shortcuts (|delta| >= 30%) #}
  <h2 style="margin-top: 30px;">Strong Spurious Shortcuts</h2>
  <p style="color: #666;">High-impact shortcuts — the model heavily relies on these non-essential features (|delta| >= 30%).</p>
  {% for cls in classes %}{% for r in cls.edit_results %}
  {% set v = r.verdict.get(model_name, {}) %}
  {% set d = r.per_model.get(model_name, {}).delta|default(0) %}
  {% if v and v.verdict == 'spurious' and (d|abs) >= 0.30 %}
  <div class="feature-group spurious">
    <h3 class="feature-title">{{ r.feature_name }} <span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span></h3>
    <p style="margin: 4px 0; color: #666; font-size: 0.9em;">{{ cls.class_name }} — {{ r.edit_type }}</p>
    <p style="margin: 4px 0;">{{ r.edit_instruction }}</p>
    <p><strong class="{{ 'delta-neg' if d < 0 else 'delta-pos' }}">Delta: {{ "%.1f"|format(d * 100) }}%</strong>
      {% if v.reasoning %} — <em>{{ v.reasoning }}</em>{% endif %}</p>
    <div class="img-group">
      <div class="img-item"><img src="{{ r.original_image }}"><small>Original</small></div>
      {% if r.gradcam_image %}<div class="img-item"><img src="{{ r.gradcam_image }}"><small>Grad-CAM</small></div>{% endif %}
      <div class="img-item"><img src="{{ r.edited_image }}"><small>Edited</small></div>
    </div>
  </div>
  {% endif %}{% endfor %}{% endfor %}

  {# Subtle spurious shortcuts (|delta| < 30%) #}
  <h2 style="margin-top: 30px;">Subtle Spurious Shortcuts</h2>
  <p style="color: #666;">Lower-impact shortcuts — the model partially relies on these (|delta| < 30% but above threshold).</p>
  {% for cls in classes %}{% for r in cls.edit_results %}
  {% set v = r.verdict.get(model_name, {}) %}
  {% set d = r.per_model.get(model_name, {}).delta|default(0) %}
  {% if v and v.verdict == 'spurious' and (d|abs) < 0.30 %}
  <div class="feature-group spurious" style="opacity: 0.85;">
    <h3 class="feature-title">{{ r.feature_name }} <span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span></h3>
    <p style="margin: 4px 0; color: #666; font-size: 0.9em;">{{ cls.class_name }} — {{ r.edit_type }}</p>
    <p style="margin: 4px 0;">{{ r.edit_instruction }}</p>
    <p><strong class="{{ 'delta-neg' if d < 0 else 'delta-pos' }}">Delta: {{ "%.1f"|format(d * 100) }}%</strong>
      {% if v.reasoning %} — <em>{{ v.reasoning }}</em>{% endif %}</p>
    <div class="img-group">
      <div class="img-item"><img src="{{ r.original_image }}"><small>Original</small></div>
      {% if r.gradcam_image %}<div class="img-item"><img src="{{ r.gradcam_image }}"><small>Grad-CAM</small></div>{% endif %}
      <div class="img-item"><img src="{{ r.edited_image }}"><small>Edited</small></div>
    </div>
  </div>
  {% endif %}{% endfor %}{% endfor %}

  {# State bias and essential — keep as before #}
  {% for verdict_type, label, css in [('state_bias', 'State Bias', 'state_bias'), ('essential', 'Essential Features', 'essential')] %}
  <h2 style="margin-top: 30px;">{{ label }}</h2>
  {% for cls in classes %}{% for r in cls.edit_results %}
  {% set v = r.verdict.get(model_name, {}) %}
  {% if v and v.verdict == verdict_type %}
  <div class="feature-group {{ css }}">
    <h3 class="feature-title">{{ r.feature_name }} <span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span></h3>
    <p style="margin: 4px 0; color: #666; font-size: 0.9em;">{{ cls.class_name }} — {{ r.edit_type }}</p>
    <p style="margin: 4px 0;">{{ r.edit_instruction }}</p>
    <p><strong class="{{ 'delta-neg' if r.per_model.get(model_name, {}).delta|default(0) < 0 else 'delta-pos' }}">
        Delta: {{ "%.1f"|format(r.per_model.get(model_name, {}).delta|default(0) * 100) }}%</strong>
      {% if v.reasoning %} — <em>{{ v.reasoning }}</em>{% endif %}</p>
    <div class="img-group">
      <div class="img-item"><img src="{{ r.original_image }}"><small>Original</small></div>
      {% if r.gradcam_image %}<div class="img-item"><img src="{{ r.gradcam_image }}"><small>Grad-CAM</small></div>{% endif %}
      <div class="img-item"><img src="{{ r.edited_image }}"><small>Edited</small></div>
    </div>
  </div>
  {% endif %}{% endfor %}{% endfor %}
  {% endfor %}

  <!-- Not Significant Results -->
  <h2 style="margin-top: 30px; color: #888;">Not Significant (inconclusive)</h2>
  <p style="color: #888;">These edits produced confidence changes below the threshold — the test was inconclusive, not proof of no effect.</p>
  <details>
    <summary style="cursor: pointer; font-weight: 500; color: #666;">Show {{ n_not_significant }} not-significant results</summary>
    {% for cls in classes %}
    {% for r in cls.edit_results %}
    {% set v = r.verdict.get(model_name, {}) %}
    {% if v and v.verdict == 'not_significant' %}
    <div class="card" style="border-left: 3px solid #ccc; margin: 8px 0; padding: 10px 15px;">
      <strong>{{ r.feature_name }}</strong> <span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span>
      — {{ cls.class_name }} — {{ r.edit_type }}
      <br><small>Delta: {{ "%.1f"|format(r.per_model.get(model_name, {}).delta|default(0) * 100) }}%</small>
    </div>
    {% endif %}
    {% endfor %}
    {% endfor %}
  </details>

  <!-- Edit Failed Results -->
  {% if n_edit_failed > 0 %}
  <h2 style="margin-top: 30px; color: #b71c1c;">Edit Failed (image unchanged)</h2>
  <p style="color: #888;">The image editor produced images nearly identical to the original — these tests are invalid.</p>
  <details>
    <summary style="cursor: pointer; font-weight: 500; color: #666;">Show {{ n_edit_failed }} failed edits</summary>
    {% for cls in classes %}
    {% for r in cls.edit_results %}
    {% set v = r.verdict.get(model_name, {}) %}
    {% if v and v.verdict == 'edit_failed' %}
    <div class="card" style="border-left: 3px solid #b71c1c; margin: 8px 0; padding: 10px 15px;">
      <strong>{{ r.feature_name }}</strong> <span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span>
      — {{ cls.class_name }} — {{ r.edit_type }}
      <br><small>Edit instruction: {{ r.edit_instruction }}</small>
    </div>
    {% endif %}
    {% endfor %}
    {% endfor %}
  </details>
  {% endif %}
</div>

<!-- ===== TAB: BY FEATURE (EVIDENCE) ===== -->
<div id="by-feature" class="tab-content">
  <h2>Feature Evidence (grouped for human evaluation)</h2>
  <p>Each feature shows ALL test results across images. Majority verdict is computed; conflicting evidence is flagged.</p>
  {% for cls in classes %}
  {% if cls.get('feature_summaries') %}
  <h3 style="margin-top: 25px;">{{ cls.class_name }}</h3>
  {% for fs in cls.feature_summaries %}
  {% set mv = fs.per_model_majority.get(model_name, {}) %}
  {% set v_class = mv.get('verdict', 'unknown') %}
  <div class="card" style="border-left: 4px solid {{ '#c62828' if v_class == 'spurious' else '#2e7d32' if v_class == 'essential' else '#e65100' if v_class == 'state_bias' else '#888' }};">
    <h4 style="margin: 0 0 8px;">
      {{ fs.feature_name }} <span class="tag tag-{{ fs.feature_type }}">{{ fs.feature_type }}</span>
      — <strong class="v-{{ v_class }}">{{ v_class }}</strong>
      {% if mv %} ({{ mv.agreement }}){% endif %}
      {% if mv.get('conflicted') %} <span style="color: #e65100; font-weight: bold;">CONFLICTED</span>{% endif %}
    </h4>
    <p style="color: #666; margin: 0 0 10px;">Tested on {{ fs.test_count }} image(s)</p>
    {% for r in cls.edit_results %}
    {% if r.feature_name == fs.feature_name %}
    {% set rv = r.verdict.get(model_name, {}) %}
    <div style="display: flex; gap: 12px; align-items: center; margin: 8px 0; padding: 8px; background: #fafafa; border-radius: 4px;">
      <div class="img-group" style="flex-shrink: 0;">
        <div class="img-item"><img src="{{ r.original_image }}" style="max-height:100px"><small>Original</small></div>
        {% if r.gradcam_image %}<div class="img-item"><img src="{{ r.gradcam_image }}" style="max-height:100px"><small>Grad-CAM</small></div>{% endif %}
        <div class="img-item"><img src="{{ r.edited_image }}" style="max-height:100px"><small>Edited</small></div>
      </div>
      <div>
        <strong class="{{ 'delta-neg' if r.per_model.get(model_name, {}).delta|default(0) < 0 else 'delta-pos' }}">
          Delta: {{ "%.1f"|format(r.per_model.get(model_name, {}).delta|default(0) * 100) }}%
        </strong>
        — {{ rv.get('verdict', '—') }}
        {% if rv.get('reasoning') %}<br><em style="color:#666; font-size: 0.85em;">{{ rv.reasoning }}</em>{% endif %}
        {% if r.get('edit_failed') %}<br><span style="color:#b71c1c; font-weight:bold;">EDIT FAILED</span>{% endif %}
      </div>
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endfor %}
  {% endif %}
  {% endfor %}
</div>

<!-- ===== TAB 3: ALL IMAGES ===== -->
<div id="all-images" class="tab-content">
  {% set edit_type_sections = [
    ('feature_removal', 'Feature Removal (positive)', 'Removing target features from correctly-classified images'),
    ('state_change', 'State Change (positive)', 'Modifying state/pose of target features'),
    ('environment_removal', 'Environment Removal (positive)', 'Removing environmental elements from target images'),
    ('environment_change', 'Environment Change (positive)', 'Replacing environmental elements with different ones'),
    ('negative_modify', 'Negative Modify', 'Modifying features on non-target images toward target class'),
    ('negative_environment_add', 'Negative Environment Add', 'Adding target-class environmental elements to non-target images'),
  ] %}

  {% for et, et_label, et_desc in edit_type_sections %}
  {% set et_results = [] %}
  {% for cls in classes %}{% for r in cls.edit_results %}{% if r.edit_type == et %}{% set _ = et_results.append({'r': r, 'cls': cls.class_name}) %}{% endif %}{% endfor %}{% endfor %}

  {% if et_results %}
  <h2 style="margin-top: 30px; border-bottom: 2px solid #1976d2; padding-bottom: 5px;">{{ et_label }}</h2>
  <p style="color: #666;">{{ et_desc }}</p>

  {% for item in et_results %}
  {% set r = item.r %}
  <div class="card" style="padding: 12px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <span>
        <strong>{{ item.cls }}</strong> — {{ r.feature_name }}
        <span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span>
      </span>
      <span>
        {% set m = r.per_model.get(model_name, {}) %}
        {% if m %}
        <strong class="{{ 'delta-neg' if m.delta < 0 else 'delta-pos' }}" style="font-size: 1.1em;">
          {{ "%.1f"|format(m.delta * 100) }}%
        </strong>
        {% endif %}
        {% set v = r.verdict.get(model_name, {}) %}
        {% if v and v.verdict %}<span class="v-{{ v.verdict }}"> {{ v.verdict|upper }}</span>{% endif %}
      </span>
    </div>
    <p style="margin: 4px 0; color: #555; font-size: 0.9em;">{{ r.edit_instruction }}</p>
    <div class="img-group">
      <div class="img-item"><img src="{{ r.original_image }}"><small>Original</small></div>
      {% if r.gradcam_image %}<div class="img-item"><img src="{{ r.gradcam_image }}"><small>Grad-CAM</small></div>{% endif %}
      <div class="img-item"><img src="{{ r.edited_image }}"><small>Edited</small></div>
    </div>
  </div>
  {% endfor %}
  {% endif %}
  {% endfor %}
</div>

<!-- ===== TAB 4: PER-CLASS DETAIL ===== -->
<div id="per-class" class="tab-content">
  {% for cls in classes %}
  <div class="card">
    <h2>{{ cls.class_name }}</h2>
    <h4>Discovered Features</h4>
    <div>{% for c in cls.unique_concepts %}<span class="tag tag-{{ c.type }}">{{ c.name }}</span>{% endfor %}</div>

    <h4 style="margin-top: 15px;">Results</h4>
    <table>
      <tr><th>Feature</th><th>Type</th><th>Edit Type</th><th>Delta</th><th>Verdict</th></tr>
      {% for r in cls.edit_results %}
      {% set m = r.per_model.get(model_name, {}) %}
      {% set v = r.verdict.get(model_name, {}) %}
      <tr>
        <td>{{ r.feature_name }}</td>
        <td><span class="tag tag-{{ r.feature_type }}">{{ r.feature_type }}</span></td>
        <td>{{ r.edit_type }}</td>
        <td class="{{ 'delta-neg' if m.delta|default(0) < 0 else 'delta-pos' }}">{{ "%.1f"|format(m.delta|default(0) * 100) }}%</td>
        <td class="v-{{ v.verdict|default('unknown') }}">{{ v.verdict|default('—')|upper }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endfor %}
</div>

</body>
</html>
"""


# =============================================================================
# COMPARISON REPORT TEMPLATE
# =============================================================================

_COMPARISON_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Coralation — Cross-Model Analysis</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; color: #333; }
  h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
  h2 { color: #283593; margin-top: 30px; }
  .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
  .stat { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
  .stat h3 { margin: 0 0 10px; color: #1a237e; }
  .stat-big { font-size: 2.2em; font-weight: bold; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }
  th, td { border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left; }
  th { background: #eceff1; font-weight: 600; }
  tr:hover { background: #fafafa; }
  .v-essential { color: #2e7d32; font-weight: bold; }
  .v-spurious { color: #c62828; font-weight: bold; }
  .v-state_bias { color: #e65100; font-weight: bold; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; color: white; margin: 2px; }
  .tag-target { background: #1976d2; }
  .tag-negative { background: #7b1fa2; }
  .tag-environmental { background: #00796b; }
  .tag-spurious { background: #c62828; }
  .tag-essential { background: #2e7d32; }
  .tag-state_bias { background: #e65100; }
  .heat-0 { background: #e8f5e9; }
  .heat-low { background: #fff9c4; }
  .heat-med { background: #ffe0b2; }
  .heat-high { background: #ffcdd2; }
  .agreement { background: #e8f5e9; font-weight: bold; }
  .disagreement { background: #fff3e0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
</style>
</head>
<body>
<h1>CORALATION — Cross-Model Comparative Analysis</h1>
<p style="color: #666;">Comparing {{ models|length }} classifiers across {{ class_stats|length }} classes — v{{ version }} — {{ generated_at }}</p>

<!-- ===== 1. MODEL OVERVIEW ===== -->
<h2>Model Robustness Overview</h2>
<div class="stats">
  {% for model in models %}
  <div class="stat">
    <h3>{{ model }}</h3>
    <div class="stat-big v-spurious">{{ model_stats[model].spurious }}</div>
    <div style="color: #666; margin-bottom: 10px;">spurious shortcuts found</div>
    <p>Essential: <strong class="v-essential">{{ model_stats[model].essential }}</strong></p>
    <p>State Bias: <strong class="v-state_bias">{{ model_stats[model].state_bias }}</strong></p>
    {% set total = model_stats[model].essential + model_stats[model].spurious + model_stats[model].state_bias %}
    {% if total > 0 %}
    <p style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">
      Spurious rate: <strong>{{ "%.0f"|format(model_stats[model].spurious / total * 100) }}%</strong>
    </p>
    {% endif %}
  </div>
  {% endfor %}
</div>

<!-- ===== 2. VULNERABILITY HEATMAP ===== -->
<div class="card">
  <h2>Vulnerability Heatmap — Spurious Features per Class</h2>
  <table>
    <tr>
      <th>Class</th>
      {% for model in models %}<th>{{ model }}</th>{% endfor %}
      <th>Features</th>
    </tr>
    {% for cls in class_stats %}
    <tr>
      <td><strong>{{ cls.class_name }}</strong></td>
      {% for model in models %}
      {% set count = cls.spurious_per_model.get(model, 0) %}
      <td class="{{ 'heat-0' if count == 0 else 'heat-low' if count <= 2 else 'heat-med' if count <= 5 else 'heat-high' }}">
        {{ count }}
      </td>
      {% endfor %}
      <td>{{ cls.total_features }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<!-- ===== 3. SHARED vs MODEL-SPECIFIC SHORTCUTS ===== -->
<div class="card">
  <h2>Shared vs Model-Specific Shortcuts</h2>
  <p style="color: #666;">Shared = all models agree it's spurious (dataset problem). Model-specific = only some models flag it (architecture weakness).</p>

  <div class="two-col">
    <div>
      <h3 style="color: #c62828;">Universal Shortcuts (all models agree)</h3>
      {% set universal = [] %}
      {% set seen_universal = [] %}
      {% for row in rows %}
      {% set key = row.class_name + '::' + row.feature_name %}
      {% if key not in seen_universal %}
      {% set all_spurious = true %}
      {% for model in models %}{% if row.verdicts.get(model) not in ['spurious', 'state_bias'] %}{% set all_spurious = false %}{% endif %}{% endfor %}
      {% if all_spurious and row.verdicts|length >= models|length %}{% set _ = universal.append(row) %}{% set _ = seen_universal.append(key) %}{% endif %}
      {% endif %}
      {% endfor %}
      {% if universal %}
      <table>
        <tr><th>Class</th><th>Feature</th><th>Type</th></tr>
        {% for row in universal %}
        <tr>
          <td>{{ row.class_name }}</td>
          <td><strong>{{ row.feature_name }}</strong></td>
          <td><span class="tag tag-{{ row.feature_type }}">{{ row.feature_type }}</span></td>
        </tr>
        {% endfor %}
      </table>
      {% else %}<p><em>No universal shortcuts found</em></p>{% endif %}
    </div>
    <div>
      <h3 style="color: #e65100;">Model-Specific Shortcuts (disagreement)</h3>
      {% set specific = [] %}
      {% set seen_specific = [] %}
      {% for row in rows %}
      {% set key = row.class_name + '::' + row.feature_name %}
      {% if key not in seen_specific %}
      {% set verdicts_set = [] %}
      {% for model in models %}{% set _ = verdicts_set.append(row.verdicts.get(model, '')) %}{% endfor %}
      {% set has_spurious = 'spurious' in verdicts_set or 'state_bias' in verdicts_set %}
      {% set has_essential = 'essential' in verdicts_set %}
      {% if has_spurious and has_essential %}{% set _ = specific.append(row) %}{% set _ = seen_specific.append(key) %}{% endif %}
      {% endif %}
      {% endfor %}
      {% if specific %}
      <table>
        <tr><th>Class</th><th>Feature</th>{% for m in models %}<th>{{ m }}</th>{% endfor %}</tr>
        {% for row in specific %}
        <tr>
          <td>{{ row.class_name }}</td>
          <td><strong>{{ row.feature_name }}</strong></td>
          {% for m in models %}<td class="v-{{ row.verdicts.get(m, 'unknown') }}">{{ row.verdicts.get(m, '—')|upper }}</td>{% endfor %}
        </tr>
        {% endfor %}
      </table>
      {% else %}<p><em>No disagreements found</em></p>{% endif %}
    </div>
  </div>
</div>

<!-- ===== 4. PER-CLASS: ESSENTIAL vs SPURIOUS FEATURES ===== -->
<h2>Per-Class Feature Analysis</h2>
{% for cls in class_stats %}
<div class="card">
  <h3>{{ cls.class_name }}</h3>
  <div class="two-col">
    <div>
      <h4 class="v-essential">Essential Features</h4>
      {% for model in models %}
      <p style="margin: 6px 0;"><strong>{{ model }}:</strong>
        {% for f in cls.essential_per_model.get(model, []) %}
          <span class="tag tag-essential">{{ f }}</span>
        {% endfor %}
        {% if not cls.essential_per_model.get(model) %}<em style="color: #aaa;">none confirmed</em>{% endif %}
      </p>
      {% endfor %}
    </div>
    <div>
      <h4 class="v-spurious">Spurious Features</h4>
      {% for model in models %}
      <p style="margin: 6px 0;"><strong>{{ model }}:</strong>
        {% for f in cls.spurious_per_model_names.get(model, []) %}
          <span class="tag tag-spurious">{{ f }}</span>
        {% endfor %}
        {% if not cls.spurious_per_model_names.get(model) %}<em style="color: #aaa;">none found</em>{% endif %}
      </p>
      {% endfor %}
    </div>
    <div>
      <h4 class="v-state_bias">State Bias</h4>
      {% for model in models %}
      <p style="margin: 6px 0;"><strong>{{ model }}:</strong>
        {% for f in cls.state_bias_per_model_names.get(model, []) %}
          <span class="tag tag-state_bias">{{ f }}</span>
        {% endfor %}
        {% if not cls.state_bias_per_model_names.get(model) %}<em style="color: #aaa;">none found</em>{% endif %}
      </p>
      {% endfor %}
    </div>
  </div>
</div>
{% endfor %}

<footer style="color: #999; margin-top: 40px; font-size: 0.85em;">
  Generated by Coralation V2 | Models: {{ models|join(', ') }}
  <br>For detailed per-model analysis with images, see individual model reports.
</footer>
</body>
</html>
"""


# =============================================================================
# REPORTER CLASS
# =============================================================================

class ReporterV2:
    """Generate per-model + comparison HTML reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, results: list[dict], models: list[str]) -> list[Path]:
        """Generate per-model reports + comparison report."""
        paths = []
        for model in models:
            p = self._generate_model_report(results, model)
            paths.append(p)
        p = self._generate_comparison(results, models)
        paths.append(p)
        return paths

    def _generate_model_report(self, results, model_name) -> Path:
        """Generate one per-model tabbed report."""
        rel_results = self._make_relative(results)
        self._add_verdict_counts(rel_results, model_name)
        counts = self._count_verdicts(rel_results, model_name)
        html = Template(_MODEL_TEMPLATE).render(
            model_name=model_name, classes=rel_results,
            total_features=sum(len(c.get("unique_concepts", [])) for c in rel_results),
            total_edits=sum(len(c.get("edit_results", [])) for c in rel_results),
            version=__version__,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            **counts,
        )
        path = self.output_dir / f"{model_name}_report.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved report: %s", path)
        return path

    def _generate_comparison(self, results, models) -> Path:
        """Generate cross-model comparison report."""
        rel_results = self._make_relative(results)
        rows = self._build_comparison_rows(rel_results, models)
        model_stats = self._build_model_stats(rel_results, models)
        class_stats = self._build_class_stats(rel_results, models)
        html = Template(_COMPARISON_TEMPLATE).render(
            models=models, rows=rows, model_stats=model_stats,
            class_stats=class_stats,
            version=__version__,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )
        path = self.output_dir / "comparison_report.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved comparison: %s", path)
        return path

    def _add_verdict_counts(self, results, model_name):
        """Add verdict counts to each class dict."""
        for cls in results:
            cls["_essential"] = 0
            cls["_spurious"] = 0
            cls["_state_bias"] = 0
            cls["_not_significant"] = 0
            cls["_edit_failed"] = 0
            for er in cls.get("edit_results", []):
                v = er.get("verdict", {}).get(model_name, {})
                verdict = v.get("verdict", "") if isinstance(v, dict) else ""
                if verdict == "essential":
                    cls["_essential"] += 1
                elif verdict == "spurious":
                    cls["_spurious"] += 1
                elif verdict == "state_bias":
                    cls["_state_bias"] += 1
                elif verdict == "not_significant":
                    cls["_not_significant"] += 1
                elif verdict == "edit_failed":
                    cls["_edit_failed"] += 1

    def _count_verdicts(self, results, model_name) -> dict:
        """Count total verdicts across all classes."""
        n_e = sum(c.get("_essential", 0) for c in results)
        n_s = sum(c.get("_spurious", 0) for c in results)
        n_sb = sum(c.get("_state_bias", 0) for c in results)
        n_ns = sum(c.get("_not_significant", 0) for c in results)
        n_ef = sum(c.get("_edit_failed", 0) for c in results)
        return {
            "n_essential": n_e, "n_spurious": n_s, "n_state_bias": n_sb,
            "n_not_significant": n_ns, "n_edit_failed": n_ef,
        }

    def _build_comparison_rows(self, results, models):
        """Build flat rows for comparison table."""
        rows = []
        for cls in results:
            for er in cls.get("edit_results", []):
                deltas = {}
                verdicts = {}
                for m in models:
                    pm = er.get("per_model", {}).get(m, {})
                    deltas[m] = pm.get("delta", 0) if pm else 0
                    v = er.get("verdict", {}).get(m, {})
                    verdicts[m] = v.get("verdict", "") if isinstance(v, dict) else ""
                rows.append({
                    "class_name": cls["class_name"],
                    "feature_name": er.get("feature_name", ""),
                    "feature_type": er.get("feature_type", ""),
                    "edit_type": er.get("edit_type", ""),
                    "deltas": deltas,
                    "verdicts": verdicts,
                })
        return rows

    def _build_model_stats(self, results, models):
        """Count verdicts per model."""
        stats = {}
        for m in models:
            e, s, sb = 0, 0, 0
            for cls in results:
                for er in cls.get("edit_results", []):
                    v = er.get("verdict", {}).get(m, {})
                    vt = v.get("verdict", "") if isinstance(v, dict) else ""
                    if vt == "essential": e += 1
                    elif vt == "spurious": s += 1
                    elif vt == "state_bias": sb += 1
            stats[m] = {"essential": e, "spurious": s, "state_bias": sb}
        return stats

    def _build_class_stats(self, results, models):
        """Build per-class stats with feature names per model."""
        stats = []
        for cls in results:
            entry = {
                "class_name": cls["class_name"],
                "total_features": len(cls.get("unique_concepts", [])),
                "spurious_per_model": {},
                "essential_per_model": {},
                "spurious_per_model_names": {},
                "state_bias_per_model_names": {},
            }
            for m in models:
                s_names, sb_names, e_names = [], [], []
                for er in cls.get("edit_results", []):
                    v = er.get("verdict", {}).get(m, {})
                    vt = v.get("verdict", "") if isinstance(v, dict) else ""
                    fname = er.get("feature_name", "")
                    if vt == "spurious":
                        if fname not in s_names:
                            s_names.append(fname)
                    elif vt == "state_bias":
                        if fname not in sb_names:
                            sb_names.append(fname)
                    elif vt == "essential":
                        if fname not in e_names:
                            e_names.append(fname)
                entry["spurious_per_model"][m] = len(s_names)
                entry["spurious_per_model_names"][m] = s_names
                entry["state_bias_per_model_names"][m] = sb_names
                entry["essential_per_model"][m] = e_names
            stats.append(entry)
        return stats

    def _make_relative(self, results):
        """Convert image paths to relative."""
        import copy
        results = copy.deepcopy(results)
        for cls in results:
            for er in cls.get("edit_results", []):
                for key in ("original_image", "edited_image", "gradcam_image"):
                    if key in er and er[key]:
                        er[key] = _rel(er[key], self.output_dir)
        return results
