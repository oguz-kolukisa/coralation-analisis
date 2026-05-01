"""HTML report for the probe pipeline.

Renders a single self-contained HTML page that shows:
  - ranking table sorted by top1_accuracy_overall
  - per-class image grid (bias_heavy / bias_stripped / real_feature_only)
  - per-class × per-model metrics table
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .__version__ import __version__


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_probe_report(manifest_path: Path | str, evaluation_path: Path | str,
                        output_path: Path | str) -> Path:
    """Load manifest + evaluation from disk, render and write the HTML report."""
    manifest = json.loads(Path(manifest_path).read_text())
    evaluation = json.loads(Path(evaluation_path).read_text())
    images_base = _relative_images_base(output_path, manifest_path)
    html_doc = render_probe_report(manifest, evaluation, images_base)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc)
    return output


def render_probe_report(manifest: dict, evaluation: dict,
                         images_base: str = "../probes") -> str:
    """Build the full HTML document from manifest + evaluation dicts."""
    parts = [
        _html_header(evaluation),
        _ranking_table(evaluation),
        _per_class_sections(manifest, evaluation, images_base),
        _html_footer(),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------

def _html_header(evaluation: dict) -> str:
    """Return <html><head>…<body> and the page title."""
    generated = _escape(evaluation.get("generated_at", ""))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Probe evaluation report</title>
<style>{_css()}</style>
</head>
<body>
<header>
  <h1>Probe evaluation report</h1>
  <p class="meta">Coralation v{__version__} — generated {generated}</p>
</header>
<main>"""


def _html_footer() -> str:
    return "</main>\n</body>\n</html>"


def _css() -> str:
    """Inline stylesheet — minimal, dark-friendly, printable."""
    return """
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
             max-width: 1400px; margin: 2rem auto; padding: 0 1.5rem; color: #222; }
      header h1 { margin-bottom: .25rem; }
      .meta { color: #666; font-size: .9rem; }
      section.class-block { border: 1px solid #ddd; border-radius: 6px;
                            padding: 1rem 1.25rem; margin: 2rem 0; }
      h2 { border-bottom: 2px solid #222; padding-bottom: .25rem; }
      h3 { margin-top: 1.25rem; }
      table { border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; }
      th, td { text-align: right; padding: .35rem .6rem; border-bottom: 1px solid #eee;
               font-variant-numeric: tabular-nums; }
      th { background: #f4f4f4; font-weight: 600; text-align: right; }
      th:first-child, td:first-child { text-align: left; font-weight: 500; }
      .feature-list { font-size: .85rem; color: #555; margin: .25rem 0 .75rem; }
      .grid { display: grid; grid-template-columns: 120px repeat(auto-fit, minmax(180px, 1fr));
              gap: .75rem; margin: 1rem 0 1.5rem; align-items: start; }
      .grid .hdr { font-weight: 600; padding: .35rem; background: #f4f4f4;
                   text-align: center; }
      .grid figure { margin: 0; }
      .grid img { width: 100%; border-radius: 4px; display: block; }
      .grid figcaption { font-size: .75rem; color: #666; margin-top: .25rem;
                         line-height: 1.25; overflow-wrap: anywhere; }
      .variant-label { align-self: center; font-weight: 500; color: #444; }
      .rank-1 { background: #eef8ee; }
      .rank-last { background: #fdecec; }
      .pos { color: #c0392b; }
      .neg { color: #27ae60; }
      .neutral { color: #444; }
    """


# ---------------------------------------------------------------------------
# Ranking table
# ---------------------------------------------------------------------------

def _ranking_table(evaluation: dict) -> str:
    """Build the top-level ranking table, sorted by overall top-1."""
    rows = _sorted_models(evaluation)
    body = "\n".join(_ranking_row(r, i, len(rows)) for i, r in enumerate(rows))
    return f"""
<h2>Leaderboard — overall probe-set scores</h2>
<p class="meta">Every model's overall scores on the full probe set (all
variants aggregated). Sorted by <code>overall top-1</code> (descending).
The <b>overall</b> columns are the two headline scores:
<code>top-1</code> = fraction of probe images whose argmax prediction matched
the target class; <code>mean_conf</code> = average confidence for the target
class across all probe images. Per-variant columns follow.
<code>adversarial</code>: top-1 on trick probes — LOW means the model was fooled.
<code>bias_lift</code>: colored red when positive (model leans on biased context),
green when negative.</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>model</th>
      <th colspan="2">overall</th>
      <th>bias_heavy</th><th>bias_stripped</th>
      <th>adversarial</th><th>bias_lift</th>
      <th>n_scored</th><th>n_skipped</th>
    </tr>
    <tr>
      <th></th><th></th>
      <th>top-1</th><th>mean_conf</th>
      <th>top-1</th><th>top-1</th><th>top-1</th><th></th>
      <th></th><th></th>
    </tr>
  </thead>
  <tbody>
    {body}
  </tbody>
</table>"""


def _sorted_models(evaluation: dict) -> list[dict]:
    """Return list of (name, metrics) dicts sorted by overall top-1 desc."""
    scores = evaluation.get("scores", {})
    rows = [{"name": n, **m} for n, m in scores.items()]
    rows.sort(key=lambda r: r["top1_accuracy_overall"], reverse=True)
    return rows


def _ranking_row(row: dict, i: int, total: int) -> str:
    """One <tr> for the ranking table."""
    cls = "rank-1" if i == 0 else ("rank-last" if i == total - 1 else "")
    lift = row["bias_lift_score"]
    return (
        f'<tr class="{cls}">'
        f"<td>{i+1}</td><td>{_escape(row['name'])}</td>"
        f"<td>{row['top1_accuracy_overall']:.3f}</td>"
        f"<td>{row.get('mean_conf_overall', 0):.3f}</td>"
        f"<td>{row['top1_accuracy_bias_heavy']:.3f}</td>"
        f"<td>{row['top1_accuracy_bias_stripped']:.3f}</td>"
        f"<td>{row.get('top1_accuracy_adversarial', 0):.3f}</td>"
        f"<td class='{_lift_css(lift)}'>{lift:+.3f}</td>"
        f"<td>{row['n_classes_scored']}</td>"
        f"<td>{row['n_classes_skipped']}</td>"
        "</tr>"
    )


def _lift_css(lift: float) -> str:
    """Colour class for bias_lift (positive = red, negative = green)."""
    if lift > 0.05:
        return "pos"
    if lift < -0.05:
        return "neg"
    return "neutral"


# ---------------------------------------------------------------------------
# Per-class sections
# ---------------------------------------------------------------------------

def _per_class_sections(manifest: dict, evaluation: dict,
                         images_base: str) -> str:
    """Render one <section> per class containing images + per-model metrics."""
    classes = manifest.get("classes", {})
    blocks = [
        _class_block(name, entry, evaluation, images_base)
        for name, entry in classes.items()
    ]
    if not blocks:
        return "<p><em>No classes in manifest.</em></p>"
    return "<h2>Per-class probes &amp; metrics</h2>\n" + "\n".join(blocks)


def _class_block(class_name: str, entry: dict, evaluation: dict,
                  images_base: str) -> str:
    """Build one class section: features + image grid + per-model table."""
    features_html = _feature_list(entry)
    image_grid = _image_grid(entry, images_base)
    metrics_table = _per_class_model_table(class_name, evaluation)
    return f"""
<section class="class-block">
  <h3>{_escape(class_name)}</h3>
  {features_html}
  <h4>Probe images</h4>
  {image_grid}
  <h4>Per-model metrics on this class</h4>
  {metrics_table}
</section>"""


def _feature_list(entry: dict) -> str:
    """Render the bias, real, and confusing-class lists."""
    bias = ", ".join(entry.get("bias_features", [])) or "(none)"
    real = ", ".join(entry.get("real_features", [])) or "(none)"
    confusing = ", ".join(entry.get("confusing_classes", [])) or "(none)"
    source = entry.get("feature_source_used", "none")
    return (
        f'<p class="feature-list"><b>Biased:</b> {_escape(bias)}<br>'
        f'<b>Real:</b> {_escape(real)}<br>'
        f'<b>Confusing classes:</b> {_escape(confusing)}<br>'
        f'<b>Feature source:</b> <code>{_escape(source)}</code></p>'
    )


# ---------------------------------------------------------------------------
# Image grid (variants × indices)
# ---------------------------------------------------------------------------

_VARIANTS: tuple[str, ...] = (
    "bias_heavy", "bias_stripped", "real_feature_only", "adversarial",
)


def _image_grid(entry: dict, images_base: str) -> str:
    """Render the 3-row grid: one row per variant, one column per index."""
    by_variant = _group_images_by_variant(entry)
    max_idx = max((len(imgs) for imgs in by_variant.values()), default=0)
    if max_idx == 0:
        return "<p><em>No probe images rendered for this class.</em></p>"
    rows = [_grid_header_row(max_idx)]
    for variant in _VARIANTS:
        rows.append(_grid_variant_row(variant, by_variant[variant],
                                       images_base, max_idx))
    return "<div class='grid' style='grid-template-columns: 120px repeat(" \
           f"{max_idx}, minmax(180px, 1fr));'>\n" + "\n".join(rows) + "\n</div>"


def _group_images_by_variant(entry: dict) -> dict[str, list[dict]]:
    """Return {variant -> [ProbeImage-like dicts sorted by index]}."""
    groups: dict[str, list[dict]] = {v: [] for v in _VARIANTS}
    for img in entry.get("images", []):
        groups.setdefault(img["variant"], []).append(img)
    for v in groups:
        groups[v].sort(key=lambda x: x.get("index", 0))
    return groups


def _grid_header_row(max_idx: int) -> str:
    """First row: corner blank + index labels."""
    cells = ["<div class='hdr'></div>"]
    cells.extend(f"<div class='hdr'>#{i}</div>" for i in range(max_idx))
    return "\n".join(cells)


def _grid_variant_row(variant: str, imgs: list[dict], images_base: str,
                      max_idx: int) -> str:
    """One row: variant label + image cells (blank for missing indices)."""
    cells = [f"<div class='variant-label'>{_escape(variant)}</div>"]
    for i in range(max_idx):
        cells.append(_image_cell(imgs[i] if i < len(imgs) else None, images_base))
    return "\n".join(cells)


def _image_cell(img: dict | None, images_base: str) -> str:
    """One figure with the image + its prompt as figcaption."""
    if img is None:
        return "<figure></figure>"
    src = f"{images_base}/{img['image_path']}"
    caption = _escape(img.get("prompt", ""))
    return (
        f"<figure>"
        f"<img src='{_escape(src)}' alt='{_escape(img['variant'])} {img['index']}' loading='lazy'>"
        f"<figcaption>{caption}</figcaption>"
        f"</figure>"
    )


# ---------------------------------------------------------------------------
# Per-class model metrics table
# ---------------------------------------------------------------------------

def _per_class_model_table(class_name: str, evaluation: dict) -> str:
    """Table of each model's stats for this single class."""
    rows = []
    for model_name, model_metrics in evaluation.get("scores", {}).items():
        per_class = model_metrics.get("per_class", {}).get(class_name)
        rows.append(_per_class_row(model_name, per_class))
    return f"""<table>
<thead>
  <tr><th>model</th>
      <th>bh top-1</th><th>bh conf</th>
      <th>bs top-1</th><th>bs conf</th>
      <th>rf top-1</th><th>rf conf</th>
      <th>adv top-1</th><th>adv conf</th>
      <th>lift</th><th>skipped</th></tr>
</thead>
<tbody>
{''.join(rows)}
</tbody></table>"""


def _per_class_row(model_name: str, cm: dict | None) -> str:
    """One <tr> inside the per-class model-metrics table."""
    if cm is None:
        return (
            f"<tr><td>{_escape(model_name)}</td>"
            + "<td>—</td>" * 9
            + "<td>missing</td></tr>"
        )
    bh = _variant_cells(cm.get("bias_heavy"))
    bs = _variant_cells(cm.get("bias_stripped"))
    rf = _variant_cells(cm.get("real_feature_only"))
    adv = _variant_cells(cm.get("adversarial"))
    lift = cm.get("bias_lift", 0.0)
    skip = cm.get("skip_reason") or "" if cm.get("skipped") else ""
    return (
        f"<tr><td>{_escape(model_name)}</td>"
        f"{bh}{bs}{rf}{adv}"
        f"<td class='{_lift_css(lift)}'>{lift:+.3f}</td>"
        f"<td>{_escape(skip)}</td></tr>"
    )


def _variant_cells(vs: dict | None) -> str:
    """Two cells (top-1, mean_conf) for a single VariantStats dict."""
    if not vs:
        return "<td>—</td><td>—</td>"
    return f"<td>{vs['top1_accuracy']:.3f}</td><td>{vs['mean_conf']:.3f}</td>"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _escape(value: object) -> str:
    """HTML-escape any value."""
    return html.escape(str(value))


def _relative_images_base(output_path: Path | str,
                           manifest_path: Path | str) -> str:
    """Compute a relative path from the report file to the probes root."""
    output = Path(output_path).resolve().parent
    manifest_dir = Path(manifest_path).resolve().parent
    return os.path.relpath(manifest_dir, output).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(description="Render probe_report.html")
    p.add_argument("--manifest", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> None:
    args = _cli_parse_args(argv)
    path = write_probe_report(args.manifest, args.evaluation, args.output)
    print(f"Wrote probe report: {path}")


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
