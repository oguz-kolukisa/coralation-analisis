"""Generate framework diagram for IEEE paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Colors ──
C_INPUT = "#E8F4FD"
C_MODEL = "#FFF3E0"
C_VLM = "#F3E5F5"
C_EDIT = "#E8F5E9"
C_MEASURE = "#FBE9E7"
C_OUTPUT = "#E0F2F1"
C_BORDER = "#37474F"
C_ARROW = "#546E7A"
C_TITLE = "#1A237E"
C_LOOP = "#EDE7F6"


def box(x, y, w, h, label, sublabel, color, fontsize=9):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2 + 0.15, label,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold", color="#212121",
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.2, sublabel,
            ha="center", va="center", fontsize=7,
            color="#616161", style="italic",
        )


def arrow(x1, y1, x2, y2, label="", color=C_ARROW):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.8, connectionstyle="arc3,rad=0",
        ),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx, my + 0.15, label, ha="center", va="bottom",
            fontsize=6.5, color="#455A64", style="italic",
        )


def curved_arrow(x1, y1, x2, y2, label="", rad=0.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color="#7E57C2",
            lw=1.5, connectionstyle=f"arc3,rad={rad}",
            linestyle="dashed",
        ),
    )
    if label:
        mx = (x1 + x2) / 2 + rad * 1.5
        my = (y1 + y2) / 2
        ax.text(
            mx, my, label, ha="center", va="center",
            fontsize=6.5, color="#7E57C2", style="italic",
        )


# ── Title ──
ax.text(
    7, 9.6, "CORALATION: Automated Spurious Correlation Discovery Framework",
    ha="center", va="center", fontsize=13, fontweight="bold", color=C_TITLE,
)
ax.text(
    7, 9.25, "Counterfactual editing pipeline for classifier bias detection",
    ha="center", va="center", fontsize=9, color="#455A64", style="italic",
)

# ── Phase 1: Input ──
box(0.3, 7.5, 2.5, 1.2, "1. Sample Collection", "ImageNet dataset", C_INPUT, 9)
ax.text(1.55, 7.55, "Positive + Negative samples", ha="center", fontsize=6, color="#616161")

# ── Phase 2: Baseline Classification ──
box(3.8, 7.5, 2.8, 1.2, "2. Baseline Classification", "ResNet / DINOv2 / ViT", C_MODEL, 9)
ax.text(5.2, 7.55, "Confidence + Attention Maps", ha="center", fontsize=6, color="#616161")

# ── Phase 3: Feature Discovery ──
box(7.6, 7.5, 2.8, 1.2, "3. Feature Discovery", "Qwen2.5-VL-7B", C_VLM, 9)
ax.text(9.0, 7.55, "Hypotheses + Edit Instructions", ha="center", fontsize=6, color="#616161")

# ── Phase 4: Counterfactual Editing ──
box(11.2, 7.5, 2.5, 1.2, "4. Counterfactual Editing", "FLUX.2-klein", C_EDIT, 9)
ax.text(12.45, 7.55, "Edited images", ha="center", fontsize=6, color="#616161")

# ── Arrows between top row ──
arrow(2.8, 8.1, 3.8, 8.1, "samples")
arrow(6.6, 8.1, 7.6, 8.1, "images +\nGrad-CAM")
arrow(10.4, 8.1, 11.2, 8.1, "edit\ninstructions")

# ── Phase 5: Impact Measurement (center) ──
box(4.5, 5.2, 5.0, 1.4, "5. Impact Measurement & Statistical Validation", "", C_MEASURE, 10)
ax.text(7.0, 5.55, "Re-classify edited images  |  Confidence \u0394  |  t-test + Cohen's d  |  Attention diff", ha="center", fontsize=7, color="#616161")
ax.text(7.0, 5.25, "Determine: Essential  |  Contextual shortcut  |  Spurious correlation  |  Robust", ha="center", fontsize=6.5, color="#BF360C")

# ── Arrow from editing down to measurement ──
arrow(12.45, 7.5, 12.45, 6.6)
arrow(12.45, 6.2, 9.5, 6.0, "edited images")

# ── Arrow from classifier to measurement ──
arrow(5.2, 7.5, 5.7, 6.6, "baseline\nconfidence")

# ── Iterative loop arrow (from measurement back to feature discovery) ──
curved_arrow(7.0, 6.6, 9.0, 7.5, "Iterative\nrefinement", rad=-0.4)

# ── Phase 6: Feature Classification (left center) ──
box(0.3, 5.2, 3.5, 1.4, "6. Feature Classification", "VLM semantic analysis", C_VLM, 9)
ax.text(2.05, 5.3, "Intrinsic vs Contextual labeling", ha="center", fontsize=6.5, color="#616161")

arrow(4.5, 5.9, 3.8, 5.9, "edit results")

# ── Phase 7: Multi-Model Comparison ──
box(0.3, 3.0, 3.5, 1.5, "7. Multi-Model Comparison", "", C_MODEL, 9)
ax.text(2.05, 3.4, "ResNet-50  vs  DINOv2  vs  ViT-L", ha="center", fontsize=7, color="#616161")
ax.text(2.05, 3.1, "Same edits, different classifiers", ha="center", fontsize=6.5, color="#616161")

arrow(2.05, 5.2, 2.05, 4.5, "classified\nresults")

# ── Phase 8: Report Generation ──
box(4.5, 3.0, 5.0, 1.5, "8. Report Generation", "", C_OUTPUT, 10)
ax.text(7.0, 3.55, "Interactive HTML  |  Markdown  |  JSON", ha="center", fontsize=7.5, color="#616161")
ax.text(7.0, 3.15, "Per-model reports + Cross-model comparison", ha="center", fontsize=7, color="#00695C")

arrow(3.8, 3.75, 4.5, 3.75, "per-model\nresults")
arrow(7.0, 5.2, 7.0, 4.5, "analysis\nresults")

# ── Output artifacts ──
box(10.5, 3.4, 3.2, 0.7, "Spurious Features", "", "#FFCDD2", 8)
box(10.5, 2.5, 3.2, 0.7, "Shortcut Evidence", "", "#FFCDD2", 8)

arrow(9.5, 3.75, 10.5, 3.75, "findings")
arrow(9.5, 3.3, 10.5, 2.85)

# ── Legend ──
legend_y = 1.2
legend_items = [
    (C_INPUT, "Input/Data"),
    (C_MODEL, "Classifier"),
    (C_VLM, "Vision-Language Model"),
    (C_EDIT, "Image Editor"),
    (C_MEASURE, "Analysis"),
    (C_OUTPUT, "Output"),
]
ax.text(0.3, legend_y + 0.5, "Components:", fontsize=8, fontweight="bold", color="#37474F")
for i, (color, label) in enumerate(legend_items):
    x = 0.3 + i * 2.2
    rect = FancyBboxPatch(
        (x, legend_y - 0.05), 0.35, 0.35,
        boxstyle="round,pad=0.05", facecolor=color, edgecolor=C_BORDER, linewidth=1,
    )
    ax.add_patch(rect)
    ax.text(x + 0.5, legend_y + 0.12, label, fontsize=7, va="center", color="#37474F")

# ── Model labels at bottom ──
ax.text(
    7, 0.4,
    "Models: Classifier = ResNet-50 / DINOv2-ViT-B/14 / ViT-L/16  |  "
    "VLM = Qwen2.5-VL-7B  |  Editor = FLUX.2-klein  |  "
    "Attention = Score-CAM / Grad-CAM",
    ha="center", va="center", fontsize=7, color="#78909C",
)

plt.tight_layout(pad=0.5)
plt.savefig("docs/framework_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("docs/framework_diagram.pdf", bbox_inches="tight", facecolor="white")
print("Saved: docs/framework_diagram.png and docs/framework_diagram.pdf")
