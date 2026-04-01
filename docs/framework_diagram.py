"""Generate V2 framework diagram — bold, readable, with example images."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 13))
ax.set_xlim(0, 16)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Saturated Colors ──
PAL = {
    "data": "#AED6F1", "clf": "#F9E79F", "vlm": "#D7BDE2",
    "edit": "#A9DFBF", "measure": "#F5B7B1", "verdict": "#FAD7A0",
    "border": "#1C2833", "arrow": "#2C3E50", "title": "#1B2631",
    "sub": "#566573", "text": "#1C2833",
}


def box(x, y, w, h, num, title, sub, color):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                        facecolor=color, edgecolor=PAL["border"], linewidth=1.8)
    ax.add_patch(r)
    ax.text(x + 0.25, y + h - 0.25, num, fontsize=9, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.22", fc=PAL["border"], ec="none"))
    ax.text(x + w / 2, y + h / 2 + 0.12, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=PAL["text"])
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center",
                fontsize=8.5, color=PAL["sub"], fontweight="medium")


def arr(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=PAL["arrow"],
                                lw=2.0, connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom",
                fontsize=7.5, color=PAL["sub"], fontweight="medium")


def add_example_image(path, x, y, w, h, label=""):
    """Add a real image from the output as an example."""
    try:
        img = mpimg.imread(str(path))
        ax_img = fig.add_axes([x, y, w, h])
        ax_img.imshow(img)
        ax_img.axis("off")
        if label:
            ax_img.set_title(label, fontsize=6, color=PAL["sub"], pad=1)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════
ax.text(8, 12.5, "CORALATION", ha="center", fontsize=24, fontweight="bold",
        color=PAL["title"], family="serif")
ax.text(8, 12.05, "Concept-Based Counterfactual Framework for Spurious Correlation Discovery",
        ha="center", fontsize=11, color=PAL["sub"], fontweight="medium")

# ═════════════════════════════════════════════════════════════════
# ROW 1: COLLECTION + CLASSIFICATION + DISCOVERY
# ═════════════════════════════════════════════════════════════════
y1 = 10.0
box(0.4, y1, 3.5, 1.3, "1", "Sample Collection", "Positive + negative images", PAL["data"])
box(4.4, y1, 3.5, 1.3, "2", "Baseline Classification", "Confidence + Grad-CAM", PAL["clf"])
box(8.4, y1, 4.0, 1.3, "3", "Feature Discovery", "Target | Negative | Environmental", PAL["vlm"])

arr(3.9, 10.65, 4.4, 10.65, "images")
arr(7.9, 10.65, 8.4, 10.65, "images +\nheatmaps")

# Example images row (between row 1 and 2)
img_base = Path("/tmp/coralation_v2/images")
# Try to find example images from a recent run
example_dirs = sorted(img_base.iterdir()) if img_base.exists() else []
if example_dirs:
    d = example_dirs[0]
    originals = sorted(d.glob("pos_0_original*"))
    gradcams = sorted(d.glob("pos_0_gradcam*"))
    editeds = sorted(d.glob("pos_*_removal_*"))

    # Add example images as insets using figure coordinates
    if originals:
        add_example_image(originals[0], 0.06, 0.68, 0.08, 0.06, "Original")
    if gradcams:
        add_example_image(gradcams[0], 0.16, 0.68, 0.08, 0.06, "Grad-CAM")
    if editeds:
        add_example_image(editeds[0], 0.26, 0.68, 0.08, 0.06, "Edited")

# ═════════════════════════════════════════════════════════════════
# ROW 2: DEDUP + EDIT GEN + MAPPING
# ═════════════════════════════════════════════════════════════════
y2 = 7.6
box(0.4, y2, 3.5, 1.3, "4", "Feature Dedup", "Merge across all images", PAL["data"])
box(4.4, y2, 3.5, 1.3, "5", "Edit Generation", "Per feature \u00d7 edit type", PAL["vlm"])
box(8.4, y2, 4.0, 1.3, "6", "Edit Mapping", "Feature \u2192 matching images", PAL["data"])

arr(10.4, 10.0, 2.2, 8.9, "concept list")
arr(3.9, 8.25, 4.4, 8.25, "unique\nfeatures")
arr(7.9, 8.25, 8.4, 8.25, "edit plans")

# ═════════════════════════════════════════════════════════════════
# ROW 3: EDITING + MEASUREMENT
# ═════════════════════════════════════════════════════════════════
y3 = 5.2
box(0.4, y3, 4.5, 1.3, "7", "Image Editing", "Counterfactual generation (FLUX.2)", PAL["edit"])
box(5.4, y3, 7.0, 1.3, "8", "Impact Measurement", "ResNet-50  \u00b7  DINOv2-ViT-B/14  \u00b7  ViT-L/16", PAL["measure"])

arr(10.4, 7.6, 2.7, 6.5, "edit\u2013image pairs")
arr(4.9, 5.85, 5.4, 5.85, "edited images")

# ═════════════════════════════════════════════════════════════════
# ROW 4: VERDICT
# ═════════════════════════════════════════════════════════════════
y4 = 2.8
box(0.4, y4, 12.0, 1.3, "9", "Feature Verdict", "VLM judges per model:  Essential  |  Spurious  |  State Bias", PAL["verdict"])

arr(8.9, 5.2, 6.4, 4.1, "confidence \u0394 + images")

# Verdict color indicators
vx = 1.2
for label, color in [("ESSENTIAL", "#27AE60"), ("SPURIOUS", "#E74C3C"), ("STATE BIAS", "#E67E22")]:
    ax.add_patch(FancyBboxPatch((vx, 2.9), 0.25, 0.25, boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="none"))
    ax.text(vx + 0.35, 3.02, label, fontsize=7.5, fontweight="bold",
            va="center", color=color)
    vx += 3.0

# ═════════════════════════════════════════════════════════════════
# BOTTOM: Edit type examples
# ═════════════════════════════════════════════════════════════════
ax.text(8, 2.2, "Edit Types", ha="center", fontsize=10, fontweight="bold", color=PAL["text"])

edit_types = [
    ("Feature\nRemoval", PAL["vlm"]),
    ("State\nChange", PAL["verdict"]),
    ("Environment\nRemoval", PAL["edit"]),
    ("Environment\nChange", PAL["data"]),
    ("Negative\nModify", PAL["measure"]),
    ("Negative\nEnv Add", PAL["clf"]),
]
ex = 1.0
for label, color in edit_types:
    ax.add_patch(FancyBboxPatch((ex, 0.8), 2.0, 0.9, boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor=PAL["border"], linewidth=1.0))
    ax.text(ex + 1.0, 1.25, label, ha="center", va="center",
            fontsize=8, fontweight="bold", color=PAL["text"])
    ex += 2.2

plt.tight_layout(pad=0.5)
plt.savefig("docs/framework_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("docs/framework_diagram.pdf", bbox_inches="tight", facecolor="white")
print("Saved: docs/framework_diagram.png + .pdf")
