"""Download Colored MNIST via the .h5 files in the repo.

The HF datasets builder for FrankCCCCC/colored_mnist_28 tries to fetch
~9000 individual PNGs and gets rate-limited. The repo also ships
training.h5 and testing.h5 — single files that bundle all images. We
download those and decode locally into a directory layout the framework
can ingest as an imagefolder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for var in ("HF_HOME", "HF_HUB_CACHE"):
    if var not in os.environ:
        sys.exit(f"ERROR: {var} not set. Source /coralation-analisis/setup/env.sh first.")

TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

from huggingface_hub import hf_hub_download   # noqa: E402
import h5py   # noqa: E402
import numpy as np   # noqa: E402
from PIL import Image   # noqa: E402

REPO = "FrankCCCCC/colored_mnist_28"
OUT_DIR = Path("/root/data/colored_mnist_28")


def download_h5(filename: str) -> Path:
    print(f"  -> downloading {filename}", flush=True)
    return Path(hf_hub_download(repo_id=REPO, filename=filename,
                                 repo_type="dataset", token=TOKEN))


def inspect(h5_path: Path) -> None:
    """Print the structure of an h5 file so we know what fields exist."""
    with h5py.File(h5_path, "r") as f:
        def _walk(name, obj):
            print(f"     {name}: {type(obj).__name__}", end="")
            if hasattr(obj, "shape"):
                print(f"  shape={obj.shape}  dtype={obj.dtype}", end="")
            print()
        f.visititems(_walk)


def decode_h5_to_imagefolder(h5_path: Path, split_name: str) -> None:
    """Write images from .h5 into <out>/<split>/<digit>/<idx>.png imagefolder layout."""
    out = OUT_DIR / split_name
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        # Discover field names — try common shapes
        keys = list(f.keys())
        print(f"  -> {h5_path.name}: top-level keys = {keys}")
        # Heuristics for the data
        img_key = None
        lbl_key = None
        for cand in ("images", "image", "X", "x_data", "data"):
            if cand in keys:
                img_key = cand; break
        for cand in ("labels", "label", "y", "y_data", "digit", "digits"):
            if cand in keys:
                lbl_key = cand; break
        if img_key is None or lbl_key is None:
            raise RuntimeError(f"Could not find image/label fields in {keys}")
        imgs = f[img_key][:]
        labels = f[lbl_key][:]
        # Spurious attribute (color) — preserve in filename for later analysis
        colors = f["colors"][:] if "colors" in keys else None
        color_names = {0: "red", 1: "green", 2: "blue"}
        print(f"  -> {len(imgs)} images, label range {labels.min()}-{labels.max()}")
        for i, (im, lbl) in enumerate(zip(imgs, labels)):
            digit = int(lbl)
            digit_dir = out / str(digit)
            digit_dir.mkdir(exist_ok=True)
            arr = im
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            # Channel order: HWC vs CHW
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            color_tag = color_names.get(int(colors[i]), "x") if colors is not None else "x"
            Image.fromarray(arr).save(digit_dir / f"{color_tag}_{i:06d}.png")
        print(f"  -> wrote {len(imgs)} files to {out}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== Inspecting and decoding training.h5 ==")
    train_h5 = download_h5("training.h5")
    inspect(train_h5)
    decode_h5_to_imagefolder(train_h5, "train")
    print()

    print("== Inspecting and decoding testing.h5 ==")
    test_h5 = download_h5("testing.h5")
    inspect(test_h5)
    decode_h5_to_imagefolder(test_h5, "test")

    print()
    print(f"DONE. Imagefolder layout at {OUT_DIR}/")
    for d in sorted(OUT_DIR.iterdir()):
        n_total = sum(1 for _ in d.rglob("*.png"))
        print(f"  {d.name}: {n_total} files across {len(list(d.iterdir()))} class dirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
