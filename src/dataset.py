"""
HuggingFace dataset sampler for image classification datasets.
Supports ImageNet, CIFAR-100, and other datasets with similar structure.
"""
from __future__ import annotations
import contextlib
import logging
import os
import random
import sys
import time
from typing import Iterator

from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr and tqdm during dataset loading."""
    import tqdm

    # Save original
    old_stdout = sys.stdout
    old_stderr = sys.stderr

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


_SPLIT_ONLY_PATTERNS: dict[str, str] = {
    # The parquet builder for these datasets lists train+val+test data_files
    # at config time, so load_dataset(..., split=X) downloads ALL splits before
    # slicing. Constraining data_files to the requested split avoids that.
    "ILSVRC/imagenet-1k": "data/{split}-*.parquet",
}


def _split_only_data_files(dataset_name: str, split: str) -> dict | None:
    """Return data_files mapping that constrains download to one split, or None."""
    pattern = _SPLIT_ONLY_PATTERNS.get(dataset_name)
    if pattern is None:
        return None
    return {split: pattern.format(split=split)}


def _is_local_path(name: str) -> bool:
    """True if ``name`` is a filesystem path rather than an HF dataset id."""
    return name.startswith(("/", "./", "../"))


def _resolve_imagefolder_root(path: str) -> str:
    """Pick the imagefolder root for ``<root>/<class>/*`` layout.

    NICO++ ships as ``<root>/<domain>/<class>/*`` — we flatten that into
    one effective imagefolder by symlinking ``<root>/<domain>/<class>`` →
    ``<flatten>/<class>`` (merging across domains). Idempotent: re-uses an
    existing flattened root.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Local dataset path is not a directory: {p}")
    if _is_two_level_layout(p):
        return _flatten_two_level(p)
    return str(p)


def _is_two_level_layout(root) -> bool:
    """True if every immediate child of root is itself a class-dir-of-dirs."""
    from pathlib import Path
    children = [c for c in Path(root).iterdir() if c.is_dir()]
    if not children:
        return False
    # 'splits' is a metadata folder in NICO++ — ignore for this heuristic
    children = [c for c in children if c.name != "splits"]
    if not children:
        return False
    for c in children:
        sub = [s for s in c.iterdir() if s.is_dir()]
        if not sub:
            return False
    return True


def _flatten_two_level(root) -> str:
    """Merge ``<root>/<domain>/<class>`` → ``<flat>/<class>`` via symlinks.

    The flattened root is cached as ``<root>/_flat_imagefolder`` and is
    rebuilt only if missing.
    """
    from pathlib import Path
    flat = Path(root) / "_flat_imagefolder"
    if flat.exists():
        return str(flat)
    flat.mkdir()
    for domain_dir in Path(root).iterdir():
        if not domain_dir.is_dir() or domain_dir.name in ("_flat_imagefolder", "splits"):
            continue
        for class_dir in domain_dir.iterdir():
            if not class_dir.is_dir():
                continue
            target_class = flat / class_dir.name
            target_class.mkdir(exist_ok=True)
            for img in class_dir.iterdir():
                if img.is_file():
                    dst = target_class / f"{domain_dir.name}_{img.name}"
                    if not dst.exists():
                        dst.symlink_to(img.resolve())
    return str(flat)


def _matches_label(query: str, label: str) -> bool:
    """Check if query matches label by exact synonym matching."""
    query_lower = query.lower().strip()
    label_lower = label.lower().strip()
    if query_lower == label_lower:
        return True
    synonyms = [s.strip() for s in label_lower.split(",")]
    return query_lower in synonyms


class ImageNetSampler:
    """
    Streams samples from HuggingFace datasets without downloading the full dataset.
    Supports various datasets including ImageNet, CIFAR-100, etc.
    Requires HF_TOKEN if the dataset is gated.
    """

    def __init__(self, dataset_name: str = "ILSVRC/imagenet-1k",
                 split: str = "validation", hf_token: str | None = None,
                 max_scan: int = 10000, seed: int | None = None):
        from datasets import load_dataset, disable_progress_bar
        from huggingface_hub import login
        import logging as _logging

        # Silence datasets library
        disable_progress_bar()
        _logging.getLogger("datasets").setLevel(_logging.ERROR)
        _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

        # Ensure we're logged in if token provided
        if hf_token:
            with suppress_output():
                login(token=hf_token, add_to_git_credential=False)

        logger.debug("Loading dataset %s / %s", dataset_name, split)
        with suppress_output():
            if _is_local_path(dataset_name):
                # Local folder: use imagefolder loader. NICO++ has nested
                # <domain>/<class>/* — flatten via _resolve_imagefolder_root.
                root = _resolve_imagefolder_root(dataset_name)
                self._ds = load_dataset(
                    "imagefolder", data_dir=root, split="train",
                    verification_mode="no_checks",
                )
            else:
                self._ds = load_dataset(
                    dataset_name,
                    data_files=_split_only_data_files(dataset_name, split),
                    split=split,
                    token=hf_token,
                    verification_mode="no_checks",
                )
        self._max_scan = max_scan
        self._seed = seed if seed is not None else int(time.time())

        # Auto-detect label field name (varies by dataset)
        features = self._ds.features
        if "fine_label" in features:  # CIFAR-100
            self._label_field = "fine_label"
        elif "label" in features:  # ImageNet, most others
            self._label_field = "label"
        else:
            raise ValueError(f"Could not find label field in dataset. Available: {list(features.keys())}")

        # Auto-detect image field name
        if "img" in features:  # CIFAR-100
            self._image_field = "img"
        elif "image" in features:  # ImageNet, most others
            self._image_field = "image"
        else:
            raise ValueError(f"Could not find image field in dataset. Available: {list(features.keys())}")

        logger.debug("Using label field: %s, image field: %s", self._label_field, self._image_field)

        # Cache label→name mapping
        self._label_names: list[str] | None = None
        # Cache for sampled data to avoid re-scanning
        self._positive_cache: dict[str, list[tuple[Image.Image, str]]] = {}
        self._negative_cache: dict[str, list[tuple[Image.Image, str]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_label_names(self) -> list[str]:
        """Return the class names for this dataset."""
        if self._label_names is None:
            features = self._ds.features
            self._label_names = features[self._label_field].names
        return self._label_names

    def find_label_index(self, class_name: str) -> int | None:
        """Find the numeric label index for a class name (exact synonym match)."""
        names = self.get_label_names()
        for i, label in enumerate(names):
            if _matches_label(class_name, label):
                return i
        return None

    def find_label_indices(self, class_name: str) -> list[int]:
        """Return all matching label indices (exact synonym match)."""
        names = self.get_label_names()
        return [i for i, label in enumerate(names)
                if _matches_label(class_name, label)]

    def sample_positive(self, class_name: str, n: int = 5) -> list[tuple[Image.Image, str]]:
        """Return up to n images that belong to the given class."""
        # Check cache first
        cache_key = class_name.lower()
        if cache_key in self._positive_cache:
            cached = self._positive_cache[cache_key]
            if len(cached) >= n:
                return cached[:n]

        indices = self.find_label_indices(class_name)
        if not indices:
            logger.warning("No label found for class '%s'", class_name)
            return []

        idx_set = set(indices)
        results: list[tuple[Image.Image, str]] = []
        label_names = self.get_label_names()

        shuffled = self._ds.shuffle(seed=self._seed)
        scan_limit = min(self._max_scan, len(shuffled))
        for i in range(scan_limit):
            item = shuffled[i]
            label_idx = item[self._label_field]
            if label_idx in idx_set:
                img = item[self._image_field]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                results.append((img.convert("RGB"), label_names[label_idx]))
                if len(results) >= n:
                    break

        # Cache results
        self._positive_cache[cache_key] = results
        logger.debug("Sampled %d positive images for class '%s'", len(results), class_name)
        return results

    def sample_negative(self, class_name: str, n: int = 5) -> list[tuple[Image.Image, str]]:
        """Return up to n images that do NOT belong to the given class."""
        # Check cache first
        cache_key = class_name.lower()
        if cache_key in self._negative_cache:
            cached = self._negative_cache[cache_key]
            if len(cached) >= n:
                return cached[:n]

        indices = set(self.find_label_indices(class_name))
        results: list[tuple[Image.Image, str]] = []
        label_names = self.get_label_names()

        # Use different seed for negatives to get variety
        shuffled = self._ds.shuffle(seed=self._seed + 1000)
        scan_limit = min(self._max_scan, len(shuffled))
        for i in range(scan_limit):
            item = shuffled[i]
            label_idx = item[self._label_field]
            if label_idx not in indices:
                img = item[self._image_field]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                results.append((img.convert("RGB"), label_names[label_idx]))
                if len(results) >= n:
                    break

        # Cache results
        self._negative_cache[cache_key] = results
        logger.debug("Sampled %d negative images for class '%s'", len(results), class_name)
        return results

    def sample_from_classes(
        self,
        class_names: list[str],
        n_per_class: int = 1,
    ) -> list[tuple[Image.Image, str]]:
        """
        Sample images from specific classes (for VLM-selected confusing classes).

        Args:
            class_names: List of class names to sample from
            n_per_class: Number of samples per class

        Returns:
            List of (image, label) tuples
        """
        results: list[tuple[Image.Image, str]] = []
        label_names = self.get_label_names()

        for class_name in class_names:
            indices = set(self.find_label_indices(class_name))
            if not indices:
                logger.warning("Class '%s' not found in dataset", class_name)
                continue

            # Sample from this specific class
            count = 0
            shuffled = self._ds.shuffle(seed=self._seed + hash(class_name) % 10000)
            scan_limit = min(self._max_scan, len(shuffled))

            for i in range(scan_limit):
                item = shuffled[i]
                label_idx = item[self._label_field]
                if label_idx in indices:
                    img = item[self._image_field]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    results.append((img.convert("RGB"), label_names[label_idx]))
                    count += 1
                    if count >= n_per_class:
                        break

        logger.debug("Sampled %d images from %d specified classes",
                    len(results), len(class_names))
        return results
