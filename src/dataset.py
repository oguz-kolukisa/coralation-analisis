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
            self._ds = load_dataset(
                dataset_name,
                split=split,
                token=hf_token,
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
        """Find the numeric label index for a class name (case-insensitive substring)."""
        names = self.get_label_names()
        name_lower = class_name.lower()
        for i, n in enumerate(names):
            if name_lower in n.lower() or n.lower() in name_lower:
                return i
        return None

    def find_label_indices(self, class_name: str) -> list[int]:
        """Return all matching label indices (for classes with multiple HF entries)."""
        names = self.get_label_names()
        name_lower = class_name.lower()
        return [i for i, n in enumerate(names)
                if name_lower in n.lower() or n.lower() in name_lower]

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
