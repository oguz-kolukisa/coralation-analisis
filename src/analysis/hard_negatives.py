"""
Hard negative mining for more rigorous bias detection.
Finds images that are almost classified as the target class - these are
the most informative for detecting shortcuts.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class HardNegative:
    """A hard negative sample."""
    image: Image.Image
    true_label: str
    target_class_confidence: float  # How confident the model is this is the target class
    predicted_label: str
    predicted_confidence: float


class HardNegativeMiner:
    """
    Finds hard negative samples - images that are almost misclassified as the target.

    These are more informative than random negatives because:
    1. They share visual features with the target class
    2. Small changes might flip the classification
    3. They reveal what the model finds "similar"
    """

    def __init__(
        self,
        classifier,
        min_confidence: float = 0.05,  # Minimum confidence for target class
        max_confidence: float = 0.40,  # Maximum confidence (not already classified as target)
    ):
        self.classifier = classifier
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def mine(
        self,
        images: Iterator[tuple[Image.Image, str]],
        target_class: str,
        n_samples: int = 10,
        max_scan: int = 5000,
    ) -> list[HardNegative]:
        """
        Mine hard negatives from a stream of images.

        Args:
            images: Iterator of (image, label) tuples
            target_class: The class we're analyzing
            n_samples: Number of hard negatives to find
            max_scan: Maximum images to scan

        Returns:
            List of HardNegative samples, sorted by target confidence (highest first)
        """
        hard_negatives: list[HardNegative] = []

        for i, (img, true_label) in enumerate(images):
            if i >= max_scan:
                break

            # Skip if this IS the target class
            if true_label.lower() == target_class.lower():
                continue

            # Get confidence for target class
            target_conf = self.classifier.get_class_confidence(img, target_class)

            # Check if it's in the "hard negative" range
            if self.min_confidence <= target_conf <= self.max_confidence:
                # Also get the predicted class
                result = self.classifier.predict(img, compute_gradcam=False)

                hard_negatives.append(HardNegative(
                    image=img,
                    true_label=true_label,
                    target_class_confidence=target_conf,
                    predicted_label=result.label_name,
                    predicted_confidence=result.confidence,
                ))

                logger.debug(
                    "Found hard negative: %s (target conf: %.2f, predicted: %s)",
                    true_label, target_conf, result.label_name
                )

                if len(hard_negatives) >= n_samples:
                    break

        # Sort by target confidence (highest first - most "confusable")
        hard_negatives.sort(key=lambda x: x.target_class_confidence, reverse=True)

        logger.info(
            "Mined %d hard negatives for '%s' (scanned %d images)",
            len(hard_negatives), target_class, min(i + 1, max_scan)
        )

        return hard_negatives

    def categorize_negatives(
        self,
        images: Iterator[tuple[Image.Image, str]],
        target_class: str,
        max_scan: int = 2000,
    ) -> dict[str, list[HardNegative]]:
        """
        Categorize negatives by their "hardness" level.

        Returns:
            Dict with categories:
            - "very_hard": 0.30-0.40 confidence (almost misclassified)
            - "hard": 0.15-0.30 confidence
            - "medium": 0.05-0.15 confidence
            - "easy": <0.05 confidence
        """
        categories = {
            "very_hard": [],  # 0.30-0.40
            "hard": [],       # 0.15-0.30
            "medium": [],     # 0.05-0.15
            "easy": [],       # <0.05
        }

        for i, (img, true_label) in enumerate(images):
            if i >= max_scan:
                break

            if true_label.lower() == target_class.lower():
                continue

            target_conf = self.classifier.get_class_confidence(img, target_class)
            result = self.classifier.predict(img, compute_gradcam=False)

            hn = HardNegative(
                image=img,
                true_label=true_label,
                target_class_confidence=target_conf,
                predicted_label=result.label_name,
                predicted_confidence=result.confidence,
            )

            if target_conf >= 0.30:
                categories["very_hard"].append(hn)
            elif target_conf >= 0.15:
                categories["hard"].append(hn)
            elif target_conf >= 0.05:
                categories["medium"].append(hn)
            else:
                categories["easy"].append(hn)

        # Sort each category
        for cat in categories.values():
            cat.sort(key=lambda x: x.target_class_confidence, reverse=True)

        return categories
