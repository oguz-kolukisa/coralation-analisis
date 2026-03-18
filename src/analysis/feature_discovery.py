"""
Feature discovery and analysis utilities.
Provides structured feature extraction and importance ranking.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Measured importance of a feature."""
    name: str
    category: str

    # From VLM analysis
    predicted_essential: float  # 1-5 scale
    predicted_spurious: float   # 1-5 scale

    # From actual experiments
    measured_delta: float       # Average confidence change when removed
    measured_std: float         # Standard deviation
    n_experiments: int          # Number of experiments

    # Statistical validation
    p_value: float = 1.0
    confirmed_shortcut: bool = False
    confirmed_essential: bool = False

    @property
    def importance_score(self) -> float:
        """Combined importance score based on measured delta."""
        return abs(self.measured_delta)

    @property
    def shortcut_likelihood(self) -> float:
        """
        Likelihood this is a shortcut (0-1).
        High if: high impact + predicted spurious + low essential
        """
        if not self.confirmed_shortcut:
            return 0.0

        # Normalize predicted scores to 0-1
        spurious_norm = self.predicted_spurious / 5.0
        essential_norm = self.predicted_essential / 5.0

        # Shortcut = high spurious, low essential, high impact
        impact_score = min(abs(self.measured_delta) * 2, 1.0)  # Cap at 1

        return (spurious_norm * 0.3 + (1 - essential_norm) * 0.3 + impact_score * 0.4)


class FeatureAnalyzer:
    """
    Analyzes and ranks features by their importance to the classifier.
    """

    def __init__(self):
        self.features: Dict[str, FeatureImportance] = {}

    def add_experiment(
        self,
        feature_name: str,
        category: str,
        delta: float,
        predicted_essential: float = 3.0,
        predicted_spurious: float = 3.0,
    ):
        """Add an experiment result for a feature."""
        if feature_name not in self.features:
            self.features[feature_name] = FeatureImportance(
                name=feature_name,
                category=category,
                predicted_essential=predicted_essential,
                predicted_spurious=predicted_spurious,
                measured_delta=delta,
                measured_std=0.0,
                n_experiments=1,
            )
        else:
            # Update running statistics
            f = self.features[feature_name]
            n = f.n_experiments
            old_mean = f.measured_delta

            # Welford's online algorithm for mean and variance
            new_n = n + 1
            new_mean = old_mean + (delta - old_mean) / new_n

            # Update variance (simplified)
            if n > 1:
                new_var = ((n - 1) * f.measured_std**2 + (delta - old_mean) * (delta - new_mean)) / (new_n - 1)
                f.measured_std = new_var ** 0.5

            f.measured_delta = new_mean
            f.n_experiments = new_n

    def rank_by_importance(self) -> List[FeatureImportance]:
        """Rank features by measured importance (absolute delta)."""
        return sorted(
            self.features.values(),
            key=lambda f: abs(f.measured_delta),
            reverse=True
        )

    def rank_by_shortcut_likelihood(self) -> List[FeatureImportance]:
        """Rank features by likelihood of being a shortcut."""
        return sorted(
            self.features.values(),
            key=lambda f: f.shortcut_likelihood,
            reverse=True
        )

    def get_confirmed_shortcuts(self) -> List[FeatureImportance]:
        """Get features confirmed as shortcuts."""
        return [f for f in self.features.values() if f.confirmed_shortcut]

    def get_confirmed_essential(self) -> List[FeatureImportance]:
        """Get features confirmed as essential."""
        return [f for f in self.features.values() if f.confirmed_essential]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of feature analysis."""
        all_features = list(self.features.values())

        if not all_features:
            return {"error": "No features analyzed"}

        shortcuts = self.get_confirmed_shortcuts()
        essential = self.get_confirmed_essential()

        return {
            "total_features": len(all_features),
            "confirmed_shortcuts": len(shortcuts),
            "confirmed_essential": len(essential),
            "top_shortcuts": [
                {"name": f.name, "delta": f.measured_delta, "likelihood": f.shortcut_likelihood}
                for f in self.rank_by_shortcut_likelihood()[:5]
            ],
            "top_essential": [
                {"name": f.name, "delta": f.measured_delta}
                for f in essential[:5]
            ],
            "by_category": self._group_by_category(),
        }

    def _group_by_category(self) -> Dict[str, List[str]]:
        """Group features by category."""
        groups: Dict[str, List[str]] = {}
        for f in self.features.values():
            if f.category not in groups:
                groups[f.category] = []
            groups[f.category].append(f.name)
        return groups
