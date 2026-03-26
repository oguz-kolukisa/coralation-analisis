"""
Statistical validation for shortcut/bias detection.
Provides rigorous statistical tests instead of simple threshold-based confirmation.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Results from statistical validation."""
    # Basic statistics
    mean_delta: float
    std_delta: float
    n_samples: int

    # T-test results
    t_statistic: float
    p_value: float

    # Confidence interval
    ci_lower: float
    ci_upper: float

    # Effect size
    cohens_d: float
    effect_size_interpretation: str  # "negligible", "small", "medium", "large"

    # Final decision
    statistically_significant: bool
    practically_significant: bool  # Effect size > threshold
    confirmed: bool  # Both statistically and practically significant

    # Field with default must come last
    confidence_level: float = 0.95

    def __str__(self) -> str:
        sig = "✓" if self.confirmed else "✗"
        return (
            f"{sig} Δ={self.mean_delta:+.3f}±{self.std_delta:.3f} "
            f"(p={self.p_value:.4f}, d={self.cohens_d:.2f} [{self.effect_size_interpretation}])"
        )


class StatisticalValidator:
    """
    Validates whether observed confidence changes are statistically significant.

    Uses:
    - One-sample t-test: Is mean delta significantly different from 0?
    - Cohen's d: Is the effect size meaningful?
    - Confidence intervals: Range of plausible true effect
    """

    def __init__(
        self,
        alpha: float = 0.05,           # Significance level
        min_effect_size: float = 0.5,  # Minimum Cohen's d for practical significance
        min_samples: int = 3,          # Minimum samples for valid test
    ):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.min_samples = min_samples

    def validate(
        self,
        deltas: List[float],
        expected_direction: str = "negative",  # "negative" for feature removal, "positive" for addition
    ) -> StatisticalResult:
        """
        Validate if the observed deltas are statistically significant.

        Args:
            deltas: List of confidence changes (edited - original)
            expected_direction: Expected direction of change
                - "negative": feature removal should decrease confidence
                - "positive": feature addition should increase confidence
                - "any": any significant change

        Returns:
            StatisticalResult with all test results
        """
        n = len(deltas)

        if n < self.min_samples:
            logger.warning(f"Insufficient samples ({n} < {self.min_samples}) for statistical test")
            return self._insufficient_samples_result(deltas)

        arr = np.array(deltas)
        mean_delta = float(np.mean(arr))
        std_delta = float(np.std(arr, ddof=1))  # Sample std

        # One-sample t-test: is mean significantly different from 0?
        t_stat, p_value_two_tailed = stats.ttest_1samp(arr, 0)

        # Convert to one-tailed if we have a directional hypothesis
        if expected_direction == "negative":
            # We expect negative delta, so use lower tail
            p_value = p_value_two_tailed / 2 if t_stat < 0 else 1 - p_value_two_tailed / 2
        elif expected_direction == "positive":
            # We expect positive delta, so use upper tail
            p_value = p_value_two_tailed / 2 if t_stat > 0 else 1 - p_value_two_tailed / 2
        else:
            p_value = p_value_two_tailed

        # Confidence interval
        sem = std_delta / np.sqrt(n)  # Standard error of mean
        ci_margin = stats.t.ppf(1 - self.alpha / 2, n - 1) * sem
        ci_lower = mean_delta - ci_margin
        ci_upper = mean_delta + ci_margin

        # Effect size: Cohen's d
        cohens_d = abs(mean_delta) / std_delta if std_delta > 0 else 0
        effect_interpretation = self._interpret_cohens_d(cohens_d)

        # Statistical significance
        statistically_significant = bool(p_value < self.alpha)

        # Practical significance (effect size)
        practically_significant = bool(cohens_d >= self.min_effect_size)

        # Direction check
        direction_correct = (
            (expected_direction == "negative" and mean_delta < 0) or
            (expected_direction == "positive" and mean_delta > 0) or
            (expected_direction == "any")
        )

        # Final confirmation: must be statistically significant, practically significant, and correct direction
        confirmed = bool(statistically_significant and practically_significant and direction_correct)

        return StatisticalResult(
            mean_delta=mean_delta,
            std_delta=std_delta,
            n_samples=n,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=1 - self.alpha,
            cohens_d=float(cohens_d),
            effect_size_interpretation=effect_interpretation,
            statistically_significant=statistically_significant,
            practically_significant=practically_significant,
            confirmed=confirmed,
        )

    def validate_multiple(
        self,
        all_deltas: List[List[float]],
        expected_direction: str = "negative",
        correction: str = "bonferroni",  # Multiple comparison correction
    ) -> List[StatisticalResult]:
        """
        Validate multiple hypotheses with correction for multiple comparisons.

        Args:
            all_deltas: List of delta lists, one per hypothesis
            expected_direction: Expected direction
            correction: "bonferroni", "holm", or "none"

        Returns:
            List of StatisticalResults with corrected p-values
        """
        results = [self.validate(deltas, expected_direction) for deltas in all_deltas]

        if correction == "none" or len(results) <= 1:
            return results

        # Apply multiple comparison correction
        p_values = [r.p_value for r in results]

        if correction == "bonferroni":
            corrected_alpha = self.alpha / len(results)
            for r in results:
                r.statistically_significant = bool(r.p_value < corrected_alpha)
                r.confirmed = bool(r.statistically_significant and r.practically_significant)

        elif correction == "holm":
            # Holm-Bonferroni method (step-down)
            sorted_indices = np.argsort(p_values)
            n = len(p_values)
            for rank, idx in enumerate(sorted_indices):
                corrected_alpha = self.alpha / (n - rank)
                results[idx].statistically_significant = bool(p_values[idx] < corrected_alpha)
                results[idx].confirmed = bool(
                    results[idx].statistically_significant and
                    results[idx].practically_significant
                )

        return results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _insufficient_samples_result(self, deltas: List[float]) -> StatisticalResult:
        """Return a result for insufficient samples."""
        arr = np.array(deltas) if deltas else np.array([0])
        return StatisticalResult(
            mean_delta=float(np.mean(arr)),
            std_delta=float(np.std(arr)) if len(arr) > 1 else 0,
            n_samples=len(deltas),
            t_statistic=0,
            p_value=1.0,
            ci_lower=float(np.min(arr)) if len(arr) > 0 else 0,
            ci_upper=float(np.max(arr)) if len(arr) > 0 else 0,
            cohens_d=0,
            effect_size_interpretation="insufficient_data",
            statistically_significant=False,
            practically_significant=False,
            confirmed=False,
        )


def calculate_power(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
) -> float:
    """
    Calculate statistical power for a one-sample t-test.

    Power = probability of detecting an effect if it exists.

    Args:
        effect_size: Expected Cohen's d
        n_samples: Number of samples
        alpha: Significance level

    Returns:
        Statistical power (0-1)
    """
    from scipy.stats import nct

    # Non-centrality parameter
    nc = effect_size * np.sqrt(n_samples)

    # Critical value
    critical_t = stats.t.ppf(1 - alpha, n_samples - 1)

    # Power = 1 - beta = P(reject H0 | H1 is true)
    power = 1 - nct.cdf(critical_t, n_samples - 1, nc)

    return float(power)


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """
    Calculate required sample size for desired power.

    Args:
        effect_size: Expected Cohen's d
        power: Desired statistical power
        alpha: Significance level

    Returns:
        Required number of samples
    """
    from scipy.optimize import brentq

    def power_diff(n):
        return calculate_power(effect_size, int(n), alpha) - power

    try:
        n = brentq(power_diff, 2, 1000)
        return int(np.ceil(n))
    except ValueError:
        return 3  # Minimum
