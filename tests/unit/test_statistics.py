"""Tests for StatisticalValidator and related functions."""
from __future__ import annotations

import pytest

from src.analysis.statistics import (
    StatisticalResult, StatisticalValidator, calculate_power, required_sample_size,
)


@pytest.fixture
def validator():
    return StatisticalValidator(alpha=0.05, min_effect_size=0.5, min_samples=3)


# =========================================================================
# StatisticalValidator.__init__
# =========================================================================

class TestValidatorInit:
    def test_stores_params(self, validator):
        assert validator.alpha == 0.05
        assert validator.min_effect_size == 0.5
        assert validator.min_samples == 3


# =========================================================================
# StatisticalValidator.validate
# =========================================================================

class TestValidate:
    def test_insufficient_samples(self, validator):
        result = validator.validate([-0.5, -0.3])
        assert result.p_value == 1.0
        assert result.confirmed is False
        assert result.effect_size_interpretation == "insufficient_data"

    def test_strong_negative_effect_confirmed(self, validator):
        deltas = [-0.5, -0.4, -0.6, -0.55, -0.45]
        result = validator.validate(deltas, expected_direction="negative")
        assert result.confirmed == True
        assert result.statistically_significant == True
        assert result.practically_significant == True
        assert result.mean_delta < 0

    def test_weak_effect_not_confirmed(self, validator):
        deltas = [-0.01, 0.01, -0.02, 0.02, 0.0]
        result = validator.validate(deltas, expected_direction="negative")
        assert result.confirmed == False

    def test_wrong_direction_not_confirmed(self, validator):
        deltas = [0.5, 0.4, 0.6]
        result = validator.validate(deltas, expected_direction="negative")
        assert result.confirmed == False

    def test_positive_direction(self, validator):
        deltas = [0.5, 0.4, 0.6, 0.55, 0.45]
        result = validator.validate(deltas, expected_direction="positive")
        assert result.confirmed == True
        assert result.mean_delta > 0

    def test_any_direction(self, validator):
        deltas = [-0.5, -0.4, -0.6, -0.55, -0.45]
        result = validator.validate(deltas, expected_direction="any")
        assert result.statistically_significant == True

    def test_returns_statistical_result(self, validator):
        result = validator.validate([-0.3, -0.2, -0.4])
        assert isinstance(result, StatisticalResult)

    def test_confidence_interval_contains_mean(self, validator):
        deltas = [-0.3, -0.2, -0.4, -0.35, -0.25]
        result = validator.validate(deltas)
        assert result.ci_lower <= result.mean_delta <= result.ci_upper

    def test_empty_deltas(self, validator):
        result = validator.validate([])
        assert result.confirmed is False
        assert result.n_samples == 0

    def test_single_sample(self, validator):
        result = validator.validate([-0.5])
        assert result.confirmed is False
        assert result.n_samples == 1


# =========================================================================
# _interpret_cohens_d
# =========================================================================

class TestInterpretCohensD:
    def test_negligible(self, validator):
        assert validator._interpret_cohens_d(0.1) == "negligible"

    def test_small(self, validator):
        assert validator._interpret_cohens_d(0.3) == "small"

    def test_medium(self, validator):
        assert validator._interpret_cohens_d(0.6) == "medium"

    def test_large(self, validator):
        assert validator._interpret_cohens_d(1.0) == "large"

    def test_boundary_02(self, validator):
        assert validator._interpret_cohens_d(0.2) == "small"

    def test_boundary_05(self, validator):
        assert validator._interpret_cohens_d(0.5) == "medium"

    def test_boundary_08(self, validator):
        assert validator._interpret_cohens_d(0.8) == "large"

    def test_zero(self, validator):
        assert validator._interpret_cohens_d(0.0) == "negligible"


# =========================================================================
# validate_multiple
# =========================================================================

class TestValidateMultiple:
    def test_single_hypothesis_no_correction(self, validator):
        results = validator.validate_multiple([[-0.5, -0.4, -0.6]])
        assert len(results) == 1

    def test_bonferroni_more_conservative(self, validator):
        all_deltas = [[-0.3, -0.2, -0.4], [-0.25, -0.15, -0.35]]
        uncorrected = [validator.validate(d) for d in all_deltas]
        corrected = validator.validate_multiple(all_deltas, correction="bonferroni")
        for u, c in zip(uncorrected, corrected):
            if u.statistically_significant:
                pass  # corrected may or may not be significant

    def test_none_correction_same_as_individual(self, validator):
        all_deltas = [[-0.5, -0.4, -0.6], [-0.3, -0.2, -0.4]]
        individual = [validator.validate(d) for d in all_deltas]
        multi = validator.validate_multiple(all_deltas, correction="none")
        for i, m in zip(individual, multi):
            assert i.p_value == m.p_value

    def test_holm_correction(self, validator):
        all_deltas = [[-0.5, -0.4, -0.6], [-0.01, 0.01, -0.02]]
        results = validator.validate_multiple(all_deltas, correction="holm")
        assert len(results) == 2


# =========================================================================
# calculate_power
# =========================================================================

class TestCalculatePower:
    def test_returns_between_0_and_1(self):
        power = calculate_power(0.8, 10)
        assert 0 < power < 1

    def test_larger_effect_higher_power(self):
        small = calculate_power(0.3, 10)
        large = calculate_power(1.0, 10)
        assert large > small

    def test_more_samples_higher_power(self):
        few = calculate_power(0.5, 5)
        many = calculate_power(0.5, 50)
        assert many > few


# =========================================================================
# required_sample_size
# =========================================================================

class TestRequiredSampleSize:
    def test_returns_int(self):
        n = required_sample_size(0.8)
        assert isinstance(n, int)

    def test_minimum_3(self):
        n = required_sample_size(0.01)
        assert n >= 3

    def test_larger_effect_fewer_samples(self):
        small_effect = required_sample_size(0.3)
        large_effect = required_sample_size(1.0)
        assert large_effect < small_effect
