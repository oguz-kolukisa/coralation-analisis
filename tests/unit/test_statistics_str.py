"""Tests for StatisticalResult.__str__ (the only uncovered line in statistics.py)."""
from src.analysis.statistics import StatisticalResult


class TestStatisticalResultStr:
    def test_confirmed_str(self):
        r = StatisticalResult(
            mean_delta=-0.3, std_delta=0.05, n_samples=10,
            t_statistic=-6.0, p_value=0.001, ci_lower=-0.35, ci_upper=-0.25,
            confidence_level=0.95, cohens_d=1.2,
            effect_size_interpretation="large",
            statistically_significant=True,
            practically_significant=True, confirmed=True,
        )
        s = str(r)
        assert "✓" in s
        assert "-0.300" in s
        assert "large" in s

    def test_not_confirmed_str(self):
        r = StatisticalResult(
            mean_delta=-0.01, std_delta=0.02, n_samples=3,
            t_statistic=-0.5, p_value=0.65, ci_lower=-0.05, ci_upper=0.03,
            confidence_level=0.95, cohens_d=0.1,
            effect_size_interpretation="negligible",
            statistically_significant=False,
            practically_significant=False, confirmed=False,
        )
        s = str(r)
        assert "✗" in s
