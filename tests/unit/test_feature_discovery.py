"""Tests for src/analysis/feature_discovery.py — FeatureImportance and FeatureAnalyzer."""
from __future__ import annotations

import pytest

from src.analysis.feature_discovery import FeatureAnalyzer, FeatureImportance


# ============================================================================
# FeatureImportance dataclass
# ============================================================================

class TestImportanceScore:
    def test_positive_delta(self):
        f = FeatureImportance("a", "ctx", 1.0, 4.0, 0.3, 0.0, 1)
        assert f.importance_score == pytest.approx(0.3)

    def test_negative_delta(self):
        f = FeatureImportance("a", "ctx", 1.0, 4.0, -0.5, 0.0, 1)
        assert f.importance_score == pytest.approx(0.5)

    def test_zero_delta(self):
        f = FeatureImportance("a", "ctx", 1.0, 4.0, 0.0, 0.0, 1)
        assert f.importance_score == 0.0


class TestShortcutLikelihood:
    def test_not_confirmed_returns_zero(self):
        f = FeatureImportance("a", "ctx", 1.0, 5.0, -0.4, 0.0, 1, confirmed_shortcut=False)
        assert f.shortcut_likelihood == 0.0

    def test_confirmed_high_spurious_low_essential(self):
        f = FeatureImportance("a", "ctx", 1.0, 5.0, -0.8, 0.0, 1, confirmed_shortcut=True)
        score = f.shortcut_likelihood
        assert 0.0 < score <= 1.0

    def test_confirmed_low_spurious_high_essential(self):
        f = FeatureImportance("a", "ctx", 5.0, 1.0, -0.1, 0.0, 1, confirmed_shortcut=True)
        score = f.shortcut_likelihood
        assert score < 0.5  # Should be low likelihood

    def test_high_impact_increases_score(self):
        low_impact = FeatureImportance("a", "ctx", 1.0, 5.0, -0.1, 0.0, 1, confirmed_shortcut=True)
        high_impact = FeatureImportance("a", "ctx", 1.0, 5.0, -0.9, 0.0, 1, confirmed_shortcut=True)
        assert high_impact.shortcut_likelihood > low_impact.shortcut_likelihood

    def test_impact_capped_at_one(self):
        f = FeatureImportance("a", "ctx", 0.0, 5.0, -5.0, 0.0, 1, confirmed_shortcut=True)
        assert f.shortcut_likelihood <= 1.0

    def test_all_max_values(self):
        f = FeatureImportance("a", "ctx", 0.0, 5.0, -1.0, 0.0, 1, confirmed_shortcut=True)
        # spurious=5/5=1.0 → 0.3, essential=0/5=0 → (1-0)*0.3=0.3, impact=min(2,1)=1 → 0.4
        assert f.shortcut_likelihood == pytest.approx(1.0)


# ============================================================================
# FeatureAnalyzer — add_experiment
# ============================================================================

class TestAddExperiment:
    def test_first_experiment_creates_feature(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "object_part", -0.3)
        assert "ears" in fa.features
        assert fa.features["ears"].n_experiments == 1
        assert fa.features["ears"].measured_delta == pytest.approx(-0.3)

    def test_second_experiment_updates_mean(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "object_part", -0.3)
        fa.add_experiment("ears", "object_part", -0.5)
        assert fa.features["ears"].n_experiments == 2
        assert fa.features["ears"].measured_delta == pytest.approx(-0.4)

    def test_three_experiments_updates_std(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "object_part", -0.1)
        fa.add_experiment("ears", "object_part", -0.5)
        fa.add_experiment("ears", "object_part", -0.9)
        assert fa.features["ears"].n_experiments == 3
        assert fa.features["ears"].measured_std > 0

    def test_single_experiment_std_is_zero(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "object_part", -0.3)
        assert fa.features["ears"].measured_std == 0.0

    def test_two_experiments_std_still_zero(self):
        """Welford update only kicks in for n>1 which means n_experiments>=3."""
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "object_part", -0.3)
        fa.add_experiment("ears", "object_part", -0.5)
        # std stays 0 because the variance update only runs when n>1 (prior count)
        assert fa.features["ears"].measured_std == 0.0

    def test_predicted_scores_stored(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("bg", "context", -0.2, predicted_essential=1.0, predicted_spurious=4.5)
        assert fa.features["bg"].predicted_essential == 1.0
        assert fa.features["bg"].predicted_spurious == 4.5

    def test_multiple_features(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "object_part", -0.3)
        fa.add_experiment("bg", "context", -0.1)
        assert len(fa.features) == 2


# ============================================================================
# FeatureAnalyzer — ranking
# ============================================================================

class TestRankByImportance:
    def test_sorted_by_abs_delta(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("small", "a", -0.1)
        fa.add_experiment("big", "b", -0.5)
        fa.add_experiment("medium", "c", 0.3)
        ranked = fa.rank_by_importance()
        assert ranked[0].name == "big"
        assert ranked[-1].name == "small"

    def test_empty_returns_empty(self):
        fa = FeatureAnalyzer()
        assert fa.rank_by_importance() == []


class TestRankByShortcutLikelihood:
    def test_confirmed_shortcut_ranks_higher(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("bg", "context", -0.5, predicted_spurious=5.0, predicted_essential=1.0)
        fa.add_experiment("ears", "object_part", -0.5, predicted_spurious=1.0, predicted_essential=5.0)
        fa.features["bg"].confirmed_shortcut = True
        ranked = fa.rank_by_shortcut_likelihood()
        assert ranked[0].name == "bg"

    def test_empty_returns_empty(self):
        fa = FeatureAnalyzer()
        assert fa.rank_by_shortcut_likelihood() == []


# ============================================================================
# FeatureAnalyzer — filters
# ============================================================================

class TestGetConfirmedShortcuts:
    def test_returns_only_confirmed(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("bg", "context", -0.3)
        fa.add_experiment("ears", "part", -0.4)
        fa.features["bg"].confirmed_shortcut = True
        shortcuts = fa.get_confirmed_shortcuts()
        assert len(shortcuts) == 1
        assert shortcuts[0].name == "bg"

    def test_none_confirmed(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "part", -0.3)
        assert fa.get_confirmed_shortcuts() == []


class TestGetConfirmedEssential:
    def test_returns_only_confirmed(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "part", -0.4)
        fa.add_experiment("bg", "context", -0.1)
        fa.features["ears"].confirmed_essential = True
        essential = fa.get_confirmed_essential()
        assert len(essential) == 1
        assert essential[0].name == "ears"

    def test_none_confirmed(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "part", -0.3)
        assert fa.get_confirmed_essential() == []


# ============================================================================
# FeatureAnalyzer — summary
# ============================================================================

class TestGetSummary:
    def test_empty_features_returns_error(self):
        fa = FeatureAnalyzer()
        summary = fa.get_summary()
        assert "error" in summary

    def test_with_features_returns_all_keys(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "part", -0.3)
        fa.add_experiment("bg", "context", -0.1)
        fa.features["bg"].confirmed_shortcut = True
        summary = fa.get_summary()
        assert summary["total_features"] == 2
        assert summary["confirmed_shortcuts"] == 1
        assert summary["confirmed_essential"] == 0
        assert "top_shortcuts" in summary
        assert "by_category" in summary

    def test_by_category_grouping(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("ears", "part", -0.3)
        fa.add_experiment("eyes", "part", -0.2)
        fa.add_experiment("bg", "context", -0.1)
        summary = fa.get_summary()
        cats = summary["by_category"]
        assert set(cats["part"]) == {"ears", "eyes"}
        assert cats["context"] == ["bg"]


# ============================================================================
# FeatureAnalyzer — _group_by_category
# ============================================================================

class TestGroupByCategory:
    def test_groups_correctly(self):
        fa = FeatureAnalyzer()
        fa.add_experiment("a", "cat1", -0.1)
        fa.add_experiment("b", "cat1", -0.2)
        fa.add_experiment("c", "cat2", -0.3)
        groups = fa._group_by_category()
        assert len(groups["cat1"]) == 2
        assert groups["cat2"] == ["c"]

    def test_empty_features(self):
        fa = FeatureAnalyzer()
        assert fa._group_by_category() == {}
