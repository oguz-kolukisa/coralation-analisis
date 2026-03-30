"""Tests for tautological edit detection in pipeline.py."""
from src.pipeline import _is_tautological_edit


class TestIsTautologicalEdit:
    def test_class_name_in_instruction(self):
        assert _is_tautological_edit(
            "Modify feather to resemble a house finch", "house finch",
        ) is True

    def test_concrete_feature_not_tautological(self):
        assert _is_tautological_edit(
            "Add red breast feathers with white wing bars", "house finch",
        ) is False

    def test_goose_in_instruction(self):
        assert _is_tautological_edit(
            "Make it look like a goose", "goose",
        ) is True

    def test_concrete_edit_for_goose(self):
        assert _is_tautological_edit(
            "Add long white neck with orange beak", "goose",
        ) is False

    def test_synonym_detected(self):
        assert _is_tautological_edit(
            "Add sulphur-crested cockatoo crest feathers",
            "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
        ) is True

    def test_short_synonym_skipped(self):
        assert _is_tautological_edit(
            "Add a red patch", "red, crimson",
        ) is False

    def test_empty_class_name(self):
        assert _is_tautological_edit(
            "Add red feathers", "",
        ) is False

    def test_cock_in_instruction(self):
        assert _is_tautological_edit(
            "Replace with cock features", "cock",
        ) is True

    def test_unrelated_instruction(self):
        assert _is_tautological_edit(
            "Remove the background completely", "cock",
        ) is False
