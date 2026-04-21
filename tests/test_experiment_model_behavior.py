import os

import pytest

from src.experiments.model_behavior import run

EXPECTED_RESULT_KEYS = {
    "accuracies",
    "disagreement_count",
    "disagreement_indices",
    "artifact_paths",
}
EXPECTED_ARTIFACT_KEYS = {
    "cm_logistic_regression",
    "cm_svm",
    "cm_combined",
    "agreement_chart",
    "accuracy_chart",
}


def test_run_returns_expected_result_keys(tmp_path):
    result = run(artifacts_dir=str(tmp_path))

    assert set(result.keys()) == EXPECTED_RESULT_KEYS


def test_run_accuracies_contain_both_models(tmp_path):
    result = run(artifacts_dir=str(tmp_path))

    assert set(result["accuracies"].keys()) == {"logistic_regression", "svm"}
    assert all(0.0 <= acc <= 1.0 for acc in result["accuracies"].values())


def test_run_disagreement_count_matches_indices_length(tmp_path):
    result = run(artifacts_dir=str(tmp_path))

    assert result["disagreement_count"] == len(result["disagreement_indices"])


def test_run_disagreement_indices_are_valid_test_set_indices(tmp_path):
    from src.data_loader import load_data

    _, X_test, _, _ = load_data()
    result = run(artifacts_dir=str(tmp_path))

    assert all(0 <= i < len(X_test) for i in result["disagreement_indices"])


def test_run_artifact_paths_contain_expected_keys(tmp_path):
    result = run(artifacts_dir=str(tmp_path))

    assert EXPECTED_ARTIFACT_KEYS.issubset(set(result["artifact_paths"].keys()))


def test_run_creates_all_artifact_files(tmp_path):
    result = run(artifacts_dir=str(tmp_path))

    for key, path in result["artifact_paths"].items():
        assert os.path.isfile(path), f"Expected artifact not found: {key} -> {path}"


def test_run_is_reproducible(tmp_path):
    result_a = run(artifacts_dir=str(tmp_path / "a"))
    result_b = run(artifacts_dir=str(tmp_path / "b"))

    assert result_a["accuracies"] == result_b["accuracies"]
    assert result_a["disagreement_count"] == result_b["disagreement_count"]
    assert result_a["disagreement_indices"] == result_b["disagreement_indices"]
