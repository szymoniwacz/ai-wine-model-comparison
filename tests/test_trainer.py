import joblib

from src.trainer import AVAILABLE_MODELS, compare_models, train


def test_train_returns_expected_result_keys_without_saving():
    result = train(save_model=False)

    assert set(result.keys()) == {
        "accuracy",
        "confusion_matrix",
        "classification_report",
        "model_path",
    }
    assert 0 <= result["accuracy"] <= 1
    assert "precision" in result["classification_report"]


def test_train_saves_model_payload_when_enabled(monkeypatch, tmp_path):
    output_path = tmp_path / "model.joblib"

    def _tmp_model_path(_model_type):
        return output_path

    monkeypatch.setattr("src.trainer.get_model_path", _tmp_model_path)

    train(model_type="decision_tree", save_model=True)

    assert output_path.exists()
    payload = joblib.load(output_path)
    assert "model" in payload
    assert "class_names" in payload


def test_compare_models_returns_all_supported_models():
    results = compare_models()

    assert len(results) == len(AVAILABLE_MODELS)
    model_names = {r["model"] for r in results}
    assert model_names == set(AVAILABLE_MODELS)
    assert all(0 <= r["accuracy"] <= 1 for r in results)
