from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.available_models import AVAILABLE_MODELS
from src.data_loader import load_data
from src.model_factory import create_model


def get_model_path(model_type):
    return Path(f"artifacts/model_{model_type}.joblib")


def train(
    model_type=AVAILABLE_MODELS[0],
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    wine=None,
    save_model=True,
):
    # If no data provided, load and split
    if (
        X_train is None
        or X_test is None
        or y_train is None
        or y_test is None
        or wine is None
    ):
        X_train, X_test, y_train, y_test = load_data()
        wine = load_wine()

    model = create_model(model_type=model_type, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=wine.target_names, digits=3
    )

    result = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "model_path": str(get_model_path(model_type)),
    }

    if save_model:
        payload = {"model": model, "class_names": wine.target_names.tolist()}
        model_path = get_model_path(model_type)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, model_path)

    return result


def compare_models():
    # Load and split data only once
    X_train, X_test, y_train, y_test = load_data()
    wine = load_wine()
    results = []
    for model_type in AVAILABLE_MODELS:
        result = train(
            model_type=model_type,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            wine=wine,
            save_model=False,  # Don't save model during comparison
        )
        results.append(
            {
                "model": model_type,
                "accuracy": result["accuracy"],
                "confusion_matrix": result["confusion_matrix"],
                "classification_report": result["classification_report"],
                "model_path": result["model_path"],
            }
        )
    return results
