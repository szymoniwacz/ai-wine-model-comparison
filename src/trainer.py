from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from src.available_models import AVAILABLE_MODELS
from src.data_loader import load_data
from src.model_factory import create_model


def get_model_path(model_type):
    return Path(f"artifacts/model_{model_type}.joblib")


def train(model_type=AVAILABLE_MODELS[0]):
    X_train, X_test, y_train, y_test = load_data()

    model = create_model(model_type=model_type, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    wine = load_wine()
    payload = {"model": model, "class_names": wine.target_names.tolist()}

    model_path = get_model_path(model_type)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)

    return {"accuracy": accuracy, "confusion_matrix": cm, "model_path": str(model_path)}


def compare_models():
    results = []
    for model_type in AVAILABLE_MODELS:
        result = train(model_type=model_type)
        results.append(
            {
                "model": model_type,
                "accuracy": result["accuracy"],
                "confusion_matrix": result["confusion_matrix"],
                "model_path": result["model_path"],
            }
        )
    return results
