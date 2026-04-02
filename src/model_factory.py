from src.available_models import AVAILABLE_MODELS
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(model_type=AVAILABLE_MODELS[0], max_depth=3):
    if model_type == "decision_tree":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    elif model_type == "svm":
        return make_pipeline(StandardScaler(), SVC(random_state=42))
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Available: {AVAILABLE_MODELS}"
        )
