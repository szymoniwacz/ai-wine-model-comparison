import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.model_factory import create_model


def test_create_decision_tree_model():
    model = create_model("decision_tree", max_depth=5)

    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 5


def test_create_logistic_regression_pipeline():
    model = create_model("logistic_regression")

    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "standardscaler"
    assert model.steps[1][0] == "logisticregression"


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model_type"):
        create_model("unknown_model")
