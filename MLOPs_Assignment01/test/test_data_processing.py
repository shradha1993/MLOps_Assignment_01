import numpy as np
import pandas as pd
import pytest
from src.features.prepare_data import prepare_data
from src.data.validate_data import validate_data
from src.models.model_pipelines import get_models
from src.models.evaluate_models import evaluate_models


@pytest.fixture
def sample_heart_data():
    """Generates synthetic heart disease tabular data for testing."""
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.rand(50, len(columns)) * 50,
        columns=columns
    )
    outcome = pd.DataFrame({"num": np.random.choice([0, 1, 2, 3], size=50)})
    return data, outcome


def test_data_validation(sample_heart_data):
    X, y = sample_heart_data
    assert validate_data(X) is True
    assert validate_data(y) is True


def test_prepare_data_output_shapes(sample_heart_data):
    X, y = sample_heart_data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)

    # Check 80/20 train/test split
    assert len(X_train) == 40
    assert len(X_test) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10

    # Target must be strictly binary (0 or 1)
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_model_training_and_evaluation(sample_heart_data):
    X, y = sample_heart_data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)

    models = get_models()
    # Test with Logistic Regression
    lr = models["Logistic Regression"]
    lr.fit(X_train, y_train)

    results = evaluate_models({"Logistic Regression": lr}, X_test, y_test)
    assert "Logistic Regression" in results
    res = results["Logistic Regression"]
    assert "accuracy" in res
    assert "f1" in res
    assert "roc_auc" in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0
