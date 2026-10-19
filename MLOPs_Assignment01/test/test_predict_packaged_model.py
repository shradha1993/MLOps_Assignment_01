import pytest
import numpy as np
from deployment.predict_packaged_model import predict_with_packaged_model
 
class DummyModel:
    def predict(self, X):
        # Simulate model predictions
        return np.ones(len(X))
 
def test_predict_with_packaged_model_returns_correct_length(capsys):
    model = DummyModel()
    X = np.array([[45, 1, 120, 240, 0, 1, 0, 0, 150, 0, 2, 1, 2],[40, 2, 100, 250, 1, 1, 0, 0, 130, 1, 1, 2, 2]])  # sample input
 
    predictions = predict_with_packaged_model(model, X)
 
    # Check output length matches input length
    assert len(predictions) == len(X)
 
    # Capture printed output
    captured = capsys.readouterr()
    assert "Predictions generated:" in captured.out