import os
from pathlib import Path
import mlflow.pyfunc


def load_exported_model(model_path=None):
    """
    Loads packaged model using MLflow PyFunc.
    """
    if model_path is None:
        model_path = os.getenv(
            "MODEL_PATH",
            str(Path(__file__).resolve().parent / "model_export"),
        )

    model = mlflow.pyfunc.load_model(model_path)
    return model
