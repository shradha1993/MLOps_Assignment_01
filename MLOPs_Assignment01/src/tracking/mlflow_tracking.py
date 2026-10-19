import os
from pathlib import Path
import mlflow
import mlflow.sklearn


def get_default_tracking_uri():
    db_path = Path(__file__).resolve().parent.parent.parent / "notebook" / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def track_models(
    trained_models,
    X_test,
    y_test,
    results,
    tracking_uri=None,
    experiment_name="Heart Disease Prediction",
):
    """
    Logs parameters, evaluation metrics, and model artifacts to MLflow.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or get_default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    for name, model in trained_models.items():
        with mlflow.start_run(run_name=name):
            metrics = results[name]

            # Log parameters
            mlflow.log_param("model_name", name)
            if hasattr(model, "get_params"):
                params = model.get_params()
                for p_key in ["C", "max_iter", "n_estimators", "max_depth", "learning_rate"]:
                    if p_key in params:
                        mlflow.log_param(p_key, params[p_key])

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format="pickle",
            )
