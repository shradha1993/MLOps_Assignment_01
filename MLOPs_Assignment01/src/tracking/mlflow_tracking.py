import mlflow
import mlflow.sklearn


def track_models(
    trained_models,
    X_test,
    y_test,
    results,
):
    mlflow.set_tracking_uri(
    "sqlite:///C:/Users/shrad/MLOps_Assignment_01/MLOPs_Assignment01/notebook/mlflow.db"
    )

    mlflow.set_experiment(
        "Heart Disease Prediction"
    )

    for name, model in trained_models.items():
        with mlflow.start_run(
            run_name=name
        ):
            metrics = results[name]

            mlflow.log_param(
                "model_name",
                name,
            )

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format="pickle",
                registered_model_name="HeartDiseasePredictionChampion"
            )
