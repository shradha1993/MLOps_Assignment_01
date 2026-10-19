import os
from pathlib import Path
import shutil
import mlflow
from mlflow import MlflowClient


def get_default_tracking_uri():
    db_path = Path(__file__).resolve().parent.parent / "notebook" / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def export_champion_model(
    model_name="HeartDiseasePredictionChampion",
    model_version=None,
    output_dir=None,
    tracking_uri=None,
):
    """
    Exports registered champion model from MLflow Model Registry to a deployment directory.
    """
    project_root = Path(__file__).resolve().parent.parent

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or get_default_tracking_uri()
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()

    if model_version is None:
        # Pick latest version
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for registered model '{model_name}'.")
        model_version_info = versions[0]
    else:
        model_version_info = client.get_model_version(
            name=model_name,
            version=str(model_version),
        )

    export_dir = Path(output_dir) if output_dir else project_root / "deployment" / "model_export"

    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_version_info.run_id,
        artifact_path="model",
    )

    shutil.copytree(local_path, export_dir, dirs_exist_ok=True)
    print(f"Model exported successfully to: {export_dir}")
    return export_dir
