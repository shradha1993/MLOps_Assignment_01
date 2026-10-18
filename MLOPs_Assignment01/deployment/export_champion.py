from pathlib import Path
import shutil

import mlflow
from mlflow import MlflowClient


def export_champion_model(
    model_name,
    model_version,
    output_dir,
):
    project_root = Path(__file__).resolve().parent.parent

    mlflow.set_tracking_uri(
    "sqlite:///C:/Users/shrad/MLOps_Assignment_01/MLOPs_Assignment01/notebook/mlflow.db"
    )


    client = MlflowClient()

    model_version_info = client.get_model_version(
        name=model_name,
        version=model_version,
    )

    export_dir = Path(output_dir)

    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_version_info.run_id,
        artifact_path="model",
    )

    shutil.copytree(
        local_path,
        export_dir,
        dirs_exist_ok=True,
    )

    print(
        f"Model exported successfully to: "
        f"{export_dir}"
    )

    return export_dir
