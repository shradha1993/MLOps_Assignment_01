# Webinar 1 — End-to-End MLOps Lab Manual
## Diabetes Prediction Project

This manual is the source guide for the complete Webinar 1 lab. Follow the steps in order.

The notebook is the teaching/orchestration layer. Reusable implementation logic is kept in Python modules.

---

# 1. Prerequisites

## 1.1 Recommended Environment

- Windows 10/11
- WSL2
- Ubuntu 24.04 LTS
- VS Code with WSL support
- Git
- Python 3.12
- `uv`
- Docker Desktop for the containerization section

Use the Ubuntu terminal inside WSL.

## 1.2 Install WSL

From Windows PowerShell:

```powershell
wsl --install
```

Restart Windows if prompted.

Check:

```powershell
wsl --list --verbose
```

Start Ubuntu:

```powershell
wsl
```

Update Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git curl build-essential
```

## 1.3 Install uv

Check Python:

```bash
python3 --version
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source ~/.bashrc
```

Check:

```bash
uv --version
```

## 1.4 VS Code

Install VS Code on Windows and the WSL extension.

From WSL:

```bash
code .
```

Use VS Code for the project, terminal, and Jupyter notebook.

---

# 2. Create the Project

```bash
mkdir -p ~/webinar1-mlops
cd ~/webinar1-mlops
```

Create the Python environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Create directories:

```bash
mkdir -p notebook
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/tracking
mkdir -p src/inference
mkdir -p deployment
mkdir -p dags
```

Project structure:

```text
webinar1-mlops/
├── notebook/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── validate_data.py
│   ├── features/
│   │   └── prepare_data.py
│   ├── models/
│   │   ├── model_pipelines.py
│   │   ├── train_models.py
│   │   └── evaluate_models.py
│   ├── tracking/
│   │   ├── mlflow_tracking.py
│   │   ├── select_champion.py
│   │   └── register_model.py
│   └── inference/
│       ├── health_check.py
│       ├── invoke_model.py
│       ├── predict.py
│       └── validate_response.py
├── deployment/
│   ├── export_champion.py
│   ├── load_exported_model.py
│   ├── predict_packaged_model.py
│   ├── inspect_model_export.py
│   ├── inspect_model_requirements.py
│   └── batch_inference.py
├── dags/
│   └── diabetes_mlops_pipeline.py
├── Dockerfile
└── ...
```

---

# 3. Install Libraries

Use `uv`:

```bash
uv pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn jupyter ipykernel requests mlflow
```

Install Airflow:

```bash
uv pip install apache-airflow
```

Libraries/tools used:

| Library / Tool | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| SciPy | Scientific computing |
| scikit-learn | Preprocessing, models, pipelines and metrics |
| XGBoost | Gradient-boosting model |
| MLflow | Experiment tracking, model registry and serving |
| requests | Dataset download and model API calls |
| Jupyter | Interactive lab notebook |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| Airflow | Workflow orchestration |
| Docker | Containerization |

The packaged model may also contain:

```text
cloudpickle
psutil
```

as runtime dependencies.

---

# Phase 1 — Data Loading & Validation

# 4. Dataset

We use the PIMA Indians Diabetes Dataset.

The program downloads the dataset automatically, so students do not need to manually download the CSV.

The dataset contains:

- 768 records
- 8 input features
- 1 target column

Target:

```text
Outcome
```

Meaning:

```text
0 → No diabetes
1 → Diabetes
```

---

# 5. Load the Dataset

The program will:

1. Check whether `data/raw/diabetes.csv` exists.
2. Download it if it does not exist.
3. Create the required directory.
4. Save the CSV locally.
5. Load it into a pandas DataFrame.

Create:

```text
src/data/load_data.py
```

```python
from pathlib import Path

import pandas as pd
import requests


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

DATA_PATH = Path("data/raw/diabetes.csv")

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def download_dataset(
    url=DATA_URL,
    output_path=DATA_PATH,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading diabetes dataset...")

    response = requests.get(
        url,
        timeout=30,
    )
    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )

    print(
        f"Dataset downloaded to: "
        f"{output_path}"
    )


def load_data(
    file_path=DATA_PATH,
):
    file_path = Path(file_path)

    if not file_path.exists():
        download_dataset(
            output_path=file_path
        )

    data = pd.read_csv(
        file_path,
        header=None,
        names=COLUMN_NAMES,
    )

    print(
        f"Dataset loaded successfully: "
        f"{data.shape}"
    )

    return data
```

The source CSV is headerless, so we provide the column names explicitly using:

```python
header=None
```

and:

```python
names=COLUMN_NAMES
```

Notebook:

```python
from src.data.load_data import load_data

data = load_data()

data.head()
```

The local dataset is stored at:

```text
data/raw/diabetes.csv
```

---

# 6. Validate the Dataset

Create:

```text
src/data/validate_data.py
```

```python
REQUIRED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def validate_data(data):
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns are present.")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return True
```

Notebook:

```python
from src.data.validate_data import validate_data

validate_data(data)
```

---

# Phase 2 — Data Preparation

# 7. Prepare the Data

We:

1. Separate features and target.
2. Treat medically invalid zero values as missing.
3. Apply median imputation.
4. Scale numerical features.
5. Split the data into training and test sets.

Create:

```text
src/features/prepare_data.py
```

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"

ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def prepare_data(data):
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    X[ZERO_AS_MISSING_COLUMNS] = X[
        ZERO_AS_MISSING_COLUMNS
    ].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", preprocessing_pipeline, FEATURE_COLUMNS)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )
```

Notebook:

```python
from src.features.prepare_data import prepare_data

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
) = prepare_data(data)
```

---

# Phase 3 — Train & Compare Models

# 8. Define Candidate Models

We train:

- Logistic Regression
- Random Forest
- XGBoost
- MLP Neural Network

Create:

```text
src/models/model_pipelines.py
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42,
        ),
    }
```

---

# 9. Train the Models

Create:

```text
src/models/train_models.py
```

```python
from src.models.model_pipelines import get_models


def train_models(X_train, y_train):
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
```

Notebook:

```python
from src.models.train_models import train_models

trained_models = train_models(
    X_train,
    y_train,
)
```

---

# 10. Evaluate the Models

Create:

```text
src/models/evaluate_models.py
```

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_models(
    trained_models,
    X_test,
    y_test,
):
    results = {}

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        }

    return results
```

Notebook:

```python
from src.models.evaluate_models import evaluate_models

results = evaluate_models(
    trained_models,
    X_test,
    y_test,
)

results
```

---

# Phase 4 — MLflow Tracking

# 11. Configure MLflow

Use SQLite as the MLflow backend:

```python
import mlflow

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Diabetes Prediction"
)
```

Experiment metadata is stored in:

```text
mlflow.db
```

---

# 12. Track Experiments

Create:

```text
src/tracking/mlflow_tracking.py
```

```python
import mlflow
import mlflow.sklearn


def track_models(
    trained_models,
    X_test,
    y_test,
    results,
):
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "Diabetes Prediction"
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
            )
```

Notebook:

```python
from src.tracking.mlflow_tracking import track_models

track_models(
    trained_models,
    X_test,
    y_test,
    results,
)
```

---

# 13. Start the MLflow UI

From the project root:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Use the UI to view experiments, parameters, metrics and model artifacts.

---

# Phase 5 — Model Selection, Registration & Serving

# 14. Select the Champion Model

Select the best model using F1 score.

Create:

```text
src/tracking/select_champion.py
```

```python
def select_champion(results):
    champion_name = max(
        results,
        key=lambda name: results[name]["f1"],
    )

    champion_metrics = results[champion_name]

    return {
        "model_name": champion_name,
        "metrics": champion_metrics,
    }
```

Notebook:

```python
from src.tracking.select_champion import select_champion

champion = select_champion(results)

champion
```

---

# 15. Register the Champion Model

Register the selected model in MLflow using:

```text
DiabetesPredictionChampion
```

The registry maintains model versions:

```text
DiabetesPredictionChampion
├── Version 1
├── Version 2
└── Version 3
```

The serving and packaging flow in this lab uses:

```text
Version 3
```

---

# 16. Serve the Registered Model

```bash
mlflow models serve \
  -m "models:/DiabetesPredictionChampion/3" \
  -p 5001 \
  --host 127.0.0.1 \
  --env-manager local
```

Server:

```text
http://127.0.0.1:5001
```

Important endpoints:

```text
/ping
/health
/version
/invocations
```

---

# 17. Health Check

Create:

```text
src/inference/health_check.py
```

```python
import requests


def check_model_health(
    url="http://127.0.0.1:5001/ping",
):
    response = requests.get(url)
    response.raise_for_status()

    print("Model server is healthy.")
    return True
```

Notebook:

```python
from src.inference.health_check import check_model_health

check_model_health()
```

---

# 18. Invoke the Model

Create:

```text
src/inference/invoke_model.py
```

```python
import requests


def invoke_model_server(
    data,
    columns,
    url="http://127.0.0.1:5001/invocations",
):
    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": data,
        }
    }

    response = requests.post(
        url,
        json=payload,
    )

    response.raise_for_status()

    return response.json()
```

Feature columns:

```python
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
```

Notebook:

```python
from src.inference.invoke_model import invoke_model_server

prediction_response = invoke_model_server(
    data=[
        [
            6,
            148,
            72,
            35,
            0,
            33.6,
            0.627,
            50,
        ]
    ],
    columns=FEATURE_COLUMNS,
)

prediction_response
```

---

# Phase 6 — Model Packaging

# 19. Export the Champion Model

Create/use:

```text
deployment/export_champion.py
```

```python
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
        f"sqlite:///{project_root / 'mlflow.db'}"
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
```

Notebook:

```python
from deployment.export_champion import (
    export_champion_model,
)

champion_path = export_champion_model(
    model_name="DiabetesPredictionChampion",
    model_version="3",
    output_dir="deployment/model_export",
)
```

Packaged artifact:

```text
deployment/model_export/
├── MLmodel
├── model.pkl
├── requirements.txt
├── conda.yaml
├── python_env.yaml
└── registered_model_meta
```

---

# 20. Load the Packaged Model

Create:

```text
deployment/load_exported_model.py
```

```python
import mlflow.pyfunc


def load_exported_model(
    model_path="deployment/model_export",
):
    model = mlflow.pyfunc.load_model(
        model_path
    )

    return model
```

Notebook:

```python
from deployment.load_exported_model import (
    load_exported_model,
)

packaged_model = load_exported_model()
```

---

# 21. Predict with the Packaged Model

Create:

```text
deployment/predict_packaged_model.py
```

```python
def predict_with_packaged_model(
    model,
    X,
):
    predictions = model.predict(X)

    print(
        f"Predictions generated: "
        f"{len(predictions)}"
    )

    return predictions
```

Notebook:

```python
from deployment.predict_packaged_model import (
    predict_with_packaged_model,
)

packaged_predictions = (
    predict_with_packaged_model(
        packaged_model,
        X_test,
    )
)
```

---

# Phase 7 — Docker Containerization

Docker packages the model and runtime dependencies into a portable container.

Conceptually:

```text
Python
  +
MLflow
  +
Model dependencies
  +
Packaged model
      ↓
Docker Image
      ↓
Docker Container
```

# 22. Create the Dockerfile

Create:

```text
Dockerfile
```

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY deployment/model_export /app/model_export

RUN pip install --no-cache-dir -r /app/model_export/requirements.txt

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "/app/model_export", "--host", "0.0.0.0", "--port", "5001", "--env-manager", "local"]
```

Purpose:

```text
FROM       → Base Python image
WORKDIR    → Working directory
COPY       → Copies packaged model
RUN        → Installs model dependencies
EXPOSE     → Documents port 5001
CMD        → Starts MLflow model server
```

# 23. Build the Docker Image

```bash
docker build -t diabetes-model:v1 .
```

# 24. Run the Container

```bash
docker run --rm -p 5001:5001 diabetes-model:v1
```

Port mapping:

```text
Host 5001 → Container 5001
```

# 25. Test Containerized Inference

```bash
curl http://127.0.0.1:5001/ping
```

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
      ],
      "data": [
        [6, 148, 72, 35, 0, 33.6, 0.627, 50]
      ]
    }
  }'
```

If Docker is unavailable because of the local/corporate environment, continue with the remaining phases.

---

# Phase 8 — Batch Inference

Batch inference means generating predictions for many records.

Flow:

```text
Input Dataset
      ↓
Load Packaged Model
      ↓
Generate Predictions
      ↓
Combine Input + Predictions
      ↓
Save Prediction Results
```

# 26. Implement Batch Inference

Create:

```text
deployment/batch_inference.py
```

```python
from pathlib import Path

import mlflow.pyfunc
import pandas as pd


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def run_batch_inference(
    model_path="deployment/model_export",
    input_path="data/raw/diabetes.csv",
    output_path="data/processed/batch_predictions.csv",
):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    input_data = pd.read_csv(input_file)

    X_batch = input_data[
        FEATURE_COLUMNS
    ]

    model = mlflow.pyfunc.load_model(
        model_path
    )

    predictions = model.predict(
        X_batch
    )

    output_data = X_batch.copy()

    output_data["prediction"] = predictions

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data.to_csv(
        output_file,
        index=False,
    )

    print("Batch inference completed.")
    print(
        f"Output file: {output_file}"
    )

    return output_data
```

The `Outcome` column is not supplied to the model because it is the target being predicted.

Notebook:

```python
from deployment.batch_inference import (
    run_batch_inference,
)

batch_results = run_batch_inference()

batch_results.head(10)
```

Output:

```text
data/processed/batch_predictions.csv
```

---

# Phase 9 — Airflow Orchestration

Airflow is the orchestration layer. It coordinates the existing Python components.

Workflow:

```text
prepare_data
      ↓
train_models
      ↓
evaluate_models
      ↓
select_best_model
      ↓
package_model
      ↓
run_inference
```

# 27. Create the Airflow DAG

Create:

```text
dags/diabetes_mlops_pipeline.py
```

```python
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def prepare_data():
    from src.features.prepare_data import (
        prepare_data as prepare,
    )

    prepare()


def train_models():
    from src.models.train_models import train_models

    train_models()


def evaluate_models():
    from src.models.evaluate_models import evaluate_models

    evaluate_models()


def select_best_model():
    from src.tracking.select_champion import (
        select_champion,
    )

    champion = select_champion()

    if isinstance(champion, dict):
        return champion

    return {
        "model_version": "3"
    }


def package_model(**context):
    from deployment.export_champion import (
        export_champion_model,
    )

    champion = context["ti"].xcom_pull(
        task_ids="select_best_model"
    )

    model_version = champion[
        "model_version"
    ]

    export_champion_model(
        model_name="DiabetesPredictionChampion",
        model_version=str(model_version),
        output_dir="deployment/model_export",
    )


def run_inference():
    from deployment.batch_inference import (
        run_batch_inference,
    )

    run_batch_inference()


with DAG(
    dag_id="diabetes_mlops_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    prepare_data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    train_models_task = PythonOperator(
        task_id="train_models",
        python_callable=train_models,
    )

    evaluate_models_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
    )

    select_best_model_task = PythonOperator(
        task_id="select_best_model",
        python_callable=select_best_model,
    )

    package_model_task = PythonOperator(
        task_id="package_model",
        python_callable=package_model,
    )

    run_inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference,
    )

    (
        prepare_data_task
        >> train_models_task
        >> evaluate_models_task
        >> select_best_model_task
        >> package_model_task
        >> run_inference_task
    )
```

The DAG handles orchestration. The ML implementation remains in Python modules.

# 28. Start Airflow

For a local demonstration environment:

```bash
airflow standalone
```

Open the Airflow UI at the address shown by the command.

# 29. Trigger the MLOps Workflow

```bash
airflow dags trigger diabetes_mlops_pipeline
```

The complete workflow:

```text
Data Preparation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Champion Selection
      ↓
Model Packaging
      ↓
Batch Inference
```

Final prediction output:

```text
data/processed/batch_predictions.csv
```

---

# 30. Complete Webinar Flow

```text
Real-World Problem
        ↓
Understand Data
        ↓
Data Loading & Validation
        ↓
Data Preparation
        ↓
Train Candidate Models
        ↓
Evaluate Models
        ↓
MLflow Experiment Tracking
        ↓
Select Champion Model
        ↓
Model Registration
        ↓
Model Serving
        ↓
Model Packaging
        ↓
Docker Containerization
        ↓
Batch Inference
        ↓
Airflow Orchestration
        ↓
Prediction Output
```

The central MLOps idea:

> MLOps is not just about training a machine learning model. It is about building a repeatable, deployable, and maintainable system around the model.
