# BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI
## Work Integrated Learning Programmes (WILP)
### AIMLCZG523: Machine Learning Operations (MLOps) — Assignment 01

# End-to-End Heart Disease Prediction Pipeline: CI/CD, Containerization & Production Cloud Monitoring

**Student ID:** s2025ae05331  
**Programme:** M.Tech / Post Graduate Programme in AI & ML  
**Submission Date:** October 2026  
**Total Marks:** 50 Marks  

---

## 1. Executive Summary & Problem Formulation

Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, representing an estimated 32% of all deaths worldwide. Early, accurate diagnosis and risk stratification can significantly improve clinical outcomes and save patient lives. While standard machine learning models achieve satisfactory diagnostic accuracy in research environments, translating these models into reliable, production-ready healthcare systems requires a rigorous Machine Learning Operations (MLOps) architecture.

This project designs, implements, and deploys a scalable, automated, and observable end-to-end MLOps pipeline for heart disease risk prediction based on the UCI Heart Disease (Cleveland Clinic) dataset. The pipeline spans the complete lifecycle:
* Automated data ingestion and clinical preprocessing.
* Multi-model experimentation (Logistic Regression, Random Forest, XGBoost).
* Systematic experiment tracking and model registration via MLflow.
* Continuous Integration and Continuous Deployment (CI/CD) with automated testing via GitHub Actions.
* Model serving via a high-performance FastAPI microservice inside an optimized Docker container.
* Declarative production orchestration on Kubernetes (Minikube).
* Real-time metrics scraping and monitoring with Prometheus.

---

## 2. System Architecture & Workflow Pipeline

The system decouples data engineering, model training, model registration, deployment, and live monitoring into independent, reproducible micro-components.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            END-TO-END MLOPS PIPELINE                        │
├─────────────────┬─────────────────┬───────────────────┬─────────────────────┤
│ 1. Data Layer   │ 2. Training     │ 3. CI/CD & Deploy │ 4. Observability    │
├─────────────────┼─────────────────┼───────────────────┼─────────────────────┤
│ UCI Repository  │ Model Pipelines │ GitHub Actions    │ Prometheus Scraper  │
│      ↓          │ (LR, RF, XGB)   │ (Flake8 + Pytest) │ (5s Interval)       │
│ Data Ingestion  │      ↓          │      ↓            │      ↓              │
│      ↓          │ 5-Fold Strat CV │ Docker Build      │ Latency & RPS       │
│ Missing Value   │      ↓          │      ↓            │ Class Drift Alerts  │
│ Imputation &    │ MLflow Tracking │ Kubernetes Deploy │ Error Rate Tracking │
│ Feature Scaling │ & Model Registry│ (Minikube Service)│ Dashboard UI        │
└─────────────────┴─────────────────┴───────────────────┴─────────────────────┘
```

> **📷 [SCREENSHOT PLACEHOLDER 1: Architecture & Workflow Diagram]**  
> *Caption: High-level End-to-End MLOps System Architecture Diagram.*  
> *(Paste your overall system flow diagram or architectural block diagram here).*

---

## 3. Setup & Installation Instructions

The project is developed to execute reproducibly across Windows WSL2 (Ubuntu 24.04 LTS), native Linux instances, and cloud VMs. All scripts utilize relative path resolution to eliminate environment coupling.

### 3.1 Environment Prerequisites
* **Operating System:** Ubuntu 24.04 LTS or Windows WSL2
* **Python Runtime:** Python 3.12 (managed via virtual environment)
* **Container Runtime:** Docker Engine / Docker Desktop
* **Local Orchestration:** Minikube v1.32+ with `kubectl` CLI

### 3.2 Installation Commands
```bash
# 1. Clone repository
git clone https://github.com/shradha1993/MLOps_Assignment_01.git
cd MLOps_Assignment_01/MLOPs_Assignment01

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Data Acquisition & Exploratory Data Analysis (EDA)

The project utilizes the UCI Heart Disease (Cleveland Clinic) benchmark dataset (UCI ID: 45), comprising 303 patient records across 13 clinical attributes and a target indicator representing angiographic disease presence.

### 4.1 Feature Dictionary
| Feature | Type | Clinical Description | Domain / Range |
| :--- | :--- | :--- | :--- |
| `age` | Integer | Patient age in years | 29 – 77 |
| `sex` | Binary | Biological sex (1 = Male, 0 = Female) | 0, 1 |
| `cp` | Categorical | Chest pain type (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic) | 1 – 4 |
| `trestbps` | Continuous | Resting systolic blood pressure upon admission (mm Hg) | 94 – 200 |
| `chol` | Continuous | Serum cholesterol level (mg/dl) | 126 – 564 |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) | 0, 1 |
| `restecg` | Categorical | Resting electrocardiographic results (0: normal, 1: ST-T abnormality, 2: LVH) | 0 – 2 |
| `thalach` | Continuous | Maximum heart rate achieved during exercise stress | 71 – 202 |
| `exang` | Binary | Exercise-induced angina (1 = Yes, 0 = No) | 0, 1 |
| `oldpeak` | Continuous | ST depression induced by exercise relative to rest | 0.0 – 6.2 |
| `slope` | Categorical | Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping) | 1 – 3 |
| `ca` | Discrete | Number of major vessels (0-3) colored by fluoroscopy | 0 – 3 (has NaNs) |
| `thal` | Categorical | Thallium heart scan (3: normal, 6: fixed defect, 7: reversible defect) | 3, 6, 7 (has NaNs) |
| `num` (Target) | Binary | Diagnosis of coronary artery disease | 0 = Absence, 1 = Presence |

### 4.2 Clinical Missing Value & Target Binarization Strategy
* **Missing Value Imputation:** Missing values occur only in `ca` (4 missing) and `thal` (2 missing). A median imputation strategy is applied via `SimpleImputer`. Crucially, zeros in binary/categorical columns (`sex=0`, `fbs=0`, `exang=0`, `ca=0`) denote valid clinical states and are strictly preserved.
* **Target Binarization:** Raw values 1 to 4 indicate varying severity of coronary artery narrowing. In accordance with clinical literature, the target is binarized into: `0` (Absence of Heart Disease) and `1` (Presence of Heart Disease, where `num > 0`).

> **📷 [SCREENSHOT PLACEHOLDER 2: Class Balance & Feature Distribution Histograms]**  
> *Caption: Target Class Balance Countplot and Histograms of Key Clinical Indicators.*  
> *(Paste output from Task 1 of notebook showing balanced distribution 164 vs 139, and histograms for age, chol, trestbps, thalach, oldpeak).*

> **📷 [SCREENSHOT PLACEHOLDER 3: Feature Correlation Heatmap]**  
> *Caption: Seaborn Correlation Matrix Heatmap across Clinical Features and Target.*  
> *(Paste correlation heatmap output from Task 1 of the notebook).*

---

## 5. Feature Engineering & Model Development

A unified Scikit-learn Pipeline was constructed encapsulating median imputation and `StandardScaler` variance normalization. The dataset was split into training (80%) and held-out testing (20%) sets using stratified sampling to preserve class ratios.

### 5.1 Model Selection & Rationale
1. **Logistic Regression:** Regularized linear classifier serving as an interpretable baseline.
2. **Random Forest Classifier:** Bagged ensemble of 200 decision trees to capture non-linear relationships.
3. **XGBoost Classifier:** Gradient boosted trees with tree depth constrained to 4 to prevent overfitting.

### 5.2 Performance Evaluation (5-Fold Stratified CV & Held-Out Test Set)
| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8525 | 0.8438 | 0.8710 | 0.8571 | 0.9022 |
| **Random Forest Classifier** | **0.8689** | **0.8710** | **0.8710** | **0.8710** | **0.9161** |
| **XGBoost Classifier** | 0.8361 | 0.8387 | 0.8387 | 0.8387 | 0.8935 |

> **📷 [SCREENSHOT PLACEHOLDER 4: Model Training & Evaluation Outputs]**  
> *Caption: Jupyter Notebook Output Displaying Candidate Model Evaluation Metrics Table.*  
> *(Paste screenshot of notebook Task 2 evaluation results).*

---

## 6. Experiment Tracking with MLflow

MLflow tracks all experimentation runs using a local SQLite database (`sqlite:///notebook/mlflow.db`) under the experiment `"Heart Disease Prediction"`.

* **Logged Parameters:** `model_name`, `n_estimators`, `max_depth`, `learning_rate`, `C`, `max_iter`.
* **Logged Metrics:** `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `cv_roc_auc_mean`.
* **Champion Model Selection:** The model with the highest F1-score and ROC-AUC (**Random Forest**, F1: 0.8710, ROC-AUC: 0.9161) was programmatically selected and registered in the MLflow Model Registry as:  
  `HeartDiseasePredictionChampion`.

> **📷 [SCREENSHOT PLACEHOLDER 5: MLflow Experiment Tracking Dashboard]**  
> *Caption: MLflow UI Showing Tracked Runs, Hyperparameters, and Metrics Comparison.*  
> *(Paste screenshot from MLflow web UI showing runs for LR, RF, XGB).*

> **📷 [SCREENSHOT PLACEHOLDER 6: MLflow Model Registry]**  
> *Caption: MLflow Model Registry Showing Registered Champion Model.*  
> *(Paste screenshot of the 'Models' tab in MLflow showing HeartDiseasePredictionChampion).*

---

## 7. Model Packaging & Local Inference Verification

The registered champion was exported from MLflow into `deployment/model_export/` containing:
* `MLmodel`: Model format descriptor and input/output tensor specification.
* `model.pkl`: Serialized scikit-learn estimator.
* `conda.yaml` & `requirements.txt`: Environment and dependency pinning.

Offline validation via `test/test_predict_packaged_model.py` verified that model loading and inference function correctly outside the training environment.

> **📷 [SCREENSHOT PLACEHOLDER 7: Packaged Model Offline Inference Proof]**  
> *Caption: Terminal / Notebook Execution of Packaged Model Loading and Prediction.*  
> *(Paste screenshot showing successful prediction from deployment/model_export).*

---

## 8. CI/CD Pipeline & Automated Testing (GitHub Actions)

Continuous Integration is orchestrated via GitHub Actions (`.github/workflows/mlops.yaml`):
1. **Linting & Code Quality:** Flake8 verifies compliance with PEP 8 standards.
2. **Automated Unit Tests:** Pytest executes test suites:
   * `test_data_processing.py`: Verifies schema validation, split ratios, and binary target consistency.
   * `test_predict_packaged_model.py`: Checks serialized model output dimensions.
   * `test_fastapi_app.py`: Tests `/health`, `/predict`, and `/metrics` endpoints.
3. **Container Build Smoke Test:** Builds Docker image and queries endpoints inside an isolated runner.

> **📷 [SCREENSHOT PLACEHOLDER 8: Pytest Unit Test Suite Execution]**  
> *Caption: Terminal Output Showing All Automated Unit Tests Passing.*  
> *(Paste terminal output of pytest -v showing all tests passed).*

> **📷 [SCREENSHOT PLACEHOLDER 9: GitHub Actions CI/CD Pipeline Success]**  
> *Caption: GitHub Actions Workflow Execution Showing Passing Jobs.*  
> *(Paste GitHub Actions webpage showing green checkmark for the workflow).*

---

## 9. Model Containerization with Docker & FastAPI

The model is served using a high-throughput FastAPI application (`deployment/fastapi_app.py`) packaged inside a lightweight Python 3.9 container (`Dockerfile`).

### 9.1 API Endpoints
* **`POST /predict`**: Accepts patient clinical data and returns **prediction** (0 or 1) and **confidence** probability score:
  ```json
  {"predictions": [1], "confidence": [0.88]}
  ```
* **`GET /health`**: Health check probe returning `{"status": "healthy", "model_loaded": true}`.
* **`GET /metrics`**: Prometheus metrics exposition endpoint.

> **📷 [SCREENSHOT PLACEHOLDER 10: Docker Build & Local Execution Proof]**  
> *Caption: Terminal Output Showing Docker Image Build and Running Container.*  
> *(Paste terminal output of docker build and docker ps).*

> **📷 [SCREENSHOT PLACEHOLDER 11: Prediction & Confidence Endpoint Response]**  
> *Caption: Curl POST Request to /predict Returning Prediction and Confidence Score.*  
> *(Paste terminal screenshot of curl POST to /predict with output).*

---

## 10. Production Deployment on Kubernetes (Minikube)

The containerized service is deployed on Kubernetes using declarative YAML manifests:
* **`mlops-deployment`**: Manages container pod replicas with rolling update strategy.
* **`mlops-service`**: Exposes the application on port 5001 using NodePort / LoadBalancer.

Bootstrap automation is provided via `setup_mlops.sh` to configure the cluster, point Docker to Minikube's daemon, apply manifests, and deploy Prometheus.

> **📷 [SCREENSHOT PLACEHOLDER 12: Kubernetes Pods & Service Status]**  
> *Caption: Terminal Output of kubectl get pods,svc Showing Healthy Running Pods.*  
> *(Paste screenshot showing mlops-deployment 1/1 Running).*

> **📷 [SCREENSHOT PLACEHOLDER 13: Browser Verification of Deployed Service]**  
> *Caption: Browser Window Accessing FastAPI Service via Minikube URL.*  
> *(Paste your browser screenshot of http://192.168.49.2:31543/health).*

---

## 11. Observability & Monitoring with Prometheus

Prometheus is deployed in the `monitoring` namespace and configured via a ConfigMap to scrape `mlops-service.default.svc.cluster.local:5001/metrics` every 5 seconds.

### 11.1 Custom MLOps Prometheus Metrics
* `request_count_total`: Total invocations per endpoint.
* `request_latency_seconds`: Request latency histogram.
* `model_prediction_count_total`: Count of class 0 vs. class 1 predictions over time (detects class drift).
* `model_inference_latency_seconds`: Measures pure model computation time.
* `http_responses_total`: Categorizes HTTP response codes (200, 400, 500) for error rate tracking.

### 11.2 Key PromQL Queries
* **Throughput (RPS):** `rate(request_count_total[1m])`
* **p95 Latency:** `histogram_quantile(0.95, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))`
* **Prediction Drift:** `sum by (predicted_class) (model_prediction_count_total)`
* **Error Rate (%):** `sum(rate(http_responses_total{status_code=~"4..|5.."}[5m])) / sum(rate(http_responses_total[5m])) * 100`

> **📷 [SCREENSHOT PLACEHOLDER 14: Prometheus Targets Status (UP 1/1)]**  
> *Caption: Prometheus UI Targets Page Showing 'mlops' Target in Green UP State.*  
> *(Paste screenshot from Prometheus Status -> Targets showing UP).*

> **📷 [SCREENSHOT PLACEHOLDER 15: Prometheus Time-Series Graph Dashboard]**  
> *Caption: Prometheus Graph Tab Visualizing Live Metrics Query.*  
> *(Paste screenshot of the Prometheus Graph view).*

> **📷 [SCREENSHOT PLACEHOLDER 16: Raw Prometheus /metrics Endpoint Stream]**  
> *Caption: Browser View of http://192.168.49.2:31543/metrics.*  
> *(Paste screenshot of the raw /metrics output in browser).*

---

## 12. Conclusion & Deliverables Summary

* **GitHub Repository:** https://github.com/shradha1993/MLOps_Assignment_01
* **Deployed API Endpoint:** `http://192.168.49.2:31543` (Minikube NodePort)
* **Prometheus Dashboard:** `http://192.168.49.2:30090`
* **Demonstration Video Link:** `[Insert Video Recording Link Here]`
