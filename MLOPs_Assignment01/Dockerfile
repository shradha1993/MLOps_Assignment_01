FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by MLflow + sklearn + scipy + xgboost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libnss3 \
    libgfortran5 \
    libopenblas-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy model artifacts
COPY deployment/model_export /app/model_export

# Install Python dependencies (including MLflow)
RUN pip install --no-cache-dir -r /app/model_export/requirements.txt

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "/app/model_export", "--host", "0.0.0.0", "--port", "5001", "--env-manager", "local"]
