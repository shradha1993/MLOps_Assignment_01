# ============================================================
# 🚀 Dockerfile for FastAPI + Prometheus monitoring
# ============================================================

# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn prometheus_client

# Expose FastAPI port
EXPOSE 5001

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]