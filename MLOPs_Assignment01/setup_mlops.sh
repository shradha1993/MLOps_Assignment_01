#!/bin/bash
# ============================================================
# 🚀 MLOps Auto‑Setup Script with Prometheus Deployment
# ============================================================

echo "🔧 Starting MLOps setup..."

# 1️⃣ Install kubectl (direct binary method)
echo "📦 Installing kubectl..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
echo "✅ kubectl installed successfully."
kubectl version --client

# 2️⃣ Start Minikube
echo "🚀 Starting Minikube..."
minikube start

# 3️⃣ Point Docker to Minikube’s environment
echo "🔗 Linking Docker to Minikube..."
eval $(minikube docker-env)

# 4️⃣ Build Docker image inside Minikube
echo "🐳 Building Docker image..."
docker build -t mlops_assignment:latest .

# 5️⃣ Apply FastAPI deployment
echo "📡 Applying FastAPI deployment..."
kubectl apply -f deployment.yaml
sleep 10
kubectl get pods

# 6️⃣ Create monitoring namespace (if not exists)
echo "🧩 Creating monitoring namespace..."
kubectl get namespace monitoring >/dev/null 2>&1 || kubectl create namespace monitoring

# 7️⃣ Deploy Prometheus
echo "📊 Deploying Prometheus..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 5s
    scrape_configs:
      - job_name: 'mlops'
        metrics_path: /metrics
        static_configs:
          - targets: ['mlops-service.default.svc.cluster.local:5001']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus
          args:
            - "--config.file=/etc/prometheus/prometheus.yml"
          volumeMounts:
            - name: prometheus-config-volume
              mountPath: /etc/prometheus
      volumes:
        - name: prometheus-config-volume
          configMap:
            name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: monitoring
spec:
  type: NodePort
  ports:
    - port: 9090
      targetPort: 9090
      nodePort: 30090
  selector:
    app: prometheus
EOF

# 8️⃣ Wait for Prometheus pod
echo "⏳ Waiting for Prometheus to start..."
sleep 15
kubectl get pods -n monitoring

# 9️⃣ Expose FastAPI service
echo "🌐 Exposing FastAPI service..."
minikube service mlops-service --url

# 🔟 Expose Prometheus dashboard
echo "📈 Opening Prometheus dashboard..."
minikube service prometheus-service -n monitoring --url

echo "🎉 Setup complete! FastAPI and Prometheus are live."