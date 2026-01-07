# GNN Deployment Guide

> **📋 Document Metadata**  
> **Type**: Deployment Guide | **Audience**: DevOps & System Administrators | **Complexity**: Advanced  
> **Cross-References**: [Development Guide](../development/README.md) | [Configuration Guide](../configuration/README.md)

## Overview
This guide covers various deployment scenarios for GeneralizedNotationNotation (GNN), from local development to production environments.

## Deployment Architectures

### 1. Local Development Setup
**Best for**: Individual researchers, model development, learning

#### Quick Setup
```bash
# Clone and setup
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation
python src/main.py --only-steps 2  # Setup environment

# Run pipeline
python src/main.py --target-dir input/gnn_files/
```

#### Development Configuration
```yaml
# config.development.yaml
pipeline:
  log_level: "DEBUG"
  verbose: true
  parallel: false  # Easier debugging
validation:
  strict_mode: false
llm:
  default_provider: "local"  # Use free local models
```

### 2. Research Lab Server
**Best for**: Research teams, shared computing resources

#### Server Requirements
```yaml
# Minimum specifications
hardware:
  cpu_cores: 8
  memory_gb: 32
  storage_gb: 500
  gpu: optional (for JAX acceleration)

software:
  os: "Ubuntu 20.04+ or CentOS 8+"
  python: "3.8+"
  julia: "1.9+"
  graphviz: "2.40+"
```

#### Multi-User Setup
```bash
# Create shared GNN installation
sudo mkdir -p /opt/gnn
sudo chown gnn:gnn /opt/gnn
cd /opt/gnn

# Clone and setup
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git .
python src/main.py --only-steps 2

# Create user workspace template
mkdir -p /home/template/gnn_workspace
cp -r src/gnn/gnn_examples/* /home/template/gnn_workspace/
```

#### User Environment Script
```bash
#!/bin/bash
# /opt/gnn/setup_user.sh

USER_DIR="/home/$USER/gnn_workspace"
GNN_DIR="/opt/gnn"

# Create user workspace
if [ ! -d "$USER_DIR" ]; then
    cp -r /home/template/gnn_workspace "$USER_DIR"
    chown -R $USER:$USER "$USER_DIR"
fi

# Setup environment
export PYTHONPATH="$GNN_DIR/src:$PYTHONPATH"
export GNN_HOME="$GNN_DIR"
export GNN_WORKSPACE="$USER_DIR"

# Activate shared virtual environment
source $GNN_DIR/src/.venv/bin/activate

echo "GNN environment ready. Workspace: $USER_DIR"
```

### 3. Cloud Deployment (AWS)

#### EC2 Instance Setup
```yaml
# CloudFormation template snippet
Resources:
  GNNInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: m5.2xlarge  # 8 vCPU, 32GB RAM
      ImageId: ami-0abcdef1234567890  # Ubuntu 20.04 LTS
      SecurityGroups:
        - !Ref GNNSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          apt-get update
          apt-get install -y python3.8 python3-pip git graphviz
          
          # Install Julia
          wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.0-linux-x86_64.tar.gz
          tar xzf julia-1.9.0-linux-x86_64.tar.gz
          sudo mv julia-1.9.0 /opt/julia
          sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
          
          # Clone and setup GNN
          cd /opt
          git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git gnn
          cd gnn
          python3 src/main.py --only-steps 2
```

#### ECS Containerized Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.0-linux-x86_64.tar.gz \
    && tar xzf julia-1.9.0-linux-x86_64.tar.gz \
    && mv julia-1.9.0 /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && rm julia-1.9.0-linux-x86_64.tar.gz

# Copy GNN code
WORKDIR /app
COPY . .

# Install Python dependencies
RUN python src/main.py --only-steps 2

# Setup entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  gnn-processor:
    build: .
    volumes:
      - ./models:/app/models:ro
      - ./output:/app/output
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    command: ["python", "src/main.py", "--target-dir", "models/", "--output-dir", "output/"]
  
  gnn-mcp-server:
    build: .
    ports:
      - "8000:8000"
    command: ["python", "src/21_mcp.py", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Kubernetes Deployment

#### Namespace and ConfigMap
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: gnn-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gnn-config
  namespace: gnn-system
data:
  config.yaml: |
    pipeline:
      log_level: "INFO"
      parallel: true
      output_dir: "/data/output"
    llm:
      default_provider: "openai"
    export:
      formats: ["json", "xml"]
```

#### Deployment and Service
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gnn-processor
  namespace: gnn-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gnn-processor
  template:
    metadata:
      labels:
        app: gnn-processor
    spec:
      containers:
      - name: gnn
        image: gnn:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gnn-secrets
              key: openai-api-key
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: config
        configMap:
          name: gnn-config
      - name: data
        persistentVolumeClaim:
          claimName: gnn-storage

---
apiVersion: v1
kind: Service
metadata:
  name: gnn-service
  namespace: gnn-system
spec:
  selector:
    app: gnn-processor
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Secrets Management
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: gnn-secrets
  namespace: gnn-system
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  anthropic-api-key: <base64-encoded-key>
```

### 5. High-Performance Computing (HPC)

#### SLURM Job Script
```bash
#!/bin/bash
#SBATCH --job-name=gnn-pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --output=gnn_%j.log

# Load modules
module load python/3.9
module load julia/1.9
module load graphviz/2.50

# Setup environment
export TMPDIR=/scratch/$USER/gnn_temp_$SLURM_JOB_ID
mkdir -p $TMPDIR

# Clone GNN (or use pre-installed version)
cd $TMPDIR
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Setup and run
python src/main.py --only-steps 2
python src/main.py \
    --target-dir $HOME/gnn_models/ \
    --output-dir $HOME/gnn_results/run_$SLURM_JOB_ID/ \
    --config pipeline.max_processes=16

# Cleanup
rm -rf $TMPDIR
```

#### PBS Job Script
```bash
#!/bin/bash
#PBS -N gnn-pipeline
#PBS -l nodes=1:ppn=16
#PBS -l mem=64gb
#PBS -l walltime=04:00:00
#PBS -q normal

cd $PBS_O_WORKDIR

# Setup environment
module load python/3.9 julia/1.9

# Run GNN pipeline
python src/main.py --target-dir models/ --parallel
```

## Production Configuration

### Performance Optimization
```yaml
# config.production.yaml
pipeline:
  parallel: true
  max_processes: 8
  max_memory_gb: 16
  
validation:
  strict_mode: true
  
visualization:
  formats: ["svg"]  # Lighter than PNG/PDF
  dpi: 150  # Lower DPI for speed
  
llm:
  openai:
    model: "gpt-3.5-turbo"  # Faster than GPT-4
    max_tokens: 1000
    
jax_eval:
  jit_compile: true
  parallel_evaluation: true
```

### Security Configuration
```yaml
# config.security.yaml
mcp:
  require_auth: true
  api_key: "${MCP_API_KEY}"
  allowed_origins: ["https://yourdomain.com"]
  rate_limit: 10  # requests per minute

llm:
  content_filter: true
  max_retries: 1
  
pipeline:
  cleanup: true  # Remove temp files
  log_level: "WARNING"  # Minimal logging
```

## Monitoring and Observability

### Health Checks
```python
# healthcheck.py
import requests
import sys
import subprocess

def check_gnn_health():
    """Basic health check for GNN deployment"""
    
    # Check if main process responds
    try:
        result = subprocess.run(
            ["python", "src/main.py", "--validate-config"],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            return False, "Config validation failed"
    except subprocess.TimeoutExpired:
        return False, "Health check timeout"
    
    # Check MCP server if enabled
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            return False, "MCP server unhealthy"
    except requests.RequestException:
        pass  # MCP might be disabled
    
    return True, "Healthy"

if __name__ == "__main__":
    healthy, message = check_gnn_health()
    print(message)
    sys.exit(0 if healthy else 1)
```

### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
PIPELINE_RUNS = Counter('gnn_pipeline_runs_total', 'Total pipeline runs')
PIPELINE_DURATION = Histogram('gnn_pipeline_duration_seconds', 'Pipeline duration')
ACTIVE_JOBS = Gauge('gnn_active_jobs', 'Currently active jobs')
MODEL_PROCESSING_TIME = Histogram('gnn_model_processing_seconds', 'Model processing time')

class GNNMetrics:
    def __init__(self):
        # Start metrics server
        start_http_server(9090)
    
    def record_pipeline_run(self, duration, success=True):
        PIPELINE_RUNS.inc()
        PIPELINE_DURATION.observe(duration)
    
    def set_active_jobs(self, count):
        ACTIVE_JOBS.set(count)
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "GNN Pipeline Monitoring",
    "panels": [
      {
        "title": "Pipeline Runs per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gnn_pipeline_runs_total[1h])"
          }
        ]
      },
      {
        "title": "Average Processing Time",
        "type": "singlestat",
        "targets": [
          {
            "expr": "avg(gnn_pipeline_duration_seconds)"
          }
        ]
      },
      {
        "title": "Active Jobs",
        "type": "singlestat",
        "targets": [
          {
            "expr": "gnn_active_jobs"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Data Backup Strategy
```bash
#!/bin/bash
# backup_gnn.sh

BACKUP_DIR="/backups/gnn/$(date +%Y%m%d_%H%M%S)"
GNN_DIR="/opt/gnn"

mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r "$GNN_DIR/config"* "$BACKUP_DIR/"

# Backup processed models
cp -r "$GNN_DIR/output" "$BACKUP_DIR/"

# Backup custom code/extensions
cp -r "$GNN_DIR/src/custom" "$BACKUP_DIR/" 2>/dev/null || true

# Create archive
tar czf "$BACKUP_DIR.tar.gz" -C "$(dirname $BACKUP_DIR)" "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### Disaster Recovery Plan
```yaml
# disaster_recovery.yaml
recovery_procedures:
  data_loss:
    1. "Restore from latest backup"
    2. "Verify configuration integrity"
    3. "Test with simple model"
    4. "Resume normal operations"
  
  service_outage:
    1. "Check health endpoints"
    2. "Restart services in order: MCP -> Pipeline"
    3. "Validate with test requests"
    4. "Monitor logs for 1 hour"
  
  corruption:
    1. "Stop all services"
    2. "Restore from backup"
    3. "Run data integrity checks"
    4. "Restart services"

backup_schedule:
  frequency: "daily"
  retention: "30 days"
  verification: "weekly"
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# Scale based on queue length
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gnn-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gnn-processor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: queue_length
      target:
        type: AverageValue
        averageValue: "5"
```

### Load Balancing
```yaml
# nginx.conf for load balancing
upstream gnn_backends {
    least_conn;
    server gnn-1:8000 max_fails=3 fail_timeout=30s;
    server gnn-2:8000 max_fails=3 fail_timeout=30s;
    server gnn-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://gnn_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_timeout 300s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://gnn_backends;
    }
}
```

This deployment guide covers the major deployment scenarios for GNN, from simple local setups to enterprise-grade production deployments with monitoring, scaling, and disaster recovery capabilities. 