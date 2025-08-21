# Dockerfile — linux/amd64 test image
FROM --platform=$BUILDPLATFORM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates python3-dev gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /workspace

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install minimal deps
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt || true
RUN pip install pytest pytest-cov pytest-asyncio pytest-json-report

# Default test command
CMD ["/bin/bash","-lc",". /opt/venv/bin/activate && python -m pytest -n0 --cov=src --cov-report=html:output/2_tests_output/singleproc_htmlcov --tb=short --maxfail=10 -q"]
