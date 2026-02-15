# Dockerfile — linux/amd64 test image
FROM --platform=$BUILDPLATFORM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates python3-dev gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Install uv (repository-standard package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project
COPY . /workspace

# Create venv and install deps via uv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Install project and test dependencies via uv
RUN uv pip install . || true
RUN uv pip install pytest pytest-cov pytest-asyncio pytest-json-report

# Default test command
CMD ["/bin/bash","-lc",". /opt/venv/bin/activate && python -m pytest -n0 --cov=src --cov-report=html:output/2_tests_output/singleproc_htmlcov --tb=short --maxfail=10 -q"]
