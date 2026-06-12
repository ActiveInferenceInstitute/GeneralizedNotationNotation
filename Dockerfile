# Dockerfile — linux/amd64 test image
FROM --platform=$BUILDPLATFORM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates python3-dev gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Install uv — pinned version for reproducible builds (supply-chain hardening)
ARG UV_VERSION=0.7.8
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
ENV PATH="/root/.local/bin:$PATH"

# Create non-root user for runtime security
RUN groupadd --gid 1000 gnn && \
    useradd --uid 1000 --gid gnn --create-home gnn

WORKDIR /workspace

# Copy project (owned by root for build layer caching)
COPY . /workspace

# Create venv and install deps via uv (as root for build performance)
RUN uv venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Install project and test dependencies via uv
RUN uv pip install . || true
RUN uv pip install pytest pytest-cov pytest-asyncio pytest-json-report
RUN uv pip install jax jaxlib

# Verify XLA compile compatibility
RUN python -c "import jax, jax.numpy as jnp; jax.jit(lambda x: x * 2)(jnp.ones(10)).block_until_ready()"

# Transfer ownership and switch to non-root user
RUN chown -R gnn:gnn /workspace /opt/venv
USER gnn

# Default test command
CMD ["/bin/bash","-lc",". /opt/venv/bin/activate && python -m pytest -n0 --cov=src --cov-report=html:output/2_tests_output/singleproc_htmlcov --tb=short --maxfail=10 -q"]
