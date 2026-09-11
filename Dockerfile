# Dockerfile — linux/amd64 test image
FROM --platform=$BUILDPLATFORM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates python3-dev gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Install uv — pinned version for reproducible builds (supply-chain hardening).
# Pin matches the astral-sh/setup-uv version used in CI (0.12 series) so local
# and CI environments resolve the same lockfile format.
ARG UV_VERSION=0.12.0
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
ENV PATH="/root/.local/bin:$PATH"

# Create non-root user for runtime security
RUN groupadd --gid 1000 gnn && \
    useradd --uid 1000 --gid gnn --create-home gnn

WORKDIR /workspace

# Copy project (owned by root for build layer caching); .dockerignore keeps
# host-side .venv/__pycache__/output/ artifacts out of the build context.
COPY . /workspace

# Create venv and install deps via uv (as root for build performance).
# `uv sync --frozen --extra dev` is the single source of truth: it installs the
# project from the lockfile exactly as CI does. JAX (jax[cpu]>=0.7.0,<0.11) is a
# CORE dependency in pyproject.toml (lines 62-63), pulled in via the lock — no
# unpinned `uv pip install jax jaxlib` bypass, and no separate ml-ai extra is
# needed: the default-suite jax imports
# (tests/render/test_jax_renderer.py:267 "hard project dep; explicit import per
# zero-skip contract"; tests/pipeline/test_pomdp_pipeline_integration.py via
# gnn.utils.jax_stack_validation) are satisfied by the core dependency itself,
# and jax-importing tests beyond that are excluded by the CMD's marker filter.
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --frozen --extra dev
RUN chown -R gnn:gnn /workspace /opt/venv
USER gnn

# Default test command — mirrors justfile test-cov (justfile:36-38) and the
# ci.yml unit/integration job (ci.yml:131-136): same marker filter
# ("not pipeline and not mcp"), same cov target (--cov=gnn, not --cov=src),
# same llm ollama ignores; pytest-xdist (-n auto) comes from the dev extra.
CMD ["/bin/bash","-lc",". /opt/venv/bin/activate && python -m pytest tests/ -n auto --dist worksteal -m 'not pipeline and not mcp' --cov=gnn --cov-report=term-missing --cov-report=html:output/2_tests_output/htmlcov --ignore=tests/llm/test_llm_ollama.py --ignore=tests/llm/test_llm_ollama_integration.py --tb=short -q"]