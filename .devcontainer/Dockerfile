# GNN Pipeline — Development Container
# Based on: Python 3.11 + uv package manager
# See: .devcontainer/devcontainer.json

FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# Install uv (Astral's fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install just (command runner)
RUN curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Set working directory
WORKDIR /workspaces/GeneralizedNotationNotation

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen from lockfile)
RUN uv sync --frozen --extra dev || true

# Copy the rest of the project
COPY . .

# Ensure PYTHONPATH includes src/
ENV PYTHONPATH="/workspaces/GeneralizedNotationNotation/src"
