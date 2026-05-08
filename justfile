# GNN Pipeline — Developer Command Reference
# See: AGENTS.md, pyproject.toml, .github/workflows/ci.yml
#
# Install: brew install just  (or: cargo install just)
# Usage:   just <recipe>       (or: just --list)

# Default recipe: show available commands
default:
    @just --list --unsorted

# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

# Run fast test suite (default)
test:
    uv run pytest src/tests/ -q --tb=short -x

# Run full test suite (with Ollama ignores)
test-full:
    uv run pytest src/tests/ -q --tb=no \
        --ignore=src/tests/test_llm_ollama.py \
        --ignore=src/tests/test_llm_ollama_integration.py

# Run tests for a specific module (e.g., just test-mod render)
test-mod MODULE:
    uv run pytest src/tests/test_{{ MODULE }}*.py -v

# Run tests with coverage report
test-cov:
    uv run pytest src/tests/ --cov=src --cov-report=term-missing \
        --ignore=src/tests/test_llm_ollama.py \
        --ignore=src/tests/test_llm_ollama_integration.py

# ─────────────────────────────────────────────
# Linting & Formatting
# ─────────────────────────────────────────────

# Run ruff linter
lint:
    uv run ruff check src/

# Run ruff linter with auto-fix
lint-fix:
    uv run ruff check src/ --fix

# Format code with ruff
format:
    uv run ruff format src/
    uv run ruff check src/ --select I --fix

# Check formatting without modifying files
format-check:
    uv run ruff format --check src/

# Run mypy type checking
typecheck:
    uv run mypy src/ --ignore-missing-imports

# Run bandit security scan
security:
    uv run bandit -r src/ -c pyproject.toml -q

# ─────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────

# Run full pipeline
pipeline:
    uv run python src/main.py --target-dir input/gnn_files --verbose

# Run specific pipeline steps (e.g., just pipeline-steps "3,5,7,8")
pipeline-steps STEPS:
    uv run python src/main.py --only-steps "{{ STEPS }}" --target-dir input/gnn_files --verbose

# Run a single pipeline step (e.g., just step 3)
step N:
    uv run python src/{{ N }}_*.py --target-dir input/gnn_files --output-dir output --verbose

# ─────────────────────────────────────────────
# Renderer Operations
# ─────────────────────────────────────────────

# Check renderer availability
render-health:
    PYTHONPATH=src uv run python -c "from render.health import check_renderers; \
        statuses = check_renderers(); \
        [print(f'  {\"✅\" if s.available else \"❌\"} {s.name}') for s in statuses.values()]"

# Render and execute for specific frameworks (e.g., just render-exec "pymdp,jax")
render-exec FRAMEWORKS:
    uv run python src/main.py --only-steps "11,12" \
        --frameworks "{{ FRAMEWORKS }}" \
        --target-dir input/gnn_files --verbose

# ─────────────────────────────────────────────
# Documentation & Audit
# ─────────────────────────────────────────────

# Run documentation audit
audit:
    uv run python doc/development/docs_audit.py

# Count test files and items
test-count:
    @echo "Test files:"
    @find src/tests -maxdepth 1 -name 'test_*.py' | wc -l
    @echo "Collected test items:"
    @uv run pytest src/tests/ --collect-only -q 2>/dev/null | tail -1

# ─────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────

# Install all dependencies (including dev extras)
setup:
    uv sync --extra dev

# Recreate the virtual environment from scratch
setup-clean:
    rm -rf .venv
    uv sync --extra dev

# Validate the JAX + PyMDP stack
validate-stack:
    uv run python -c "from utils.jax_stack_validation import verify_jax_pymdp_stack; \
        verify_jax_pymdp_stack(); print('✅ JAX + PyMDP stack OK')"
