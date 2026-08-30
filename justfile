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

# Run tests with the same marker filter as CI, so local and CI
# exercise the same surface; use test-stopfast or test-full for other scopes.
test:
    uv run pytest src/tests/ -q --tb=short -m "not pipeline and not mcp"

# Run fast test suite, stop at first failure
test-stopfast:
    uv run pytest src/tests/ -q --tb=short -x -m "not pipeline and not mcp"

# Run full test suite (with Ollama ignores)
test-full:
    uv run pytest src/tests/ -q --tb=no \
        --ignore=src/tests/llm/test_llm_ollama.py \
        --ignore=src/tests/llm/test_llm_ollama_integration.py

# Run tests for a specific module (e.g., just test-mod render)
test-mod MODULE:
    uv run pytest src/tests/{{ MODULE }}/ -v

# Run tests with coverage report
test-cov:
    uv run pytest src/tests/ --cov=src --cov-report=term-missing \
        --ignore=src/tests/llm/test_llm_ollama.py \
        --ignore=src/tests/llm/test_llm_ollama_integration.py

# ─────────────────────────────────────────────
# Linting & Formatting
# ─────────────────────────────────────────────

# Run ruff linter
lint:
    uv run ruff check src scripts

# Run ruff linter with auto-fix
lint-fix:
    uv run ruff check src scripts --fix

# Format code with ruff
format:
    uv run ruff format src scripts
    uv run ruff check src scripts --select I --fix

# Check formatting without modifying files
format-check:
    uv run ruff format --check src scripts

# Run mypy type checking
typecheck:
    uv run mypy src --show-error-codes

# Run bandit security scan (same thresholds as CI)
security:
    uv run bandit -r src -c pyproject.toml -q --severity-level medium --confidence-level medium

# Run MCP + skills resolvability health gate
skills-health:
    uv run --extra dev python scripts/check_mcp_skills_health.py

# Run capability contract audit
capability:
    uv run python scripts/check_capability_contracts.py

# Run manuscript token audit
tokens:
    uv run python scripts/check_manuscript_tokens.py

# Run POMDP gridworld outputs check
gridworld:
    uv run python scripts/check_pomdp_gridworld_outputs.py

# Emit durable v3 run manifests for a completed run (e.g. just manifest output)
manifest OUT:
    uv run python scripts/emit_run_manifest.py {{ OUT }}

# Run maintained-doc terminology audit
doc-terms:
    uv run python scripts/check_maintained_doc_terms.py --strict

# Run documentation contract audit
doc-contracts:
    uv run python scripts/check_doc_contracts.py --strict

# Run GNN documentation pattern audit
doc-patterns:
    uv run python scripts/check_gnn_doc_patterns.py --strict

# Run fast quality gates without the full pytest suite
quality: format-check lint terminology doc-terms audit doc-contracts doc-patterns typecheck security

# Run focused PyMDP/POMDP behavior checks
test-pymdp-focused:
    uv run pytest \
        src/tests/execute/test_pymdp_contracts.py \
        src/tests/execute/test_discrete_models_pymdp.py \
        src/tests/visualization/test_visualization_matrices.py \
        -q --tb=short

# Collect pytest inventory without executing tests
test-collect:
    uv run pytest --collect-only src/tests/ -q --tb=no \
        --ignore=src/tests/llm/test_llm_ollama.py \
        --ignore=src/tests/llm/test_llm_ollama_integration.py

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

# Run documentation audit without mutating reports
audit:
    uv run python doc/development/docs_audit.py --strict --check-anchors --no-write

# Run maintained-tree terminology audit
terminology:
    uv run python scripts/check_repo_terminology.py --strict


# Count test files and items
test-count:
    @echo "Test files:"
    @find src/tests -name 'test_*.py' | wc -l
    @echo "Collected test items:"
    @uv run pytest --collect-only src/tests/ -q --tb=no \
        --ignore=src/tests/llm/test_llm_ollama.py \
        --ignore=src/tests/llm/test_llm_ollama_integration.py 2>/dev/null | tail -1

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
    PYTHONPATH=src uv run python -c "from utils.jax_stack_validation import verify_jax_pymdp_stack; \
        verify_jax_pymdp_stack(); print('✅ JAX + PyMDP stack OK')"

# ─────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────

# Run performance benchmark tests (pipeline performance group)
bench:
    uv run pytest src/tests/pipeline/test_pipeline_performance.py \
        -v --tb=short \
        -m "performance" \
        --benchmark-save=baseline

# Run performance tests and compare against saved baseline
bench-compare:
    uv run pytest src/tests/pipeline/test_pipeline_performance.py \
        -v --tb=short \
        -m "performance" \
        --benchmark-compare=baseline

# ─────────────────────────────────────────────
# Step Registry
# ─────────────────────────────────────────────

# List all 25 pipeline steps from the canonical registry
steps:
    #!/usr/bin/env bash
    export PYTHONPATH=src
    uv run python - <<'PYEOF'
    from pipeline.step_registry import STEPS
    for s in STEPS:
        tags = ','.join(sorted(s.tags))
        print(f'{s.script_name:30s} {tags}')
    print()
    print(f'Total: {len(STEPS)} steps')
    print(f"Core:  {len([s for s in STEPS if 'core' in s.tags])}")
    print(f"LLM:   {len([s for s in STEPS if 'llm' in s.tags])}")
    PYEOF

# Export step registry as JSON (useful for tooling)
steps-json:
    #!/usr/bin/env bash
    export PYTHONPATH=src
    uv run python - <<'PYEOF'
    import json
    from pipeline.step_registry import STEPS
    data = [{'script_name': s.script_name, 'description': s.description,
             'module_function': s.module_function, 'tags': sorted(s.tags)}
            for s in STEPS]
    print(json.dumps(data, indent=2))
    PYEOF
