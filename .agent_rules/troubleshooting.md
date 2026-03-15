# Troubleshooting Guide

> **Quick health check**: `python src/main.py --verbose` | `python src/2_tests.py --fast-only`

## Expected Healthy Output

```
✅ Step 0  Template         [0.13s]
✅ Step 1  Setup            [2.46s]
✅ Step 2  Tests 1522/1526  [93.5s]
✅ Step 3  GNN 22 formats   [0.09s]
...
✅ All 25 steps — SUCCESS
Peak memory: ~36MB | Total: ~2m53s
```

---

## Common Issues by Category

### Import Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: pymdp` | Optional dep absent | `uv pip install inferactively-pymdp` — or ignore (expected) |
| `ModuleNotFoundError: flax` | Stale JAX code | Re-run `python src/11_render.py --target-dir input/gnn_files` |
| `ImportError: cannot import 'X'` | Version mismatch | `uv pip install -U -r requirements.txt` |
| `ModuleNotFoundError: src.gnn` | Wrong import style | Use `sys.path.insert` + direct import, not `from src.gnn import…` |

### Julia / RxInfer Issues

| Error | Fix |
|-------|-----|
| `Package RxInfer not found` | `julia -e 'using Pkg; Pkg.add("RxInfer")'` |
| `Julia not found in PATH` | Install from julialang.org, add to `~/.zshrc` |
| Slow first run | Normal — Julia compiles packages on first use (~2–5 min) |

### LLM / Ollama Issues

| Error | Fix |
|-------|-----|
| `Connection refused` | `ollama serve` (in another terminal) |
| `No models available` | `ollama pull smollm2:135m-instruct-q4_K_S` |
| LLM slow | `export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S` (smaller model) |

### Test Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Tests skipped | Optional deps absent | Expected — skip count is normal |
| `RecursionError` | Pytest plugin conflict | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest src/tests/` |
| Timeout | Tests take too long | `export FAST_TESTS_TIMEOUT=300` |
| `FileNotFoundError` | Wrong working directory | Run from project root: `/path/to/generalizednotationnotation` |

### MCP Issues

| Issue | Fix |
|-------|-----|
| Tool not found | `python -c "from mcp import MCPServer; s = MCPServer(); print(len(s.tools))"` |
| Slow module loading | Normal — 21+ modules load in ~10s |

### Performance Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Pipeline slow | Individual step bottleneck | Check `output/00_pipeline_summary/pipeline_execution_summary.json` |
| High memory | Memory leak in custom code | Use `memory_tracked()` context manager to isolate |
| Step hangs | No timeout | Set `STEP_TIMEOUT=120` env var |

---

## Diagnostic Commands

```bash
# Environment check
python --version             # Should be 3.10+
uv pip list | grep -E "numpy|scipy|matplotlib|jax|pytest"
julia --version              # Optional
ollama list                  # Optional

# File system
ls -la input/gnn_files/      # Verify input files exist
ls -la output/               # Verify output structure

# Pipeline summary
cat output/00_pipeline_summary/pipeline_execution_summary.json | python -m json.tool | grep -E "status|duration"

# Log analysis
grep -r "ERROR" output/*/  2>/dev/null | head -20
```

---

## Recovery Procedures

```bash
# Clean and restart
rm -rf output/*
python src/main.py --verbose

# Rebuild environment
rm -rf .venv
uv venv && uv pip install -e .

# Resume from checkpoint (skip completed steps)
python src/main.py --skip-steps "0,1,2,3"

# Run specific steps only
python src/main.py --only-steps "11,12,13"

# Run a single step directly
python src/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose
```

---

## Error Message Reference

| Message | Meaning | Action |
|---------|---------|--------|
| `PyMDP not available` | Optional dep missing | Install or ignore |
| `Flax not found` | Outdated generated code | Re-run render step |
| `Circuit breaker open` | Repeated execution failures | Check underlying framework issue |
| `Timeout expired` | Step exceeded time limit | Increase `STEP_TIMEOUT` or `STEP_TIMEOUT_LLM` |
| `uv: command not found` | uv not installed | Install from [astral.sh/uv](https://astral.sh/uv) |

---

**Last Updated**: March 2026 | **Status**: Production Standard
