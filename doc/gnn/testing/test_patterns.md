# GNN Test Patterns and Standards

Real conventions drawn from `src/tests/conftest.py`, `src/tests/runner.py`, and test files.

**Last Updated**: March 6, 2026

## No-Mock Policy

All 91 test files follow a strict **no-mocks** policy enforced by code review and stated in `TEST_SUITE_SUMMARY.md`:

- ❌ No `unittest.mock` imports or usage
- ❌ No monkeypatching of functions or classes
- ✅ Real code paths executed in every test
- ✅ Real data — representative GNN files from `src/tests/test_data/`
- ✅ Real dependencies — skip if unavailable via `pytest.importorskip()`, never stub
- ✅ File-based assertions on real output artifacts

```python
# ✅ CORRECT — detect real state
def test_audio_backend():
    backends = pytest.importorskip("audio.backends")
    result = backends.check_backends()
    assert isinstance(result, dict)
    assert "available" in result  # may say "unavailable" — that is valid

# ❌ WRONG — mock the dep away
@patch("audio.backends.soundfile", None)
def test_audio_backend_missing():
    ...
```

## Pytest Markers (21 defined in conftest.py)

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for component interactions |
| `performance` | Performance and resource usage tests |
| `slow` | Tests taking significant time |
| `fast` | Quick tests for rapid feedback |
| `safe_to_fail` | Can run without side effects (336 functions) |
| `destructive` | May modify system state |
| `external` | Requires external dependencies |
| `core` | Core module tests |
| `pipeline` | Pipeline infrastructure tests |
| `recovery` | Pipeline recovery tests |
| `utilities` | Utility function tests |
| `environment` | Environment validation tests |
| `render` | Rendering and code generation tests |
| `export` | Export functionality tests |
| `parsers` | Parser and format tests |
| `main_orchestrator` | Main orchestrator tests |
| `type_checking` | Type checking tests |
| `mcp` | Model Context Protocol tests |
| `sapf` | SAPF audio generation tests |
| `visualization` | Visualization tests |

```bash
# Selective execution by marker
PYTHONPATH=src python -m pytest src/tests/ -m fast -q
PYTHONPATH=src python -m pytest src/tests/ -m "not slow" -q
PYTHONPATH=src python -m pytest src/tests/ -m "integration and not slow" -q
```

## Fixture Convention (conftest.py)

All shared fixtures live in `conftest.py` (601 lines). Standard fixtures:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `tmp_path` | function | Built-in pytest temp directory |
| `sample_gnn_file` | session | `input/gnn_files/actinf_pomdp_agent.md` |
| `test_data_dir` | session | `src/tests/test_data/` |
| `output_dir` | session | `output/2_tests_output/` |
| `mcp_server_tools` | module | Live MCP server — all 131 tools registered |

## Resource Monitoring

`TestRunner` in `infrastructure/test_runner.py` wraps every test run with `ResourceMonitor`:

```python
# infrastructure/resource_monitor.py (via psutil if available)
class ResourceMonitor:
    def start_monitoring(self): ...  # spins up a thread tracking memory/CPU
    def stop_monitoring(self): ...
    def get_stats(self) -> dict:     # returns {"peak_memory_mb": float, "avg_cpu": float}
        ...
```

Memory limit default: 300 MB. CPU limit: default off. Both are configurable via `TestExecutionConfig`.

## Writing Tests

### Parametrize for matrix coverage

```python
@pytest.mark.parametrize("export_format", ["json", "xml", "graphml", "gexf", "pickle"])
def test_export_format(export_format, tmp_path, sample_gnn_file):
    result = process_export(sample_gnn_file, tmp_path, format=export_format)
    assert result["success"] is True, f"process_export failed for {export_format}"
    assert (tmp_path / f"model.{export_format}").exists()
```

### Descriptive assertions with context

```python
# ✅ Good — fails with enough context to diagnose
result = process_validation(gnn_dir, output_dir, logger)
assert result is True, f"process_validation returned {result!r} — check {output_dir}"

# ❌ Bad — silent failure 
assert process_validation(gnn_dir, output_dir, logger)
```

### Optional dependency handling

```python
def test_torch_rendering():
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    rendered = render_gnn_to_pytorch(sample_content)
    assert "import torch" in rendered
```

## File Naming Conventions

| Pattern | Category | Example count |
|---------|----------|--------------|
| `test_*_overall.py` | Per-module system tests | 20 files |
| `test_pipeline_*.py` | Pipeline orchestration | 15 files |
| `test_execute_pymdp_*.py` | PyMDP simulation | 8 files |
| `test_mcp_*.py` | MCP layer | 5 files |
| `test_environment_*.py` | Environment setup | 5 files |
| `test_*_integration.py` | Cross-module | 6 files |

## Coverage Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--tb=short"
testpaths = ["src/tests"]

[tool.coverage.run]
source = ["src"]
omit = ["src/tests/*", "src/**/__pycache__/*"]
```

Generate HTML report:

```bash
PYTHONPATH=src python -m pytest src/tests/ \
  --cov=src --cov-report=html --cov-report=term-missing -q
# → htmlcov/index.html
```

## See Also

- [testing/README.md](README.md) — test suite overview + category table
- [testing/mcp_audit.md](mcp_audit.md) — MCP audit anatomy
- [`src/tests/TEST_SUITE_SUMMARY.md`](../../../src/tests/TEST_SUITE_SUMMARY.md) — canonical reference
- [`src/tests/conftest.py`](../../../src/tests/conftest.py) — all fixtures and markers
