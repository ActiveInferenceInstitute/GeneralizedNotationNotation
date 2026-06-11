# Test Infrastructure — Technical Specification

**Version**: 1.6.0

## Test Runner

- Custom pytest runner (`test_runner.py`)
- Progress tracking with visual indicators
- Resource monitoring (memory, CPU) via `resource_monitor.py`
- HTML + JSON report generation via `report_generator.py`

## Test Configuration

- `test_config.py` detects: GPU availability, Ollama presence, network connectivity
- Auto-applies pytest markers: `@pytest.mark.gpu`, `@pytest.mark.llm`, `@pytest.mark.network`

## Utility Functions

- `utils.py` (365 lines) — Timer utilities, path helpers, assertion wrappers
