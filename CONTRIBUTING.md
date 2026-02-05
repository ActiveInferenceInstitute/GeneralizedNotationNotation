# Contributing to Generalized Notation Notation (GNN)

Thank you for your interest in contributing to GNN! This project benefits from contributions of all kinds — from bug reports and documentation improvements to new pipeline modules and framework integrations.

## Getting Started

### Prerequisites

- **Python 3.11+**
- **UV** package manager ([installation](https://docs.astral.sh/uv/))
- **Git** for version control

### Setup

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Install dependencies
uv sync

# Run the test suite to verify your setup
uv run pytest src/tests/ -v
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation instructions including optional dependency groups.

## How to Contribute

### Reporting Bugs

1. Search [existing issues](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues) first.
2. If no existing issue matches, open a new one with:
   - A clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - Python version, OS, and relevant dependency versions

### Suggesting Features

Open a [GitHub Discussion](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions) or issue describing:
- The problem the feature would solve
- Your proposed approach
- Any relevant examples or references

### Submitting Code Changes

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the conventions below.
3. **Run the tests** to ensure nothing is broken:
   ```bash
   uv run pytest src/tests/ -v
   ```
4. **Commit** with clear, descriptive messages:
   ```bash
   git commit -m "Add support for new matrix type in GNN parser"
   ```
5. **Push** to your fork and **open a Pull Request** against `main`.

## Code Conventions

### Architecture

All pipeline modules follow the **thin orchestrator pattern**:
- **Numbered scripts** (`src/N_module.py`): Handle CLI args, logging, and delegation only (<150 lines)
- **Module directories** (`src/module/`): Contain all domain logic in `processor.py`, public API in `__init__.py`

### Code Standards

- **Type hints** for all public functions
- **Exit codes**: 0 = success, 1 = error, 2 = warnings
- **No mocks** in tests — all tests must exercise real code paths
- Follow existing patterns — read neighboring modules before writing new ones

### Module Structure

Every module must include:
```
src/module_name/
  __init__.py      # Public API
  processor.py     # Core logic
  AGENTS.md        # Module documentation
  mcp.py           # MCP tool registration (if applicable)
```

### Testing

- Tests go in `src/tests/test_{module}_*.py`
- Run module-specific tests: `uv run pytest src/tests/test_gnn_*.py -v`
- Check coverage: `uv run pytest --cov=src --cov-report=term-missing`
- Aim for >80% test coverage on new code

### Documentation

- Update `AGENTS.md` in the relevant module directory
- If adding a pipeline step, update `AGENTS.md`, `DOCS.md`, and `README.md`
- Use concrete examples and real outputs over promotional language

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include a description of what changed and why
- Reference any related issues (e.g., "Fixes #42")
- Ensure all tests pass before requesting review
- Update documentation if your change affects user-facing behavior

## Style Guide

Follow the project [Style Guide](doc/style_guide.md) for formatting and naming conventions. Key points:
- Python code follows PEP 8
- GNN files use Markdown with structured sections (see [GNN Syntax](doc/gnn/gnn_syntax.md))
- Commit messages should be imperative ("Add feature" not "Added feature")

## Security

If you discover a security vulnerability, please follow the reporting process in [SECURITY.md](SECURITY.md). Do **not** open a public issue for security vulnerabilities.

## Code of Conduct

Please review and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive experience for everyone.

## Recognition

Contributors are recognized in [release notes](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases) and the [contributors graph](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/graphs/contributors).

## Questions?

- Check the [documentation](doc/)
- Open a [GitHub Discussion](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
- See [SUPPORT.md](SUPPORT.md) for additional help channels
