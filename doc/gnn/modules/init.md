# `__init__.py` — Core Package Module

## Overview

The `src/__init__.py` file defines the GNN pipeline core package. It provides package-level metadata, version information, and automatic submodule discovery.

## Key Exports

| Export | Type | Description |
|--------|------|-------------|
| `__version__` | `str` | Package version (`1.3.0`) |
| `FEATURES` | `dict` | Feature flags (`pipeline_orchestration`, `mcp_integration`) |
| `get_module_info()` | `function` | Returns package metadata and discovered modules |
| `sapf` | `module` | Structured Audio Processing Framework (lazy-loaded) |

## Module Discovery

The `_discover_top_level_modules()` function scans `src/` for subdirectories containing an `__init__.py` file, returning a sorted list of all available module packages. This enables dynamic registration of pipeline modules without hardcoded imports.

## Source

- **Script**: [src/\_\_init\_\_.py](../../../src/__init__.py)
- **Lines**: 75
