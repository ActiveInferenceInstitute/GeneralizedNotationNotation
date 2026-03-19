# pipeline (documentation)

## Overview

Canonical documentation for running and configuring the GNN pipeline (steps 0–24), hosted under `doc/pipeline/`.

**Status**: Documentation  
**Version**: 1.0  

---

## Purpose

- Execution-framework setup (optional Python backends for step 12)
- Pointers to step scripts, orchestrator, and configuration

This tree is part of the GNN documentation system; implementation lives in `src/`.

## Contents

| File | Role |
|------|------|
| [README.md](README.md) | Human entrypoint: `uv`, frameworks, links to `STEP_INDEX` / `main.py` |

## Quick navigation

- **Step index**: [src/STEP_INDEX.md](../../src/STEP_INDEX.md)
- **Orchestrator**: [src/main.py](../../src/main.py)
- **Script catalog**: [PIPELINE_SCRIPTS.md](../PIPELINE_SCRIPTS.md)
- **Pipeline module**: [src/pipeline/AGENTS.md](../../src/pipeline/AGENTS.md)

## Integration

- **25 steps (0–24)**: thin orchestrators in `src/N_*.py`
- **Routing**: `input/config.yaml` (`testing_matrix`)

## Standards

Same conventions as other `doc/*` manifests: concrete paths, no duplicate claims about code that lives elsewhere.

---

**Maintenance**: Update when step count, optional execution extras, or config keys change.
