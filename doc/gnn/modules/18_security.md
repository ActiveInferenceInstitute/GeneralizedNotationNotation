# Step 18: Security — Validation and Scanning

## Overview

Performs security validation and scanning of GNN pipeline artifacts, including dependency vulnerability checks and code security analysis.

## Usage

```bash
python src/18_security.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/18_security.py` (63 lines) |
| Module | `src/security/` |
| Processor | `src/security/processor.py` |
| Module function | `process_security()` |

## Output

- **Directory**: `output/18_security_output/`
- Security reports, vulnerability scanning results, and compliance checks

## Source

- **Script**: [src/18_security.py](../../src/18_security.py)
