# tools

## Overview

Repository-maintainer scripts (not pipeline steps).

**Status**: Maintainer utilities  
**Version**: 1.0  

## Contents

| File | Role |
|------|------|
| [README.md](README.md) | How to run each script |
| [sync_agents_exports.py](sync_agents_exports.py) | Sync `__all__` → `src/**/AGENTS.md` export blocks |

## Integration

- Does **not** register MCP tools or appear in `src/main.py`.
- Safe to run locally; only modifies files containing the export-surface markers (or inserts one section after `## API Reference` when present).

---

**Maintenance**: Extend this folder when adding new opt-in automation; keep pipeline logic in `src/`.
