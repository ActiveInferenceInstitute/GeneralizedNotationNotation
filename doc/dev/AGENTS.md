# doc/dev

## Overview

Holds lightweight scripts and generated markdown inventories used to track documentation coverage under `src/`.

## Contents

- `regenerate_src_doc_inventory.py` — scans `src/` for package-like directories missing `AGENTS.md` or `README.md`
- `src_folder_doc_inventory.md` — output of that script (regenerate after structural changes)
- `agents_test_coverage_note.md` — supplementary notes

## Usage

From repository root:

```bash
uv run python doc/dev/regenerate_src_doc_inventory.py
```

Full-repo markdown link and pairing audits: [../development/docs_audit.py](../development/docs_audit.py).

## Related

- [README.md](README.md) — directory overview
- [../development/AGENTS.md](../development/AGENTS.md) — development documentation module
