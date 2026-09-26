# Website Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/website/`.

## Purpose
- Validate real website generation, static artifact assembly, and pipeline-output discovery.
- Keep tests aligned with `src/gnn/website/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.
- Pin the composability seams arriving with the dict-driven website API: the `collection → steps` cycle-break identity (`collection.PIPELINE_STEPS is steps.PIPELINE_STEPS`), the pure no-filesystem `website_data_from_dict` + `generate_website(filesystem=False)` contract, and the `SUPPORTED_FILE_TYPES` single-source pin (`renderer.py` definition, `__init__` re-export, derived `get_supported_file_types`/`get_module_info`).

## Verification
Run `uv run --extra dev python -m pytest tests/website/ -q`.
