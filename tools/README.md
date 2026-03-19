# Maintainer tools

Small utilities that are **not** part of the numbered pipeline (`src/main.py`).

| Script | Purpose |
|--------|---------|
| [`sync_agents_exports.py`](sync_agents_exports.py) | Refresh the `### Export surface (\`__all__\`)` block in every `AGENTS.md` that sits next to an `__init__.py` under `src/`. |

## Usage

```bash
uv run python tools/sync_agents_exports.py
```

Run after changing any package’s `__all__` so agent docs stay aligned with the real export surface.
