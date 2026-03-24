# doc/dev

Small development utilities and generated inventories for the documentation tree.

## Contents

| File | Purpose |
|------|---------|
| [regenerate_src_doc_inventory.py](regenerate_src_doc_inventory.py) | Regenerates `src_folder_doc_inventory.md` from `src/` |
| [src_folder_doc_inventory.md](src_folder_doc_inventory.md) | Which `src/` dirs lack `AGENTS.md` / `README.md` |
| [agents_test_coverage_note.md](agents_test_coverage_note.md) | Notes on agent-related test coverage |

Run the inventory script from the repository root:

```bash
uv run python doc/dev/regenerate_src_doc_inventory.py
```

See [../development/README.md](../development/README.md) for the full documentation audit tool. Doc hub: [../INDEX.md](../INDEX.md).

## Related

- [AGENTS.md](AGENTS.md) — technical signposting for this folder
