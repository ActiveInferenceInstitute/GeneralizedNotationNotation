# Processor.py Facade Census

Derived 2026-09-22 from the tree at `17bd6fd30` (URL repoint #163). This
reconstructs the inventory behind scope item Back#6 ("9 processor.py
facades — keep them thin, custody owners"), whose detailed scout was not
persisted. Classification is evidence-first: a `processor.py` is a
**pure delegation facade** when it contains zero own `def`/`class` and
only re-exports sibling symbols (`__all__` + imports); a **composition
orchestrator** when it composes sibling implementation modules but adds
little of its own; otherwise a **logic holder**.

## Census (22 modules with a `processor.py`)

| Module | Lines | Class | Delegation target | Thin-still-true |
|---|---|---|---|---|
| website | 13 | pure facade | `renderer.process_website` | yes |
| type_checker | 30 | pure facade | `.checking` subpackage (10 symbols) | yes |
| visualization | 41 | pure facade | `core.process`, `plotting`, `parse`, `matrix.compat`, `analysis` | yes |
| integration | 219 | composition | `.parsing`, `.graph` (docstring: "composes the pure parsing/graph primitives") | yes |
| mcp | 322 | composition | per-module `<name>.mcp` discovery + `mcp_instance` registration | yes |
| report | 419 | logic holder | — (imports only `gnn.utils.observability`) | n/a |
| api | 457 | logic holder | helpers from `models`/`path_utils`/`pipeline_runner` only | n/a |
| advanced_visualization | 510 | composition | `._shared`, `.network_viz`, `.statistical_viz`, `.interactive_viz` (docstring: "public processing facade") | yes |
| processing | 515 | logic holder | canonical engine (`GNNProcessor`, directory wrappers) | n/a |
| template | 525 | logic holder | — (imports only `gnn.utils`) | n/a |
| gui | 555 | logic holder | `.backend`/`.runner` helpers; `process_gui` logic inline | n/a |
| ml_integration | 634 | logic holder | — (zero sibling imports) | n/a |
| research | 737 | logic holder | — (only sibling is `mcp.py`) | n/a |
| audio | 746 | logic holder | `.generator`/`.streaming` helpers; analyzer logic inline | n/a |
| analysis | 771 | logic holder | `.analyzer`/`.framework_common` helpers; orchestration inline | n/a |
| export | 992 | logic holder | `.formatters`/`.registry` helpers; format logic inline | n/a |
| ontology | 1011 | logic holder | — (only sibling is `utils.py`) | n/a |
| llm | 1024 | logic holder | `.analyzer`/`.cache`/`.generator`/`.llm_processor`/`.prompts` helpers; engine logic inline | n/a |
| security | 1235 | logic holder | — (only sibling is `mcp.py`) | n/a |
| render | 1314 | logic holder | `.framework_registry`/`.naming` helpers; rendering logic inline | n/a |
| intelligent_analysis | 1409 | logic holder | `.analyzer` helper; analysis logic inline | n/a |
| execute | 1552 | logic holder | `.data_extractors`/`.detection`/`.julia_env`/`.metadata`/`.security_gate` helpers; execution logic inline | n/a |

Line counts: `wc -l src/gnn/*/processor.py` at `17bd6fd30`.

## The three candidate lists and the 9-derivation

1. **Pure-facades** (zero own `def`/`class`, `__all__` re-exports only):
   `website`, `type_checker`, `visualization` — **3-strong**, not 9.
2. **Orphan facades** (module `__init__.py` imports zero symbols from
   `processor.py` AND the file is facade-shaped): **`website` only** —
   1-strong, not 9. (The raw zero-import set is 3-strong: `api`,
   `type_checker`, `website`; but `type_checker` and `api` are consumed
   directly, see below.)
3. **Top-9 by line count** (largest first): `execute` (1552),
   `intelligent_analysis` (1409), `render` (1314), `security` (1235),
   `llm` (1024), `ontology` (1011), `export` (992), `analysis` (771),
   `audio` (746) — **exactly 9-strong**, but every member is a *logic
   holder*, not a facade.

**Verdict:** no exactly-9-strong list of *facades* exists at HEAD. The
only evidence-derived 9-strong list is the top-9-by-line-count set, whose
members are the nine heaviest logic holders — the opposite of thin
facades. Back#6's "9 processor.py facades (keep them thin, custody
owners)" therefore cannot be a facade inventory; the defensible reading
is that these nine files are the *custody-heavy* processor.py files whose
line bands must be watched, while the "keep them thin" rule actually
applies to the **6-strong facade-shaped set** (3 pure + 3 composition),
all of which are still thin at HEAD. Wave work under Back#6 should treat
these as two separate inventories.

## Orphan-facade verdicts (consumer greps at HEAD, no deletions)

- `website/processor.py` — **true orphan facade**. `website/__init__.py`
  imports `process_website` directly from `renderer` (lines 24-34), and a
  repo-wide grep for consumers of `gnn.website.processor` returns zero
  hits. Its own docstring states it "exists for architectural
  consistency: the documented pattern expects a processor.py in every
  module directory" (`website/processor.py:5-7`). **Keep as-is** —
  deleting it would violate the documented per-module pattern for a
  13-line file.
- `type_checker/processor.py` — zero `__init__` imports but **live**:
  `src/gnn/type_checker/cli.py:34`, `tests/type_checker/test_type_checker_overall.py:12`,
  `tests/api/test_comprehensive_api.py:315`. Not an orphan.
- `api/processor.py` — zero `__init__` imports but **live** (and a logic
  holder, not a facade): `src/gnn/api/mcp.py:17`, `src/gnn/api/app.py:55`,
  `tests/api/test_api_mcp_tools.py:159,165`. Not an orphan.

## Related note

The processing package's own split between the canonical
`processor.py` wrappers and the `core_processor.py` recovery wrappers is
documented in `src/gnn/processing/README.md` ("Core-Processor Wrappers");
the dead `create_processor` factory was removed from `core_processor.py`
in the same pass (zero consumers at HEAD).

## Top-9 processor.py logic holders — Back#6 verdict (per-module)

No tracked `src/gnn` file exceeds the 2000-line MAJ-04 band at HEAD
(`wc -l` corroboration: empty output for files >2000; receipt
`TO-DO.md:84,98` — `oversized_module_lines` 13878 → 0, MAJ-04 closed
2026-09-07), so no rewrite is mandated by the split pattern; the
keep-thin rule binds the 6-strong facade-shaped set (3 pure + 3
composition) from the first table, all verified thin.

| Module | Lines | MAJ-04 2000-line band | Delegation targets available (subdirs / sibling .py) | Verdict |
|---|---|---|---|---|
| execute | 1552 | in-band | 13 / 15 | keep as logic holder |
| intelligent_analysis | 1409 | in-band | 1 / 3 | keep as logic holder |
| render | 1314 | in-band | 10 / 17 | keep as logic holder |
| security | 1235 | in-band | 1 / 0 | keep as logic holder |
| llm | 1024 | in-band | 2 / 7 | keep as logic holder |
| ontology | 1011 | in-band | 1 / 1 | keep as logic holder |
| export | 992 | in-band | 1 / 10 | keep as logic holder |
| analysis | 771 | in-band | 8 / 21 | keep as logic holder |
| audio | 746 | in-band | 3 / 6 | keep as logic holder |

Subdir / sibling counts measured at HEAD, excluding `__init__.py`,
`processor.py`, and `mcp.py`.
