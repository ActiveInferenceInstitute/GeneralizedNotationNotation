# GNN round-trip testing

## Scope

[`test_round_trip.py`](test_round_trip.py) exercises **parse → serialize → parse** for a reference Markdown model and compares semantic content. Canonical counts (**23** enum formats, **22** serializers, round-trip list): **[../SPEC.md](../SPEC.md)**.

**Default `test_formats`:** **21** strings — **`markdown`** plus **20** conversion targets. **`ebnf`** and **`pnml`** are not in that list (`pnml` commented out; **EBNF** shares the BNF / `GrammarSerializer` path).

**Registries:** [`parsers/system.py`](../parsers/system.py) — **23** parsers, **22** serializers (no PNML serializer entry).

## Status

For the reference file and default configuration, the suite targets **100%** pass rate on the listed formats.

## Configuration

Edit `FORMAT_TEST_CONFIG` and `REFERENCE_CONFIG` at the top of [`test_round_trip.py`](test_round_trip.py):

- `test_formats` — which formats to exercise (must stay consistent with docs when you change it).
- `reference_file` — default `input/gnn_files/actinf_pomdp_agent.md` (relative to repo root).

`TEST_BEHAVIOR_CONFIG` controls checksums, timeouts, and cross-format validation.

## Embedded model data

Serializers embed JSON model snapshots in comments (or equivalent) so the second parse can restore fields that would otherwise be lossy. Patterns live in the various `*_serializer.py` files under [`../parsers/`](../parsers/).

## Running

```bash
# From repo root (example)
uv run python -m pytest src/gnn/testing/test_round_trip.py -q
```

Or run the module’s main block if defined for your workflow (see file docstring).

## Historical note

Round-trip pass rates for the reference model were improved in stages (early 2025); the current default list and embedded-data approach are the maintained baseline. Treat performance and checksum lines in old notes as **non-authoritative** unless reproduced with the same `test_round_trip.py` revision and machine.

---

**Maintained with:** `src/gnn/testing/test_round_trip.py`  
**See also:** [`../alignment_status.md`](../alignment_status.md), [`../SPEC.md`](../SPEC.md)
