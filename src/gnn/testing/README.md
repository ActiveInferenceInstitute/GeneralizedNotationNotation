# Testing and Benchmarks

This directory holds round-trip tests, integration tests, and benchmarks for the GNN module.

- **`test_round_trip.py`** — Round-trip harness (`GNNRoundTripTester`). Default config lists **21** format strings in `FORMAT_TEST_CONFIG['test_formats']` (**`markdown`** plus **20** targets). See **[../SPEC.md](../SPEC.md)** for how that relates to all **23** `GNNFormat` values and **22** serializers.
- **`README_round_trip.md`** — Methodology, configuration, and historical notes.
- **`round_trip_reports/`** — Generated reports when enabled.
- **`performance_benchmarks.py`**, **`test_*.py`** — Other tests as named.

## Round-trip suite status

For the reference model (`input/gnn_files/actinf_pomdp_agent.md`) and the default `test_formats` list, the suite is configured to report **100%** pass rate.

**Outside the default round-trip list:** `ebnf` (shares BNF / `GrammarSerializer` machinery; not a separate row in `test_formats`), `pnml` (disabled in config; parse-focused). See `FORMAT_TEST_CONFIG` in `test_round_trip.py`.

### Categories (20 conversion targets + markdown reference)

- **Schema (7):** JSON, XML, YAML, PKL, ASN.1, Protobuf, XSD  
- **Languages (6):** Scala, Python, Lean, Coq, Isabelle, Haskell  
- **Formal (5):** Alloy, BNF, Z-notation, TLA+, Agda  
- **Other (2):** Pickle, Maxima  
- **Reference (1):** Markdown  

### Performance (order of magnitude)

Typical full suite run is sub-second on small models; exact numbers depend on hardware and `REFERENCE_CONFIG` / logging flags.
