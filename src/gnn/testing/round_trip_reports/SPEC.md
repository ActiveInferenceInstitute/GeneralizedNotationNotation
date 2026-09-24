# Round-Trip Reports — Specification

## Report Format

Each report contains:
- **Timestamp**: ISO 8601 execution time
- **Formats tested**: List of parser/serializer pairs exercised
- **Field comparison**: Per-field match/mismatch counts
- **Pass/fail summary**: Overall fidelity verdict

## JSON Schema

Results files use `{total_tests, successful_tests, failed_tests, success_rate}`
(shape of the `ComprehensiveTestReport` export in `round_trip_results.py`).
