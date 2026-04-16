# Round-Trip Reports — Specification

## Report Format

Each report contains:
- **Timestamp**: ISO 8601 execution time
- **Formats tested**: List of parser/serializer pairs exercised
- **Field comparison**: Per-field match/mismatch counts
- **Pass/fail summary**: Overall fidelity verdict

## JSON Schema

Results files use `{format, passed, total, mismatches: [{field, expected, actual}]}` structure.
