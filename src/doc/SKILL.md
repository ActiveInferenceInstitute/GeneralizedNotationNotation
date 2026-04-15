# Core Skill: `doc_reference`

**Function**: Internal source-adjacent documentation and reference notes supporting `src/` implementation and maintenance.

## Scope

This module is a **static documentation directory** — it does not export runtime functionality. It holds hand-maintained markdown content that supports development, including:

- Implementation notes and quick-reference guides
- Changelog tracking for the documentation subtree
- Cognitive phenomena documentation for Active Inference modeling

## Contents

| File | Purpose |
|------|---------|
| `CHANGELOG.md` | Documentation subtree change history |
| `QUICK_REFERENCE.md` | Developer quick-reference for pipeline operations |
| `cognitive_phenomena/` | Cognitive modeling reference material |

## Policy

- Does not define runtime pipeline behavior
- Complements top-level `doc/` content for implementation-near material
- Generated outputs, caches, and transient artifacts are excluded from coverage expectations
