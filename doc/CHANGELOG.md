# GNN Documentation Changelog

**Hub**: [README.md](README.md).

> **Latest update**: April 2026  
> **Status**: Maintained

This document tracks notable changes, additions, and improvements to the GNN documentation ecosystem.

---

## Version History

### v2.0.0 (March 2026)

#### New documentation

- **doc/CHANGELOG.md** — Changelog for documentation updates
- **doc/QUICK_REFERENCE.md** — Short command and pattern reference
- **doc/arc-agi/** — ARC-AGI model documentation
- **doc/cerebrum/** — Cerebrum integration notes
- **doc/muscle-mem/** — Muscle memory model documentation

#### Expanded documentation

- **cognitive_phenomena/** — Additional topic pages (attention, consciousness, effort, etc.)
- **doc/gnn/** — Subdirectories include advanced, implementations, integration, language, mcp, modules, operations, reference, testing, tutorials

#### Cross-references

- Tighter linking between GNN spec and framework docs
- **INDEX.md** navigation updates

#### Infrastructure

- AGENTS/README coverage expanded across subtrees (see [development/docs_audit.py](development/docs_audit.py))

---

### v1.3.0 (February 2026)

- **CROSS_REFERENCE_INDEX.md** — Topic navigation
- **learning_paths.md** — Multiple learning tracks
- **style_guide.md** — Documentation conventions

---

## Documentation metrics

Counts drift as the tree grows. For **mechanical** link and pairing checks, run from the repository root:

```bash
uv run python doc/development/docs_audit.py --strict
```

Approximate scale (order of magnitude): hundreds of Markdown files under `doc/`, many AGENTS/README pairs, cross-links maintained via audits rather than fixed integers here.

---

## Contributing to documentation

1. Add or update **README.md** in the relevant directory
2. Add **AGENTS.md** where the subtree uses agent-oriented signposts
3. Update **doc/INDEX.md** when adding a major entry point
4. Update **doc/CROSS_REFERENCE_INDEX.md** for topic graph changes when appropriate
5. Add a short entry to this changelog for user-visible doc changes

---

*Last updated: April 2026*
