# Parsers Tests

Pytest coverage for `src/gnn/parsers/`.

This folder contains module-focused tests for the GNN parsing surface: markdown
frontmatter helpers, the markdown parser, the parse-tree outline, and the
format registries.

Run:

```bash
uv run --extra dev python -m pytest tests/parsers/ -q
```

`test_parsers_public_api.py` covers `has_frontmatter`, `parse_frontmatter`,
`MarkdownGNNParser.parse_string` and `get_parse_tree_visualization`.

`test_parsers_registry.py` covers `PARSER_REGISTRY`, `SERIALIZER_REGISTRY` and
`GNNParsingSystem` format listing plus markdown string parsing.
