# GNN v1.1 Syntax Specification

> **Status**: Living document · Last updated 2026-03-24
> **Canonical reference for parsers, validators, and editor support.**

---

## 1  File Structure

A GNN file is a UTF-8 Markdown file (`.md`) consisting of **ordered sections**, each
introduced by a level-2 header (`## SectionName`).

### Required sections (in order)

| Section | Purpose |
|---------|---------|
| `## GNNSection` | Short identifier (no spaces; e.g. `ActInfPOMDP`) |
| `## GNNVersionAndFlags` | `GNN v1` or `GNN v1.1` with optional flags |
| `## ModelName` | Human-readable model title |
| `## StateSpaceBlock` | Variable and matrix declarations |
| `## Connections` | Edge list between state-space variables |

### Parser and validator behavior

Two code paths are used in this repository:

- **Strict schema validator**: `src/gnn/schema.py` enforces required sections, declaration formats, and connection grammar.
- **Permissive markdown parser**: `src/gnn/parsers/markdown_parser.py` is more tolerant when loading mixed or partially structured markdown.

For CI, type-checking, and pipeline validation, prefer examples that satisfy the strict schema validator.

### Optional sections

| Section | Purpose |
|---------|---------|
| `## ModelAnnotation` | Free-text description of the model |
| `## InitialParameterization` | Concrete matrix / vector values |
| `## Metadata` | Key-value metadata (author, date, tags) |

---

## 2  Variable Declarations (`StateSpaceBlock`)

Each non-comment, non-blank line declares one variable:

```
NAME[dim₁, dim₂, …, key=value, …]   # optional comment
```

### Name rules

- Alphanumeric plus `_`, `π`, `'` (prime).
- Case-sensitive: `s` ≠ `S`.

### Dimension rules

- Comma-separated positive integers: `A[3,3]`.
- A trailing key-value pair `type=<type>` is **required**: `float`, `int`, `bool`.
- Dimensions may use named references: `A[num_obs, num_states]`.

### v1.1 Extensions — Default Values

Variables may carry a default-value hint after dimensions:

```
D[3, type=float, default=uniform]
W[4,4, type=float, default=zeros]
I[3,3, type=float, default=eye]
B[3,3,3, type=float, default=ones]
```

Supported defaults: `uniform`, `zeros`, `ones`, `eye`, `random`.

---

## 3  Connections

Each line in the `## Connections` section defines one directed or undirected edge:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `A>B` | Causal / directed: A → B | `D>s` |
| `A-B` | Undirected / bidirectional | `s-A` |
| `A>B:label` | Annotated directed edge | `π>u:select_action` |
| `A-B:label` | Annotated undirected edge | `s-A:likelihood` |

### v1.1 Extension — Connection Annotations

Annotations appear after a colon following the edge:

```
D>s:prior_initialization
A-o:observation_mapping
G>π:policy_selection
```

Annotations are arbitrary strings (alphanumeric + `_`). They serve as labels
for rendering and documentation; parsers **must** accept and preserve them,
but **may** ignore them for structural validation.

---

## 4  Initial Parameterization

Matrix values use brace-delimited, comma-separated notation:

```
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9,  0.05),
  (0.05, 0.05, 0.9)
}
```

### Rules

1. Outer braces `{…}` wrap the full tensor.
2. Inner parentheses `(…)` group rows or slices.
3. Values are numeric literals (ints or floats).
4. Matrix dimensions **must** match the variable declaration in `StateSpaceBlock`
   (validator emits `GNNParseError` on mismatch).

---

## 5  Comments

- Lines starting with `#` (after optional whitespace) are comments.
- Inline comments: any `#` after a declaration.

---

## 6  Multi-Model Files (v1.1)

A single `.md` file may contain multiple models separated by a `---` (horizontal rule)
on its own line. Each model block must contain its own `## GNNSection` and
`## StateSpaceBlock`.

---

## 7  Error Taxonomy

| Error Code | Meaning |
|------------|---------|
| `GNN-E001` | Missing required section |
| `GNN-E002` | Variable dimension mismatch (declaration vs parameterization) |
| `GNN-E003` | Unknown variable in connection |
| `GNN-E004` | Duplicate variable declaration |
| `GNN-E005` | Unparseable connection syntax |

| Warning Code | Meaning |
|--------------|---------|
| `GNN-W001` | Variable declared but never used in connections |
| `GNN-W002` | Connection references undeclared variable |
| `GNN-W003` | Parameterization provided for undeclared variable |
