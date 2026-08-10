# GNN v1.1 Syntax Specification

> **Status**: Living document · Last updated 2026-04-14
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
| `## Equations` | LaTeX-rendered formulas defining model dynamics and relationships between variables |
| `## Time` | Static/Dynamic setting, discrete/continuous time variable, and model time horizon |
| `## ActInfOntologyAnnotation` | Variable semantic bindings to Ontological domains |
| `## ModelParameters` | Key-value numeric parameters for the model (e.g. counts, horizons, simulation settings) |
| `## Footer` | Closes the file and allows read-in from either end |
| `## Signature` | Cryptographic signature block / provenance information |

### `ModelParameters` keys read by renderers

Entries are `key: value`, one per line, scalar values only. Dimension keys
declared here may be referenced by name from `StateSpaceBlock`
(e.g. `A[num_obs, num_hidden_states]`). Beyond the dimension counts
(`num_hidden_states`, `num_obs`, `num_actions`, `num_timesteps`,
`num_modalities`, `learning_rate`), two keys steer inference itself:

| Key | Values | Effect |
|-----|--------|--------|
| `num_factors` | int | `> 1` selects the FACTORED model kind (see §4, *Parameterization families*) |
| `nr_agents` | int | `> 1` selects the MULTI_AGENT model kind (see §4, *Parameterization families*) |
| `inference_mode` | `batch` (default) or `online` | RxInfer FLAT models: `batch` runs smoothing over the whole observation sequence; `online` runs per-timestep `infer()` on the observation prefix, so the filtered posterior drives expected-free-energy and habit-prior action selection. Any other value is rejected. |
| `inference_iterations` | int (default `20`) | Variational iterations per `infer()` call in generated RxInfer scripts. |

`inference_mode` may also be supplied as a render option; the value declared
in `ModelParameters` is what makes a file self-describing. See
[`src/render/rxinfer/README.md`](../../src/render/rxinfer/README.md) for the
per-kind strategy table.

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

### Parameterization families

Beyond the discrete POMDP matrices `A`, `B`, `C`, `D` and the optional habit
prior `E`, three further key families are recognized. A file may carry more
than one family at once.

| Family | Keys | Meaning |
|--------|------|---------|
| Discrete POMDP | `A`, `B`, `C`, `D`, `E` | Likelihood, transition, preferences, initial prior, habit prior. `B` is ordered `(next_state, previous_state, action)`. |
| Linear-Gaussian | `F`, `H`, `Q`, `R`, `prior_mean`, `prior_cov` | State-space system matrices: state transition, observation readout, process noise, observation noise, and the Gaussian prior over the initial latent. |
| Dirichlet priors | `dirichlet_A` … `dirichlet_E` | Pseudo-counts for a matrix treated as a latent variable rather than a fixed constant. |
| Per-level / per-agent | `A_level1`, `B_level2`, …; `A_agent1`, `B_agent2`, … | Suffixed copies of the POMDP matrices, one set per hierarchy level or per agent. |

#### Dual parameterization

A file may declare **both** the discrete matrices and the linear-Gaussian
matrices. Each backend reads the family it can execute: the discrete
matrices drive the categorical backends (PyMDP and friends), while the
`F`/`H`/`Q`/`R` block drives native linear-Gaussian rendering. The three
exemplars under `input/gnn_files/continuous/` are written this way. Keep the
two descriptions of the same system consistent — nothing cross-validates them.

Dual-parameterized files provoke validator findings that are expected and
non-fatal, because the linear-Gaussian keys are deliberately not mirrored
into `StateSpaceBlock`. Running the matrix-dimension check over
`continuous/continuous_navigation.md` reports:

```text
GNN-E002 | Matrix 'F': declared shape 1 but parameterization has shape 2x2
GNN-W003 | Parameterization for undeclared variable 'H'
GNN-W003 | Parameterization for undeclared variable 'Q'
GNN-W003 | Parameterization for undeclared variable 'R'
GNN-W003 | Parameterization for undeclared variable 'prior_mean'
GNN-W003 | Parameterization for undeclared variable 'prior_cov'
```

The `GNN-E002` comes from a name collision: `F` means two unrelated things by
convention. In `StateSpaceBlock`, `F[1,type=float]` is the scalar Variational
Free Energy readout; in `InitialParameterization`, `F={…}` is the
linear-Gaussian state-transition matrix. Only the
`InitialParameterization` occurrence participates in model-kind detection.
Do not "fix" these findings by deleting the linear-Gaussian block — that
would silently demote the model from CONTINUOUS to FLAT.

#### Dirichlet pseudo-counts

Declare the pseudo-count array in `StateSpaceBlock` alongside the matrix it
governs, then give its values in `InitialParameterization`:

```gnn
## StateSpaceBlock
dirichlet_A[3,3,type=float]   # prior counts for q(A)

## InitialParameterization
# rows = observations, columns = states; each column is one Dirichlet
dirichlet_A={
  (3.0, 1.0, 1.0),
  (1.0, 3.0, 1.0),
  (1.0, 1.0, 3.0)
}
```

The renderer then treats `A` as a latent `DirichletCollection` learned
jointly with the hidden states, rather than as a fixed matrix. The example
above is identity-biased: mass 3 on the diagonal against 1 elsewhere, so the
prior leans toward a faithful likelihood without asserting it. Taken from
[`input/gnn_files/learning/dirichlet_likelihood_learning.md`](../../input/gnn_files/learning/dirichlet_likelihood_learning.md).

#### Model-kind detection

Renderers dispatch on a **model kind** derived structurally from the spec —
from the `## GNNSection` value, the parameterization keys above, and explicit
`nr_agents` / `num_factors` counts. Prose is never scanned, so text in a
`ModelName` or annotation cannot change how a model renders.

| Kind | Selected by |
|------|-------------|
| `MULTI_AGENT` | `nr_agents > 1`, or any `[ABCDE]_agent<N>` key |
| `HIERARCHICAL` | `hierarchical` in `GNNSection`, or any `[ABCDE]_level<N>` key |
| `CONTINUOUS` | `continuous` in `GNNSection`, or all of `F`/`H`/`Q`/`R`, or both `prior_mean` and `prior_cov` |
| `LEARNING` | `learning` in `GNNSection`, or any `dirichlet_[ABCDE]` key |
| `FACTORED` | `num_factors > 1` |
| `FLAT` | none of the above (the common case) |

Precedence runs top to bottom as listed: hierarchical and multi-agent files
also have multiple factors, so the more specific kinds are tested first.
`InitialParameterization` must parse to a mapping; a non-mapping value raises
`ValueError` rather than falling back to a default kind. The authority is
`detect_model_kind` in
[`src/render/pomdp_contract.py`](../../src/render/pomdp_contract.py).

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

## 7 ActInfOntologyAnnotation (v1.5)

To bind internal variables to external semantic meaning, construct a list mapping variables to CamelCase ontological states. This is requisite for the Neurosymbolic LLM Context Analysis features in `13_llm.py` and heuristics tracking in `24_intelligent_analysis.py`.

```gnn
## ActInfOntologyAnnotation
s=HiddenState
o=Observation
A=LikelihoodMatrix
B=TransitionMatrix
```

Terms are validated against
[`src/ontology/act_inf_ontology_terms.json`](../../src/ontology/act_inf_ontology_terms.json)
by Step 10. Note the conventional binding: `A` is the **likelihood**
(observation) matrix and `B` is the **transition** matrix — not the reverse.

---

## 8  Error Taxonomy

| Error Code | Meaning |
|------------|---------|
| `GNN-E001` | Missing required section |
| `GNN-E002` | Variable dimension mismatch (declaration vs parameterization) |
| `GNN-E003` | Unknown variable in connection (**reserved, not yet enforced** — `src/gnn/schema.py` reports this condition as the `GNN-W002` warning instead) |
| `GNN-E004` | Duplicate variable declaration |
| `GNN-E005` | Unparseable connection syntax |

| Warning Code | Meaning |
|--------------|---------|
| `GNN-W001` | Variable declared but never used in connections (**planned, not yet enforced** — no emitting code in `src/gnn/schema.py`) |
| `GNN-W002` | Connection references undeclared variable |
| `GNN-W003` | Parameterization provided for undeclared variable |
