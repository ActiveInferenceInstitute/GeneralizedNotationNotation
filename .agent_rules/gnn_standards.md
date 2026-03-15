# GNN Standards

> GNN (Generalized Notation Notation) files are Markdown documents encoding Active Inference generative models.

## File Format

GNN files use Markdown with specific section headers:

```markdown
## GNNVersionAndFlags
GNN_v1.5; UTF-8; OPTIONAL_FIELD

## ModelName
MyActiveInferenceModel

## ModelAnnotation
Description of what this model does.

## StateSpaceBlock
# Variable definitions
s_f0[2,1,type=hidden,distribution=categorical] # Hidden state factor 0
o_m0[3,1,type=observed,distribution=categorical] # Observation modality 0
u_c0[2,1,type=control,distribution=categorical] # Control factor 0

## Connections
s_f0 > o_m0  # Observation mapping (A matrix)
s_f0 > s_f0  # Transition dynamics (B matrix)

## InitialParameterization
A = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]] # Likelihood matrix
B = [[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]] # Transition
C = [0.0, 1.0, 0.0] # Preferences
D = [0.5, 0.5] # Initial state prior

## Footer
Created: 2026-01-01
```

---

## Variable Naming Conventions

| Prefix | Type | Example |
|--------|------|---------|
| `s_fN` | Hidden state factor N | `s_f0`, `s_f1` |
| `o_mN` | Observation modality N | `o_m0`, `o_m1` |
| `u_cN` | Control/action factor N | `u_c0` |
| `pi` | Policy | `pi[5,1]` |
| `G` | Expected free energy | `G[5,1]` |
| `F` | Variational free energy | `F[1,1]` |

---

## Dimension Notation

```
variable[dims, type, distribution]

dims: comma-separated integers, e.g. [3,1] = 3 states × 1
type: hidden, observed, control, policy, free_energy
distribution: categorical, dirichlet, gaussian, beta
```

---

## Supported Formats (21+)

| Format | Extension | Use Case |
|--------|-----------|----------|
| Markdown | `.md` | Primary, human-readable |
| JSON | `.json` | Structured data exchange |
| YAML | `.yaml` | Config-friendly |
| XML | `.xml` | Legacy systems |
| Pickle | `.pkl` | Python serialization |
| Protobuf | `.pb` | High-performance binary |
| GraphML | `.graphml` | Graph analysis tools |
| GEXF | `.gexf` | Gephi visualization |
| Maxima | `.mac` | Symbolic math |
| Julia | `.jl` | Julia ecosystem |
| ... | ... | Additional formats |

---

## Validation Levels

| Level | Description |
|-------|-------------|
| `BASIC` | File exists, header present, parseable |
| `STANDARD` | All required sections, valid variable names |
| `STRICT` | Dimension consistency, type constraints, POMDP conformance |
| `RESEARCH` | Mathematical constraints, Active Inference spec compliance |
| `ROUND_TRIP` | Serialize → deserialize → compare, semantic preservation |

---

## Matrix Conventions (POMDP)

| Matrix | Symbol | Shape | Meaning |
|--------|--------|-------|---------|
| Likelihood | A | `[n_obs × n_states]` | P(o\|s) |
| Transition | B | `[n_states × n_states × n_actions]` | P(s'\|s,a) |
| Preference | C | `[n_obs]` | log P(o) (desired observations) |
| Prior | D | `[n_states]` | Initial state distribution |
| Policy prior | E | `[n_policies]` | Prior over policies |

---

## Discovery and Parsing

```python
from gnn import discover_gnn_files, parse_gnn_file

# Discover files
files = discover_gnn_files(Path("input/gnn_files"), recursive=True)

# Parse a file
model = parse_gnn_file(files[0])

# Key fields in parsed model:
model["name"]                    # Model name
model["state_space"]             # Variable definitions
model["connections"]             # Graph edges
model["initialparameterization"] # A/B/C/D matrices
model["annotations"]             # Human-readable notes
```

---

## Cross-Format Validation

After round-trip conversion, semantic equivalence is verified:
1. Parse original `.md`
2. Export to JSON
3. Re-parse from JSON
4. Compare state space, connections, matrices

---

**Last Updated**: March 2026 | **Status**: Production Standard
