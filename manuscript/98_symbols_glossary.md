# Symbols and Glossary {#sec:symbols_glossary}

This glossary defines the Generalized Notation Notation (GNN) language constructs
used to specify Active Inference generative models, together with the canonical
Active Inference symbols those constructs carry. Definitions follow the GNN
syntax specification and the discrete POMDP exemplars distributed with the
repository, and they align with the standard discrete-state-space formulation of
Active Inference [@gnn2023; @dacosta2020; @parr2022].

## Language Constructs

A GNN file is an ordered, UTF-8 Markdown document whose level-2 headers name
required and optional sections. The strict schema validator enforces section
order, declaration grammar, and connection syntax; the constructs below are the
load-bearing sections a parser must recognize.

| Construct | Meaning |
| --- | --- |
| `## GNNSection` | Required short identifier for the model block (no spaces, e.g. `ActInfPOMDP`). |
| `## GNNVersionAndFlags` | Required version declaration (`GNN v1` or `GNN v1.1`) with optional flags. |
| `## ModelName` | Required human-readable title of the generative model. |
| `## StateSpaceBlock` | Required block declaring every variable and matrix as `NAME[dim, …, type=…]`, one per line. |
| `## Connections` | Required edge list relating state-space variables, expressing the model's factor graph. |
| `## ModelAnnotation` | Optional free-text description of the model's modalities, factors, and assumptions. |
| `## InitialParameterization` | Optional concrete numeric values for declared matrices and vectors. |
| `## ActInfOntologyAnnotation` | Optional bindings from each variable to a CamelCase Active Inference ontology term. |
| `## ModelParameters` | Optional key-value dimensions (e.g. `num_hidden_states`, `num_obs`, `num_actions`) consumed by code generators. |
| `## Time` | Optional dynamics declaration: a time variable plus `Dynamic`/`Static`, `Discrete`/`Continuous`, and `ModelTimeHorizon`. |
| `NAME[d₁,d₂,…,type=…]` | A variable or tensor declaration; dimensions are positive integers or named references, and a `type` (`float`, `int`, `bool`) is required. |
| `A>B` | Directed (causal) connection operator: edge from `A` to `B`, e.g. `D>s` (a prior conditions a hidden state). |
| `A-B` | Undirected (bidirectional) connection operator, e.g. `s-A` (a hidden state participates in the likelihood mapping). |
| `A>B:label` / `A-B:label` | A v1.1 annotated edge; the trailing label documents the relation and is preserved but may be ignored for structural validation. |
| `default=…` | A v1.1 declaration hint (`uniform`, `zeros`, `ones`, `eye`, `random`) supplying an initialization for a matrix or vector. |

## Active Inference Symbols

The exemplar discrete POMDP agent declares the standard generative-model
components of Active Inference over a discrete state space, mapping each GNN
variable to its probabilistic meaning [@dacosta2020; @smith2022; @parr2022].

| Symbol | Meaning |
| --- | --- |
| `A` | Likelihood (observation) matrix encoding $P(o \mid s)$, mapping hidden states to observation outcomes. |
| `B` | Transition matrix encoding $P(s' \mid s, u)$, mapping a previous state and action to the next state. |
| `C` | Preference vector: log-preferences over observation outcomes that bias the agent toward preferred outcomes. |
| `D` | Prior vector over initial hidden states, $P(s_0)$. |
| `E` | Habit vector: an initial policy prior (baseline preference) over actions. |
| `s` | Current hidden-state distribution; `s_prime` (`s'`) is the next hidden-state distribution. |
| `o` | Current observation, an integer index over outcome modalities. |
| `π` | Policy: a distribution over actions inferred from expected free energy. |
| `u` | The selected (sampled) action. |
| `F` | Variational free energy, minimized during state inference to update beliefs from observations [@friston2010]. |
| `G` | Expected free energy per policy, minimized during policy inference to score candidate actions [@dacosta2020]. |
| `t` | Discrete time step. |

## Ontology Bindings and Implementations

The `## ActInfOntologyAnnotation` section binds each variable to a canonical term
(`A=LikelihoodMatrix`, `B=TransitionMatrix`, `C=LogPreferenceVector`,
`D=PriorOverHiddenStates`, `s=HiddenState`, `o=Observation`, `π=PolicyVector`),
which downstream pipeline steps use for semantic analysis and validation. The
same GNN specification feeds the project's rendering backends — including the
{{GNN_BACKEND_COUNT}} executable targets ({{GNN_BACKEND_LIST}}) — so that a model
written once in this notation can be parsed, visualized, and executed across the
{{GNN_STEP_COUNT}}-step pipeline ({{GNN_STEP_RANGE}}) without restating its
mathematics [@gnn2023; @heins2022; @defelice2021].
