## GNNVersionAndFlags
Version: 1.0

## ModelName
MESSAGEix National Wrapper Blanket

## ModelAnnotation
Markov blanket for MESSAGEix as a coupled component in NEST Stage 1.

Boundary scope: this blanket is drawn around MODEL ↔ NEST PLATFORM. Of the
active states, only p crosses onward to the coupled model (STURM); e and J
are active toward the platform (gates, monitoring, convergence accounting)
and never reach STURM. cap arrives as an imposed constraint — the HIERARCHY
regime of PR #87 Q-13; the D03 (STURM) blanket has no counterpart input.

Scope vs Q1 (Technical_Responses_Draft_v1): this file specifies the wrapper's
model-level I/O blanket — Q1 functions #1–#3. Q1 function #4 (the Active
Inference latent-summary layer: distilling raw outputs into belief variables
such as decarbonization rate, via RxInfer) is deliberately NOT modeled here;
those latent states are the factor-graph deliverables (D04, D07), which will
sit behind this blanket's internal side.

- Sensory states: what crosses INTO the model — demand trajectories per commodity (from the building-stock side) and policy constraints (emissions cap).
- Active states: what crosses OUT — commodity shadow prices (PRICE_COMMODITY duals), total emissions, and objective value (see boundary scope above).
- Internal states: capacities, dispatch, and the LP solution machinery — never exchanged.
- Every exchanged variable carries the declared boundary fields (PR #80 §4 / PR #95): unit · convention · temporal support · spatial support. Prices carry the engine's annualisation convention; demand carries its allocation basis. Spatial support follows the D05 memo (res-5 exchange via declared crosswalk).
- This is the D02 pattern file; D03 (STURM wrapper) mirrors it with sensory/active reversed on the price/demand pair.
- Renderable form: this wrapper is a *structural* blanket (it declares boundary structure, not a generative parameterization), so the GNN step-11 renderer has no renderable form of it as authored — it classifies the spec as a discrete POMDP and fails on the missing `A/B/C/D` (upstream: GeneralizedNotationNotation#111). The renderable instantiation of this blanket is the continuous-exemplar fixture block (`models/active_inference/rxinfer-bridge/fixtures/d04/d04_fixture.gnn.md`), which renders to RxInfer and executes; until the upstream path exists, render the instantiation, not the wrapper.

## StateSpaceBlock
d[2,12],float
cap[1],float
p[2,12],float
e[1],float
J[1],float
K[4],float
x[4,12],float
s[8,1],float
t[1],integer

## Connections
d>s
cap>s
s>K
s>x
K>p
x>p
x>e
s>J
p>d

## InitialParameterization
d = [[0.0, 0.0]]
cap = [[200000.0]]
num_commodities = 2
num_time_slices = 12
num_technologies = 4
num_regions = 1

## Time
Dynamic
DiscreteTime
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
d = Observation
cap = Observation
p = Action
e = Action
J = Action
K = HiddenState
x = HiddenState
s = HiddenState
t = Time

## Footer
Generated: 2026-09-13T14:42:55.479610

## Signature
