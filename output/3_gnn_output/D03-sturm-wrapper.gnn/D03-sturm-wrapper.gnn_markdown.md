## GNNVersionAndFlags
Version: 1.0

## ModelName
STURM Building-Stock Wrapper Blanket

## ModelAnnotation
Markov blanket for STURM as a coupled component in NEST Stage 1.

Boundary scope: this blanket is drawn around MODEL ↔ NEST PLATFORM. STURM
receives energy prices by fuel type (downscaled from the national model) and
returns final energy demand by fuel; renovation and stock-turnover machinery
(archetype shares, vintage survival, fuel switching) stays internal and is
never exchanged. This is the D02 mirror with sensory/active reversed on the
price/demand pair — D02's sensory d ↔ D03's active d, D02's active p ↔ D03's
sensory p. There is NO counterpart to D02's cap: D02's cap is the HIERARCHY
regime imposed constraint of PR #87 Q-13; STURM has no imposed constraint
input, so this blanket has no such sensory variable.

STURM assumption (labeled): STURM = MESSAGEix-Buildings (IIASA), IIASA
confirmation pending; residential building stock POC only (commercial
excluded). Catalog boundary record: catalog/models/sturm.yaml (prices in
USD/GJ by fuel; demand out PJ/yr; 5-year steps; ~60 regions).

Scope vs Q1 (Technical_Responses_Draft_v1): this file specifies the wrapper's
model-level I/O blanket — Q1 functions #1–#3. Q1 function #4 (the Active
Inference latent-summary layer, via RxInfer) is deliberately NOT modeled here;
those latent states are the factor-graph deliverables (D04, D07), which will
sit behind this blanket's internal side.

- Sensory states: what crosses INTO the model — energy prices by fuel type (downscaled from the national model).
- Active states: what crosses OUT — final energy demand by fuel (MESSAGEix-compatible). Renovation/stock-turnover rates are produced inside STURM and consumed by its own demand calculation; they are internal here, not exchanged.
- Internal states: archetype stock shares, renovation/turnover flows, and the turnover state machinery — never exchanged.
- Every exchanged variable carries the declared boundary fields (PR #80 §4 / PR #95): unit · convention · temporal support · spatial support. Prices carry the annualisation/downscaling convention of the national→STURM handoff; demand is MESSAGEix-compatible annual demand (STURM's native 5-year steps are internal; the 12 slices below are the coupling-iteration resolution, illustrative pending IIASA packaging). Spatial support follows the D05 memo (res-5 exchange via declared crosswalk).
- This mirrors the D02 pattern file (MESSAGEix wrapper); D03 is its price/demand mirror.
- Renderable form: structural blanket, no renderable form as authored — the step-11 renderer classifies it as a discrete POMDP and fails on the missing `A/B/C/D` (upstream: GeneralizedNotationNotation#111). The renderable instantiation of this blanket is the continuous-exemplar fixture block (`models/active_inference/rxinfer-bridge/fixtures/sturm/sturm_fixture.gnn.md`); until the upstream path exists, render the instantiation, not the wrapper.

## StateSpaceBlock
p[2,12],float
d[2,12],float
q[4],float
r[4,12],float
u[8,1],float
t[1],integer

## Connections
p>u
u>q
u>r
q>d
r>d
d>p

## InitialParameterization
p = [[0.0, 0.0]]
num_fuel_types = 2
num_time_slices = 12
num_archetypes = 4
num_regions = 1

## Time
Dynamic
DiscreteTime
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
p = Observation
d = Action
q = HiddenState
r = HiddenState
u = HiddenState
t = Time

## Footer
Generated: 2026-09-13T14:42:54.219366

## Signature
