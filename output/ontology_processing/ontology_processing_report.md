# 🧬 GNN Ontological Annotations Report

## 📊 Summary of Ontology Processing

- **Files Processed:** 2 / 2
- **Total Ontological Annotations Found:** 32
- **Total Annotations Validated:** 32
  - ✅ Passed: 20
  - ❌ Failed: 12

---

��️ Report Generated: 2025-06-06 12:20:30
🎯 GNN Source Directory: `src/gnn/examples`
📖 Ontology Terms Definition: `src/ontology/act_inf_ontology_terms.json` (Loaded: 48 terms)

---

### Ontological Annotations for `src/gnn/examples/pymdp_pomdp_agent.md`
#### Mappings:
- `A_m0` -> `LikelihoodMatrixModality0`
- `A_m1` -> `LikelihoodMatrixModality1`
- `A_m2` -> `LikelihoodMatrixModality2`
- `B_f0` -> `TransitionMatrixFactor0`
- `B_f1` -> `TransitionMatrixFactor1`
- `C_m0` -> `LogPreferenceVectorModality0`
- `C_m1` -> `LogPreferenceVectorModality1`
- `C_m2` -> `LogPreferenceVectorModality2`
- `D_f0` -> `PriorOverHiddenStatesFactor0`
- `D_f1` -> `PriorOverHiddenStatesFactor1`
- `s_f0` -> `HiddenStateFactor0`
- `s_f1` -> `HiddenStateFactor1`
- `s_prime_f0` -> `NextHiddenStateFactor0`
- `s_prime_f1` -> `NextHiddenStateFactor1`
- `o_m0` -> `ObservationModality0`
- `o_m1` -> `ObservationModality1`
- `o_m2` -> `ObservationModality2`
- `π_f1` -> `PolicyVectorFactor1`
- `u_f1` -> `ActionFactor1`
- `G` -> `ExpectedFreeEnergy`

**Validation Summary**: All ontological terms are recognized.

---

### Ontological Annotations for `src/gnn/examples/rxinfer_multiagent_gnn.md`
#### Mappings:
- `dt` -> `TimeStep` (**INVALID TERM**)
- `gamma` -> `ConstraintParameter` (**INVALID TERM**)
- `nr_steps` -> `TrajectoryLength` (**INVALID TERM**)
- `nr_iterations` -> `InferenceIterations` (**INVALID TERM**)
- `nr_agents` -> `NumberOfAgents` (**INVALID TERM**)
- `softmin_temperature` -> `SoftminTemperature` (**INVALID TERM**)
- `A` -> `StateTransitionMatrix` (**INVALID TERM**)
- `B` -> `ControlInputMatrix` (**INVALID TERM**)
- `C` -> `ObservationMatrix` (**INVALID TERM**)
- `initial_state_variance` -> `InitialStateVariance` (**INVALID TERM**)
- `control_variance` -> `ControlVariance` (**INVALID TERM**)
- `goal_constraint_variance` -> `GoalConstraintVariance` (**INVALID TERM**)

**Validation Summary**: 12 unrecognized ontological term(s) found.

---
