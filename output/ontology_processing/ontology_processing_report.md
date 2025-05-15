# 🧬 GNN Ontological Annotations Report

## 📊 Summary of Ontology Processing

- **Files Processed:** 3 / 3
- **Total Ontological Annotations Found:** 54
- **Total Annotations Validated:** 54
  - ✅ Passed: 38
  - ❌ Failed: 16

---

��️ Report Generated: 2025-05-15 13:54:17
🎯 GNN Source Directory: `src/gnn/examples`
📖 Ontology Terms Definition: `src/ontology/act_inf_ontology_terms.json` (Loaded: 48 terms)

---

### Ontological Annotations for `src/gnn/examples/gnn_POMDP_example.md`
#### Mappings:
- `s_f0` -> `HiddenStateFactor0`
- `s_f1` -> `HiddenStateFactor1`
- `o_m0` -> `ObservationModality0`
- `o_m1` -> `ObservationModality1`
- `pi_c0` -> `PolicyVector`
- `pi_c1` -> `PolicyVectorFactor1`
- `u_c0` -> `Action`
- `u_c1` -> `ActionFactor1`
- `A_m0` -> `LikelihoodMatrixModality0`
- `A_m1` -> `LikelihoodMatrixModality1`
- `B_f0` -> `TransitionMatrixFactor0`
- `B_f1` -> `TransitionMatrixFactor1`
- `C_m0` -> `LogPreferenceVectorModality0`
- `C_m1` -> `LogPreferenceVectorModality1`
- `D_f0` -> `PriorOverHiddenStatesFactor0`
- `D_f1` -> `PriorOverHiddenStatesFactor1`
- `G` -> `ExpectedFreeEnergy`
- `t` -> `Time`

**Validation Summary**: All ontological terms are recognized.

---

### Ontological Annotations for `src/gnn/examples/gnn_airplane_trading_pomdp.md`
#### Mappings:
- `s_f0` -> `NivelDeCombustible` (**INVALID TERM**)
- `s_f1` -> `DesempenoDelMercado` (**INVALID TERM**)
- `o_m0` -> `MedidorDeCombustible` (**INVALID TERM**)
- `o_m1` -> `ResultadoDeTransaccion` (**INVALID TERM**)
- `pi_c0` -> `VectorPoliticaControl0` (**INVALID TERM**)
- `u_c0` -> `AccionControl0` (**INVALID TERM**)
- `A_m0` -> `MatrizVerosimilitudModalidad0` (**INVALID TERM**)
- `A_m1` -> `MatrizVerosimilitudModalidad1` (**INVALID TERM**)
- `B_f0` -> `MatrizTransicionFactor0` (**INVALID TERM**)
- `B_f1` -> `MatrizTransicionFactor1` (**INVALID TERM**)
- `C_m0` -> `VectorPreferenciaModalidad0` (**INVALID TERM**)
- `C_m1` -> `VectorPreferenciaModalidad1` (**INVALID TERM**)
- `D_f0` -> `DistribucionPreviaFactorOculto0` (**INVALID TERM**)
- `D_f1` -> `DistribucionPreviaFactorOculto1` (**INVALID TERM**)
- `G` -> `EnergiaLibreEsperada` (**INVALID TERM**)
- `t` -> `PasoDeTiempo` (**INVALID TERM**)

**Validation Summary**: 16 unrecognized ontological term(s) found.

---

### Ontological Annotations for `src/gnn/examples/gnn_example_pymdp_agent.md`
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
