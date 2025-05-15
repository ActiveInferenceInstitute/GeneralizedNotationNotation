# GNN Example: Airplane Trading POMDP
# Format: Markdown representation of a POMDP model for an airplane trading scenario.
# Version: 1.0
# This file is machine-readable and describes an agent making trading decisions on an airplane,
# where financial losses can impact fuel levels.

## GNNSection
AirplaneTradingPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Airplane Trading POMDP v1.0

## ModelAnnotation
This model represents a POMDP agent managing an investment portfolio aboard an airplane.
The key challenge is that financial losses from trading directly impact the plane's fuel reserves.
- Hidden State Factors:
    1.  `NivelCombustible` (Fuel Level): {Crítico, Bajo, Medio, Alto} - 4 states
    2.  `RendimientoMercado` (Market Performance): {Pérdidas, Neutral, Rentable} - 3 states
- Observation Modalities:
    1.  `IndicadorCombustible` (Fuel Gauge): {CasiVacío, Mitad, Lleno} - 3 outcomes
    2.  `ResultadoOperacion` (Trading Outcome): {Pérdida, SinCambio, Ganancia} - 3 outcomes
- Control Factor (Action):
    1.  `EstrategiaInversion` (Investment Strategy): {Desinvertir, Mantener, Invertir} - 3 actions

## StateSpaceBlock
# Hidden States (s_factorIndex[num_states_factor, 1, type=dataType])
s_f0[4,1,type=int]   # Hidden State Factor 0: NivelCombustible (0:Crítico, 1:Bajo, 2:Medio, 3:Alto)
s_f1[3,1,type=int]   # Hidden State Factor 1: RendimientoMercado (0:Pérdidas, 1:Neutral, 2:Rentable)

# Observations (o_modalityIndex[num_outcomes_modality, 1, type=dataType])
o_m0[3,1,type=int]   # Observation Modality 0: IndicadorCombustible (0:CasiVacío, 1:Mitad, 2:Lleno)
o_m1[3,1,type=int]   # Observation Modality 1: ResultadoOperacion (0:Pérdida, 1:SinCambio, 2:Ganancia)

# Control Factors / Policies (pi_controlIndex[num_actions_factor, type=dataType])
pi_c0[3,type=float]  # Policy for Control Factor 0: EstrategiaInversion (0:Desinvertir, 1:Mantener, 2:Invertir)

# Actions (u_controlIndex[1, type=dataType]) - chosen actions
u_c0[1,type=int]     # Chosen action for EstrategiaInversion

# Likelihood Mapping (A_modalityIndex[outcomes, factor0_states, factor1_states, ..., type=dataType])
A_m0[3,4,3,type=float] # IndicadorCombustible likelihood given NivelCombustible and RendimientoMercado
A_m1[3,4,3,type=float] # ResultadoOperacion likelihood

# Transition Dynamics (B_factorIndex[next_states, prev_states, control0_actions, ..., type=dataType])
# B_f0[next_fuel, prev_fuel, prev_market, action_investment]
B_f0[4,4,3,3,type=float] # NivelCombustible transitions
# B_f1[next_market, prev_market, action_investment]
B_f1[3,3,3,type=float]   # RendimientoMercado transitions

# Preferences (C_modalityIndex[outcomes, type=dataType]) - Log preferences over outcomes
C_m0[3,type=float]   # Preferences for IndicadorCombustible (prefer Lleno)
C_m1[3,type=float]   # Preferences for ResultadoOperacion (prefer Ganancia)

# Priors over Initial Hidden States (D_factorIndex[num_states_factor, type=dataType])
D_f0[4,type=float]   # Prior for NivelCombustible (e.g., start Alto)
D_f1[3,type=float]   # Prior for RendimientoMercado (e.g., start Neutral)

# Expected Free Energy
G[1,type=float]      # Overall Expected Free Energy

# Time
t[1,type=int]

## Connections
# Priors to initial states
(D_f0, D_f1) -> (s_f0, s_f1)

# States to likelihoods to observations
(s_f0, s_f1) -> (A_m0, A_m1)
(A_m0) -> (o_m0)
(A_m1) -> (o_m1)

# States and actions to transitions to next states (s_f0_next, s_f1_next are implied)
(s_f0, s_f1, u_c0) -> (B_f0, B_f1)
(B_f0) -> s_f0_next
(B_f1) -> s_f1_next

# Preferences and predicted outcomes/states to EFE
(C_m0, C_m1, A_m0, A_m1, B_f0, B_f1, s_f0, s_f1) > G # Simplified EFE dependency

# EFE to policies
G > pi_c0

# Policies to chosen actions
(pi_c0) -> u_c0

## InitialParameterization
# s_f0 (NivelCombustible): 0:Crítico, 1:Bajo, 2:Medio, 3:Alto
# s_f1 (RendimientoMercado): 0:Pérdidas, 1:Neutral, 2:Rentable
# o_m0 (IndicadorCombustible): 0:CasiVacío, 1:Mitad, 2:Lleno
# o_m1 (ResultadoOperacion): 0:Pérdida, 1:SinCambio, 2:Ganancia
# u_c0 (EstrategiaInversion): 0:Desinvertir, 1:Mantener, 2:Invertir

# A_m0[obs_fuel, state_fuel, state_market]. RendimientoMercado (s_f1) has minor/no direct effect on fuel gauge.
# Probabilities sum to 1 over obs_fuel for each (state_fuel, state_market) pair.
A_m0={ # Likelihood P(o_m0 | s_f0, s_f1)
  # o_m0 = 0 (CasiVacío)
  ( ((0.80,0.80,0.80), (0.60,0.60,0.60), (0.10,0.10,0.10), (0.05,0.05,0.05)) ), # s_f0 (Crítico,Bajo,Medio,Alto) for (P,N,R) of s_f1
  # o_m0 = 1 (Mitad)
  ( ((0.15,0.15,0.15), (0.30,0.30,0.30), (0.80,0.80,0.80), (0.40,0.40,0.40)) ),
  # o_m0 = 2 (Lleno)
  ( ((0.05,0.05,0.05), (0.10,0.10,0.10), (0.10,0.10,0.10), (0.55,0.55,0.55)) )
}

# A_m1[obs_trade, state_fuel, state_market]. Fuel level (s_f0) has no direct effect on trade outcome.
# Probabilities sum to 1 over obs_trade for each (state_fuel, state_market) pair.
A_m1={ # Likelihood P(o_m1 | s_f0, s_f1)
  # o_m1 = 0 (Pérdida)
  ( ((0.7,0.2,0.1), (0.7,0.2,0.1), (0.7,0.2,0.1), (0.7,0.2,0.1)) ), # s_f1 (P,N,R) for (Critico,Bajo,Medio,Alto) of s_f0
  # o_m1 = 1 (SinCambio)
  ( ((0.2,0.6,0.2), (0.2,0.6,0.2), (0.2,0.6,0.2), (0.2,0.6,0.2)) ),
  # o_m1 = 2 (Ganancia)
  ( ((0.1,0.2,0.7), (0.1,0.2,0.7), (0.1,0.2,0.7), (0.1,0.2,0.7)) )
}

# B_f0[next_fuel, prev_fuel, prev_market, action_investment]
# Fuel consumption: normal with 'Mantener'. Accelerated if trading ('Invertir'/'Desinvertir') during 'Pérdidas' market.
# Each innermost tuple is for actions (Desinvertir, Mantener, Invertir)
# For simplicity, only showing a few key transitions. A full B_f0 is large.
# This will be visualized as a (4*4*3)x3 = 48x3 matrix by the visualizer.
B_f0={ # P(s_f0' | s_f0, s_f1, u_c0)
  # s_f0' = 0 (Crítico)
  ( # s_f0 = 0 (Crítico)
    ( (1.0,1.0,1.0), (1.0,1.0,1.0), (1.0,1.0,1.0) ), # Stays Crítico
    # s_f0 = 1 (Bajo)
    ( (1.0,0.4,1.0), (0.4,0.1,0.4), (0.1,0.0,0.1) ), # If s_f1=Pérdidas (idx 0) & trade (act 0 or 2) -> Crítico
                                                    # If s_f1=Neutral/Rentable or act=Mantener -> less likely Crítico
    # s_f0 = 2 (Medio)
    ( (0.8,0.1,0.8), (0.1,0.0,0.1), (0.0,0.0,0.0) ),
    # s_f0 = 3 (Alto)
    ( (0.5,0.0,0.5), (0.0,0.0,0.0), (0.0,0.0,0.0) )
  ),
  # s_f0' = 1 (Bajo)
  (
    ( (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0) ), # From Crítico (cannot improve)
    ( (0.0,0.6,0.0), (0.6,0.8,0.6), (0.8,0.9,0.8) ), # From Bajo
    ( (0.2,0.8,0.2), (0.8,0.1,0.8), (0.1,0.0,0.1) ), # From Medio
    ( (0.4,0.1,0.4), (0.1,0.0,0.1), (0.0,0.0,0.0) )  # From Alto
  ),
  # s_f0' = 2 (Medio)
  (
    ( (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0) ), # From Crítico
    ( (0.0,0.0,0.0), (0.0,0.1,0.0), (0.1,0.1,0.1) ), # From Bajo
    ( (0.0,0.1,0.0), (0.1,0.9,0.1), (0.9,0.9,0.9) ), # From Medio
    ( (0.1,0.9,0.1), (0.9,0.1,0.9), (0.1,0.0,0.1) )  # From Alto
  ),
  # s_f0' = 3 (Alto)
  (
    ( (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0) ), # From Crítico
    ( (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0) ), # From Bajo
    ( (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.1,0.0) ), # From Medio
    ( (0.0,0.0,0.0), (0.0,0.9,0.0), (0.9,0.9,0.9) )  # From Alto (stays Alto if not trading in loss, or Mantener)
  )
  # Note: Probabilities for B_f0 need to sum to 1 over s_f0' for each (s_f0, s_f1, u_c0) combination.
  # The above is illustrative and would need careful normalization. For a real model, these would be precise.
  # Placeholder: For robust parsing, ensure this complex structure is handled or simplified if needed by the target execution engine.
  # For visualizer, it will become many rows.
}

# B_f1[next_market, prev_market, action_investment]
# Market transitions. Investing in good market might keep it good. Divesting from bad market might mitigate.
# This will be visualized as (3*3)x3 = 9x3 matrix.
B_f1={ # P(s_f1' | s_f1, u_c0)
  # s_f1' = 0 (Pérdidas)
  ( ((0.7,0.5,0.4), (0.4,0.3,0.2), (0.2,0.1,0.1)) ), # s_f1 (P,N,R) for actions (D,M,I)
  # s_f1' = 1 (Neutral)
  ( ((0.2,0.4,0.4), (0.5,0.6,0.5), (0.4,0.3,0.2)) ),
  # s_f1' = 2 (Rentable)
  ( ((0.1,0.1,0.2), (0.1,0.1,0.3), (0.4,0.6,0.7)) )
  # Probabilities sum to 1 over s_f1' for each (s_f1, u_c0) pair.
}

# C_m0: Preferences for IndicadorCombustible (0:CasiVacío, 1:Mitad, 2:Lleno)
C_m0={(-5.0, 0.0, 5.0)} # Strongly prefer Lleno, strongly disprefer CasiVacío

# C_m1: Preferences for ResultadoOperacion (0:Pérdida, 1:SinCambio, 2:Ganancia)
C_m1={(-5.0, 0.0, 5.0)} # Strongly prefer Ganancia, strongly disprefer Pérdida

# D_f0: Prior for NivelCombustible (0:Crítico, 1:Bajo, 2:Medio, 3:Alto)
D_f0={(0.0, 0.0, 0.1, 0.9)} # Start Alto

# D_f1: Prior for RendimientoMercado (0:Pérdidas, 1:Neutral, 2:Rentable)
D_f1={(0.2, 0.6, 0.2)} # Start Neutral, some chance of P/R

## Equations
# Standard POMDP / Active Inference equations for:
# 1. State estimation (approximate posterior over hidden states)
#    q(s_t) = σ( ln(A^T o_t) + Prior_t )
# 2. Policy evaluation (Expected Free Energy)
#    G(π) = E_q(o_t, s_t | π) [ C(o_t) - D_KL[q(s_t|o_t,π)||q(s_t|π)] - H[q(o_t|s_t,π)] ]
# 3. Action selection (Softmax over -G)
#    P(u_t|π) = σ(-G(π))

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=20 # Example planning horizon

## ActInfOntologyAnnotation
# Hidden States
s_f0=NivelDeCombustible
s_f1=DesempenoDelMercado

# Observations
o_m0=MedidorDeCombustible
o_m1=ResultadoDeTransaccion

# Control/Policy Related
pi_c0=VectorPoliticaControl0
u_c0=AccionControl0

# Likelihoods
A_m0=MatrizVerosimilitudModalidad0
A_m1=MatrizVerosimilitudModalidad1

# Transitions
B_f0=MatrizTransicionFactor0
B_f1=MatrizTransicionFactor1

# Preferences
C_m0=VectorPreferenciaModalidad0
C_m1=VectorPreferenciaModalidad1

# Priors
D_f0=DistribucionPreviaFactorOculto0
D_f1=DistribucionPreviaFactorOculto1

# Other
G=EnergiaLibreEsperada
t=PasoDeTiempo

## ModelParameters
num_hidden_states_factors: [4, 3]  # s_f0[4], s_f1[3]
num_obs_modalities: [3, 3]     # o_m0[3], o_m1[3]
num_control_actions: [3]       # u_c0[3] (single control factor)

## Footer
Airplane Trading POMDP v1.0 - End of Specification.
Note: B_f0 parameterization is illustrative and requires careful normalization for a functional model.

## Signature
Creator: GNN Example Generator
Date: 2024-07-29
Status: Example for testing and demonstration. 