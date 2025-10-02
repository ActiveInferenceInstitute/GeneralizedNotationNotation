## GNNVersionAndFlags
Version: 1.0

## ModelName
Active Inference Chronic Pain Multi-Theory Model v1

## ModelAnnotation
This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:

**Multi-Theory Integration:**
- Peripheral Sensitization: Enhanced nociceptor responsiveness and reduced thresholds (slow timescale)
- Central Sensitization: Amplified CNS processing and reduced inhibition (slow timescale, one-way process)
- Gate Control Theory: Spinal modulation of ascending pain signals (fast timescale)
- Neuromatrix Theory: Distributed network generating pain experience (fast-medium coupling)
- Predictive Coding: Pain as precision-weighted prediction error (all timescales)
- Biopsychosocial Integration: Cognitive, emotional, and behavioral factors (medium timescale)

**Three Nested Timescales:**
1. Fast (ms-s): Neural signaling, gate control, descending modulation, acute pain perception
2. Medium (min-hrs): Cognitive-affective processes, behavioral strategies, functional capacity
3. Slow (hrs-days): Tissue healing, peripheral/central sensitization, chronic adaptations

**State Space Structure:**
- Six hidden state factors (378 combinations): tissue state (slow), peripheral sensitivity (slow), spinal gate (fast), central sensitization (slow), descending modulation (fast), cognitive-affective state (medium)
- Four observation modalities (72 outcomes): pain intensity (fast), pain quality (fast), functional capacity (medium), autonomic response (fast)
- Four control factors (81 actions): attention allocation (medium), behavioral strategy (medium), cognitive reappraisal (medium), descending control (fast)

**Key Features:**
- Timescale separation: ε (fast/medium) ≈ 10^-3, δ (medium/slow) ≈ 10^-2
- Cross-timescale coupling: slow states modulate fast dynamics, fast observations (averaged) drive medium cognition, medium behaviors (averaged) influence slow healing
- Testable predictions about pain chronification pathways across multiple timescales
- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)

## StateSpaceBlock
A[72,378],float
B[378,378,81],float
C[72],float
D[378],float
E[81],float
T[3,1],float
P_sens[3,1],float
G[3,1],float
C_sens[2,1],float
D_mod[3,1],float
Cog[7,1],float
Pain_I[4,1],float
Pain_Q[3,1],float
Func[3,1],float
Auto[2,1],float
Attn[3,1],float
Behav[3,1],float
Reapp[3,1],float
Desc_C[3,1],float
F[1],float
G[1],float
t_fast[1],float
t_medium[1],float
t_slow[1],float

## Connections
D>T
D>P_sens
D>C_sens
D>Cog
T>P_sens
P_sens>B
P_sens>G
D_mod>G
G>B
G>C_sens
C_sens>B
T>C_sens
Cog>D_mod
D_mod>B
Pain_I>Cog
Func>Cog
Cog>B
T>A
P_sens>A
G>A
C_sens>A
D_mod>A
Cog>A
Attn>Cog
Behav>T
Behav>Func
Reapp>Cog
Desc_C>D_mod
C>G
E>Attn
E>Behav
G>Attn
G>Behav
G>Reapp
G>Desc_C
Attn>Behav
Behav>Reapp
Reapp>Desc_C

## InitialParameterization
A = [[0.7, 0.25, 0.05, 0.0, 0.8, 0.15, 0.05, 0.9, 0.08, 0.02, 0.85, 0.15], [0.6, 0.3, 0.1, 0.0, 0.75, 0.2, 0.05, 0.85, 0.1, 0.05, 0.8, 0.2], [0.45, 0.35, 0.15, 0.05, 0.7, 0.2, 0.1, 0.75, 0.15, 0.1, 0.7, 0.3], [0.2, 0.3, 0.35, 0.15, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.4, 0.6], [0.3, 0.4, 0.25, 0.05, 0.6, 0.25, 0.15, 0.6, 0.25, 0.15, 0.5, 0.5], [0.4, 0.35, 0.2, 0.05, 0.65, 0.25, 0.1, 0.7, 0.2, 0.1, 0.6, 0.4], [0.55, 0.3, 0.12, 0.03, 0.7, 0.22, 0.08, 0.8, 0.15, 0.05, 0.75, 0.25], [0.8, 0.15, 0.05, 0.0, 0.85, 0.12, 0.03, 0.95, 0.04, 0.01, 0.9, 0.1], [0.75, 0.2, 0.05, 0.0, 0.8, 0.15, 0.05, 0.9, 0.08, 0.02, 0.85, 0.15], [0.65, 0.25, 0.08, 0.02, 0.75, 0.18, 0.07, 0.85, 0.12, 0.03, 0.8, 0.2], [0.9, 0.08, 0.02, 0.0, 0.9, 0.08, 0.02, 0.98, 0.02, 0.0, 0.95, 0.05], [0.85, 0.12, 0.03, 0.0, 0.88, 0.1, 0.02, 0.95, 0.04, 0.01, 0.92, 0.08], [0.05, 0.15, 0.45, 0.35, 0.3, 0.4, 0.3, 0.2, 0.4, 0.4, 0.25, 0.75], [0.02, 0.1, 0.4, 0.48, 0.2, 0.35, 0.45, 0.1, 0.35, 0.55, 0.15, 0.85], [0.35, 0.4, 0.2, 0.05, 0.7, 0.25, 0.05, 0.7, 0.2, 0.1, 0.75, 0.25], [0.2, 0.35, 0.3, 0.15, 0.55, 0.3, 0.15, 0.55, 0.3, 0.15, 0.6, 0.4], [0.15, 0.3, 0.4, 0.15, 0.4, 0.35, 0.25, 0.4, 0.35, 0.25, 0.45, 0.55], [0.05, 0.2, 0.45, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.35, 0.65], [0.4, 0.4, 0.15, 0.05, 0.75, 0.2, 0.05, 0.8, 0.15, 0.05, 0.8, 0.2], [0.25, 0.4, 0.25, 0.1, 0.65, 0.25, 0.1, 0.65, 0.25, 0.1, 0.7, 0.3], [0.2, 0.35, 0.3, 0.15, 0.55, 0.3, 0.15, 0.5, 0.3, 0.2, 0.6, 0.4], [0.08, 0.25, 0.4, 0.27, 0.4, 0.35, 0.25, 0.35, 0.4, 0.25, 0.45, 0.55], [0.05, 0.2, 0.4, 0.35, 0.3, 0.4, 0.3, 0.25, 0.45, 0.3, 0.35, 0.65], [0.02, 0.1, 0.35, 0.53, 0.2, 0.35, 0.45, 0.15, 0.45, 0.4, 0.2, 0.8], [0.02, 0.08, 0.3, 0.6, 0.15, 0.3, 0.55, 0.1, 0.4, 0.5, 0.15, 0.85], [0.1, 0.3, 0.4, 0.2, 0.5, 0.35, 0.15, 0.45, 0.35, 0.2, 0.5, 0.5], [0.05, 0.25, 0.45, 0.25, 0.4, 0.4, 0.2, 0.4, 0.4, 0.2, 0.45, 0.55], [0.03, 0.15, 0.42, 0.4, 0.3, 0.4, 0.3, 0.3, 0.45, 0.25, 0.35, 0.65], [0.01, 0.08, 0.35, 0.56, 0.2, 0.35, 0.45, 0.2, 0.45, 0.35, 0.25, 0.75], [0.01, 0.05, 0.3, 0.64, 0.15, 0.3, 0.55, 0.15, 0.5, 0.35, 0.2, 0.8], [0.0, 0.02, 0.2, 0.78, 0.1, 0.25, 0.65, 0.08, 0.45, 0.47, 0.1, 0.9], [0.7, 0.25, 0.05, 0.0, 0.8, 0.15, 0.05, 0.8, 0.15, 0.05, 0.85, 0.15], [0.55, 0.35, 0.08, 0.02, 0.75, 0.2, 0.05, 0.75, 0.2, 0.05, 0.8, 0.2], [0.15, 0.3, 0.4, 0.15, 0.45, 0.35, 0.2, 0.5, 0.35, 0.15, 0.55, 0.45], [0.05, 0.2, 0.45, 0.3, 0.3, 0.4, 0.3, 0.35, 0.4, 0.25, 0.4, 0.6]]
B = []
C = [[2.2, 1.4, 0.5, 0.7, -0.1, -0.8, -0.5, -1.3, -2.2, -1.7, -2.5, -3.4, 2.0, 1.2, 0.3, 0.5, -0.3, -1.0, -0.7, -1.5, -2.4, -1.9, -2.7, -3.6, 1.6, 0.8, -0.1, 0.1, -0.7, -1.4, -1.1, -1.9, -2.8, -2.3, -3.1, -4.0, 1.0, 0.2, -0.7, -0.5, -1.3, -2.0, -1.7, -2.5, -3.4, -2.9, -3.7, -4.6, 0.5, -0.3, -1.2, -1.0, -1.8, -2.5, -2.2, -3.0, -3.9, -3.4, -4.2, -5.1, -0.2, -1.0, -1.9, -1.7, -2.5, -3.2, -2.9, -3.7, -4.6, -4.1, -4.9, -5.8]]
D = [[0.1, 0.5, 0.4, 0.6, 0.3, 0.1, 0.4, 0.4, 0.2, 0.9, 0.1, 0.2, 0.6, 0.2, 0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]]
E = [[0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015, 0.03, 0.045, 0.03, 0.05, 0.075, 0.05, 0.03, 0.045, 0.03, 0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015, 0.01, 0.015, 0.01, 0.017, 0.025, 0.017, 0.01, 0.015, 0.01, 0.02, 0.03, 0.02, 0.033, 0.05, 0.033, 0.02, 0.03, 0.02, 0.01, 0.015, 0.01, 0.017, 0.025, 0.017, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006, 0.012, 0.018, 0.012, 0.02, 0.03, 0.02, 0.012, 0.018, 0.012, 0.006, 0.009, 0.006, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006]]
epsilon_fast_medium: 0.001     # ε = τ_fast / τ_medium ≈ 10^-3
delta_medium_slow: 0.01        # δ = τ_medium / τ_slow ≈ 10^-2
window_fast_to_medium: 300     # s (5 min) - Averaging window for fast→medium coupling
window_medium_to_slow: 14400   # s (4 hours) - Averaging window for medium→slow coupling
population_type: "acute"       # Options: "acute", "chronic", "high_risk", "resilient"

## Time
Dynamic
ModelTimeHorizon = Unbounded # Chronic pain model for longitudinal simulation across multiple timescales

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrices
C = LogPreferenceVector
D = PriorOverHiddenStates
E = HabitVector
F = VariationalFreeEnergy
G = ExpectedFreeEnergy
T = TissueState
P_sens = PeripheralSensitization
G = SpinalGateState
C_sens = CentralSensitization
D_mod = DescendingModulation
Cog = CognitiveAffectiveState
Pain_I = PainIntensityObservation
Pain_Q = PainQualityObservation
Func = FunctionalCapacityObservation
Auto = AutonomicResponseObservation
Attn = AttentionAllocationControl
Behav = BehavioralStrategyControl
Reapp = CognitiveReappraisalControl
Desc_C = DescendingControlAction
t_fast = FastTimescale
t_medium = MediumTimescale
t_slow = SlowTimescale

## Footer
Generated: 2025-10-02T10:52:16.320655

## Signature
