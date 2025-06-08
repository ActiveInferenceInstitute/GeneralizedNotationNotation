# Executive Control and Cognitive Control

## Overview

Executive control encompasses the higher-order cognitive processes that coordinate and control other cognitive functions. In Active Inference, executive control emerges from hierarchical belief updating and precision-weighted prediction error minimization across multiple timescales.

## Core Components

### 1. Cognitive Control Architecture

Executive control in Active Inference operates through hierarchical message passing where higher levels set priors for lower levels, implementing top-down control through precision modulation.

```gnn
## ModelName
ExecutiveControlHierarchy

## ModelAnnotation
Models executive control as hierarchical precision-weighted belief updating.
Higher levels modulate precision at lower levels to implement cognitive control.

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: hierarchical_control, precision_modulation

## StateSpaceBlock
# Executive level (slow timescale)
s_exec_f0[4,1,type=categorical]         ### Executive state: Monitor=0, Switch=1, Inhibit=2, Update=3
s_exec_f1[3,1,type=categorical]         ### Control mode: Proactive=0, Reactive=1, Automatic=2
s_exec_f2[5,1,type=categorical]         ### Goal hierarchy: Goal_L1=0, Goal_L2=1, Goal_L3=2, Goal_L4=3, No_Goal=4

# Cognitive level (medium timescale)
s_cog_f0[6,1,type=categorical]          ### Cognitive operation: Attend=0, Inhibit=1, Switch=2, Update=3, Monitor=4, Execute=5
s_cog_f1[4,1,type=categorical]          ### Processing mode: Automatic=0, Controlled=1, Conflict=2, Error=3
s_cog_f2[3,1,type=categorical]          ### Resource allocation: Low=0, Medium=1, High=2

# Perceptual/Motor level (fast timescale)
s_perc_f0[8,1,type=categorical]         ### Perceptual features: Feature_A through Feature_H
s_motor_f0[4,1,type=categorical]        ### Motor preparation: Ready=0, Prepare=1, Execute=2, Complete=3

## Observations
o_m0[6,1,type=categorical]              ### Task demands: Simple=0, Complex=1, Novel=2, Conflict=3, Switch=4, Error=5
o_m1[4,1,type=categorical]              ### Performance feedback: Success=0, Error=1, Conflict=2, Slow=3
o_m2[3,1,type=categorical]              ### Resource demand: Low=0, Medium=1, High=2

## Actions
u_c0[5,1,type=categorical]              ### Executive action: Maintain=0, Switch=1, Inhibit=2, Update=3, Disengage=4
u_c1[4,1,type=categorical]              ### Precision control: Increase=0, Decrease=1, Maintain=2, Redistribute=3

## Connections
# Hierarchical control
s_exec_f0 > s_cog_f0                    ### Executive state controls cognitive operations
s_exec_f1 > s_cog_f1                    ### Control mode influences processing mode
s_exec_f2 > s_cog_f2                    ### Goal hierarchy determines resource allocation

# Cognitive control
s_cog_f0 > s_perc_f0                    ### Cognitive operations control perception
s_cog_f1 > s_motor_f0                   ### Processing mode influences motor preparation

# Feedback loops
o_m1 > s_exec_f0                        ### Performance feedback updates executive state
o_m2 > s_exec_f2                        ### Resource demand updates goal hierarchy

## InitialParameterization
# Hierarchical precision parameters
exec_precision = 2.0                    ### High precision for executive control
cog_precision = 1.5                     ### Medium precision for cognitive level
perc_precision = 1.0                    ### Standard precision for perception

# Control strength parameters
proactive_control_strength = 0.8        ### Strength of proactive control
reactive_control_strength = 0.6         ### Strength of reactive control
automatic_processing_weight = 0.3       ### Weight for automatic processing

# Precision modulation matrices
precision_modulation = [
    [2.0, 1.8, 1.6, 1.4],              ### High control: strong precision modulation
    [1.5, 1.3, 1.1, 0.9],              ### Medium control: moderate modulation
    [1.0, 1.0, 1.0, 1.0],              ### Low control: minimal modulation
    [0.8, 0.6, 0.4, 0.2]               ### Inhibited: reduced precision
]

# A matrices with precision-dependent accuracy
A_m0_exec = [
    [0.9, 0.05, 0.03, 0.01, 0.005, 0.005],  # High executive control
    [0.7, 0.1, 0.08, 0.06, 0.04, 0.02],     # Medium executive control
    [0.5, 0.15, 0.12, 0.1, 0.08, 0.05],     # Low executive control
    [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]        # Automatic processing
]

# Preferences for control and performance
C_m1 = [2.0, -1.0, -0.5, -1.5]         ### Strong preference for success, avoid errors

## Equations
# Executive control precision modulation
γ_cog(t) = γ_base + α_exec * s_exec_precision(t)

# Hierarchical belief updating with precision weighting
Q(s_cog|π) ∝ exp(γ_cog * ln P(o|s_cog) + ln P(s_cog|s_exec))

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 100

## Footer
This model implements hierarchical executive control through Active Inference principles.