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
```

### 2. Conflict Monitoring and Resolution

```gnn
## ModelName
ConflictMonitoringSystem

## ModelAnnotation
Models conflict monitoring through prediction error and resolution through precision allocation.

## StateSpaceBlock
s_f0[4,1,type=categorical]              ### Conflict level: None=0, Low=1, Medium=2, High=3
s_f1[5,1,type=categorical]              ### Response tendency: Strong_A=0, Weak_A=1, Neutral=2, Weak_B=3, Strong_B=4
s_f2[3,1,type=categorical]              ### Control engagement: None=0, Partial=1, Full=2

o_m0[5,1,type=categorical]              ### Stimulus compatibility: Congruent=0, Neutral=1, Incongruent=2, Novel=3, Ambiguous=4
o_m1[4,1,type=categorical]              ### Response conflict: Low=0, Medium=1, High=2, Resolution=3

u_c0[4,1,type=categorical]              ### Control action: Monitor=0, Engage=1, Resolve=2, Disengage=3

## InitialParameterization
# Conflict detection sensitivity
conflict_threshold = 0.5                ### Threshold for conflict detection
conflict_sensitivity = 1.5              ### Sensitivity to conflict signals

# Resolution parameters
resolution_strength = 0.8               ### Strength of conflict resolution
resolution_persistence = 0.6            ### Persistence of control engagement

## Equations
# Conflict detection via prediction error
conflict(t) = |PE_response_A(t) - PE_response_B(t)|

# Control engagement threshold
engage_control = conflict(t) > conflict_threshold
```

### 3. Task Switching and Cognitive Flexibility

```gnn
## ModelName
TaskSwitchingModel

## ModelAnnotation
Models task switching through task set reconfiguration and switch costs via precision reallocation.

## StateSpaceBlock
s_f0[4,1,type=categorical]              ### Current task: Task_A=0, Task_B=1, Task_C=2, No_Task=3
s_f1[3,1,type=categorical]              ### Switch state: Maintain=0, Prepare=1, Switch=2
s_f2[4,1,type=categorical]              ### Task readiness: Unprepared=0, Preparing=1, Ready=2, Interfering=3

o_m0[4,1,type=categorical]              ### Task cue: Cue_A=0, Cue_B=1, Cue_C=2, No_Cue=3
o_m1[3,1,type=categorical]              ### Switch signal: Maintain=0, Prepare=1, Switch=2

u_c0[4,1,type=categorical]              ### Switch action: Maintain=0, Prepare=1, Switch=2, Clear=3

## InitialParameterization
# Switch costs
switch_cost_precision = 0.5             ### Reduced precision during switches
preparation_benefit = 0.3               ### Benefit of preparation time

# Task set strength
task_persistence = 0.7                  ### Persistence of current task set
interference_strength = 0.4             ### Cross-task interference

## Equations
# Switch cost as precision reduction
γ_switch(t) = γ_base * (1 - switch_cost_precision * switch_signal(t))

# Preparation benefit
preparation_benefit(t) = preparation_time(t) * preparation_rate
```

## Clinical Applications

### ADHD and Executive Function Deficits

```gnn
## ModelName
ADHDExecutiveModel

## ModelAnnotation
Models ADHD as reduced precision in executive control systems and altered precision allocation.

## ModifiedParameters
# Reduced executive precision
exec_precision = 1.0                    ### Reduced from normal 2.0
precision_stability = 0.4               ### Reduced from normal 0.8

# Altered precision allocation
precision_allocation_flexibility = 0.3   ### Reduced from normal 0.7
distractibility_threshold = 0.3         ### Reduced from normal 0.6

# Working memory limitations
working_memory_capacity = 2             ### Reduced from normal 4
working_memory_decay = 0.15             ### Increased from normal 0.05
```

### Aging and Executive Control

```gnn
## ModelName
AgingExecutiveModel

## ModelAnnotation
Models age-related changes in executive control through altered precision dynamics and reduced flexibility.

## ModifiedParameters
# Reduced processing speed
processing_speed_multiplier = 0.7       ### Reduced from normal 1.0

# Altered precision dynamics
precision_modulation_strength = 0.6     ### Reduced from normal 1.0
precision_switching_cost = 0.4          ### Increased from normal 0.2

# Compensatory mechanisms
experience_based_priors = 1.5           ### Increased from normal 1.0
strategic_processing = 1.3              ### Increased from normal 1.0
```

## Implementation Examples

### Python Implementation Template

```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ExecutiveControlState:
    """Executive control state representation"""
    executive_state: int
    control_mode: int
    goal_hierarchy: int
    cognitive_operation: int
    processing_mode: int
    resource_allocation: int

class ExecutiveControlModel:
    """
    Active Inference implementation of executive control
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize precision and control parameters"""
        self.exec_precision = self.config.get('exec_precision', 2.0)
        self.cog_precision = self.config.get('cog_precision', 1.5)
        self.precision_modulation = self.config.get('precision_modulation', 
                                                  np.ones((4, 4)))
        
    def update_beliefs(self, observations: np.ndarray, 
                      current_state: ExecutiveControlState) -> ExecutiveControlState:
        """Update beliefs through hierarchical message passing"""
        
        # Executive level belief updating
        exec_beliefs = self.update_executive_beliefs(observations, current_state)
        
        # Precision modulation from executive to cognitive level
        modulated_precision = self.modulate_precision(exec_beliefs)
        
        # Cognitive level belief updating with modulated precision
        cog_beliefs = self.update_cognitive_beliefs(observations, 
                                                   current_state, 
                                                   modulated_precision)
        
        return ExecutiveControlState(
            executive_state=exec_beliefs['executive_state'],
            control_mode=exec_beliefs['control_mode'],
            goal_hierarchy=exec_beliefs['goal_hierarchy'],
            cognitive_operation=cog_beliefs['cognitive_operation'],
            processing_mode=cog_beliefs['processing_mode'],
            resource_allocation=cog_beliefs['resource_allocation']
        )
    
    def modulate_precision(self, exec_beliefs: Dict) -> np.ndarray:
        """Modulate precision based on executive control signals"""
        exec_state = exec_beliefs['executive_state']
        return self.precision_modulation[exec_state] * self.cog_precision
```

## Validation Protocols

### Behavioral Validation Tasks

1. **Stroop Task**: Conflict monitoring and resolution
2. **Task Switching Paradigms**: Cognitive flexibility
3. **Working Memory Tasks**: Executive control over memory
4. **Go/No-Go Tasks**: Inhibitory control
5. **Flanker Tasks**: Selective attention and conflict resolution

### Neural Validation Measures

1. **ERP Components**: N2 (conflict monitoring), P3 (updating)
2. **fMRI Activation**: Prefrontal cortex, anterior cingulate
3. **Connectivity Measures**: Frontoparietal control networks

## Research Applications

### Computational Psychiatry Applications

- **ADHD**: Reduced executive precision and altered control dynamics
- **Depression**: Biased goal hierarchies and reduced control engagement
- **Schizophrenia**: Altered precision allocation and hierarchical dysfunction
- **OCD**: Hyperactive conflict monitoring and excessive control

### Educational Applications

- **Cognitive Training**: Targeted executive function enhancement
- **Learning Disabilities**: Understanding control deficits in learning
- **Skill Acquisition**: Role of executive control in learning complex skills

## Future Directions

1. **Multi-Agent Executive Control**: Modeling coordination between multiple agents
2. **Emotional Regulation**: Integration with affective control systems
3. **Developmental Models**: Executive control development across lifespan
4. **Metacognitive Control**: Integration with metacognitive monitoring

## References

### Core Papers
- Diamond, A. (2013). Executive functions. Annual Review of Psychology
- Miyake, A., et al. (2000). The unity and diversity of executive functions
- Braver, T. S. (2012). The variable nature of cognitive control

### Active Inference Applications
- FitzGerald, T. H., et al. (2015). Active inference, evidence accumulation, and the urn task
- Parr, T., & Friston, K. J. (2017). Working memory, attention, and salience in active inference 