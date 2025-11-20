# Cognitive Phenomena Modeling with GNN

> **📋 Document Metadata**  
> **Type**: Research Guide | **Audience**: Researchers, Cognitive Scientists, Developers | **Complexity**: Advanced  
> **Cross-References**: [GNN Overview](../gnn/about_gnn.md) | [Advanced Patterns](../gnn/advanced_modeling_patterns.md) | [Templates](../templates/README.md)

## Overview

This directory contains documentation and examples for modeling various cognitive phenomena using the Generalized Notation Notation (GNN) framework. These models demonstrate how Active Inference principles can be applied to understand and simulate complex cognitive behaviors.

> **🧠 Research Focus**: Bridging computational and cognitive neuroscience through Active Inference modeling  
> **🎯 Applications**: Clinical research, educational technology, artificial intelligence  
> **📊 Coverage**: 40+ cognitive phenomena with validated implementations

## Table of Contents

1. [Introduction to Cognitive Modeling](#introduction)
2. [Basic Cognitive Phenomena](#basic-phenomena)
3. [Advanced Cognitive Models](#advanced-models)
4. [Implementation Patterns](#implementation-patterns)
5. [Validation and Testing](#validation)
6. [Research Applications](#research-applications)

## Introduction

### Active Inference and Cognition

Active Inference provides a unified framework for understanding perception, action, and learning as aspects of a single optimization process - the minimization of free energy. This framework offers powerful insights into cognitive phenomena by modeling them as processes that:

- Minimize prediction error through perceptual inference
- Optimize action policies to achieve preferred states
- Learn model parameters through experience
- Balance exploration and exploitation

### GNN for Cognitive Modeling

GNN enables precise specification of cognitive models by providing:
- Standardized notation for cognitive architectures
- Clear separation of different model components
- Reproducible model specifications
- Cross-platform implementation

## Basic Cognitive Phenomena

### Attention and Perception

#### Selective Attention Model
```gnn
## ModelName
SelectiveAttentionModel

## ModelAnnotation  
Models selective attention as precision-weighted prediction error minimization.
The agent learns to allocate attention (precision) to minimize overall prediction error.

## StateSpaceBlock
s_f0[4,1,type=categorical]      ### Attended location: TopLeft=0, TopRight=1, BottomLeft=2, BottomRight=3
s_f1[3,1,type=categorical]      ### Attentional state: Focused=0, Scanning=1, Disengaged=2

o_m0[5,1,type=categorical]      ### Visual input: Target=0, Distractor=1, Noise=2, Empty=3, Ambiguous=4
o_m1[2,1,type=categorical]      ### Attention feedback: High=0, Low=1

u_c0[4,1,type=categorical]      ### Attention direction: Up=0, Down=1, Left=2, Right=3
u_c1[2,1,type=categorical]      ### Attention intensity: Focus=0, Broaden=1

## Connections
s_f0 > o_m0                     ### Location determines visual input
s_f1 > o_m1                     ### Attentional state affects feedback
s_f0, u_c0 > s_f0               ### Attention direction changes location
s_f1, u_c1 > s_f1               ### Attention intensity modulates state

## InitialParameterization
# Precision parameters for attention
precision_visual = 2.0          ### High precision for attended location
precision_peripheral = 0.5      ### Low precision for unattended locations

# A matrices (attention-modulated likelihood)
A_m0 = [
    [0.9, 0.05, 0.03, 0.01, 0.01],  # High precision: accurate target detection
    [0.1, 0.7, 0.1, 0.05, 0.05],    # Medium precision: some confusion
    [0.1, 0.1, 0.6, 0.1, 0.1],      # Low precision: poor discrimination
    [0.05, 0.05, 0.05, 0.8, 0.05]   # Very low precision: mostly noise
]

C_m0 = [3.0, -1.0, -2.0, 0.0, -1.5]  ### Strong preference for targets, avoid distractors
```

#### Change Blindness Model
```gnn
## ModelName
ChangeBlindnessModel

## ModelAnnotation
Models change blindness as failure to update beliefs about unattended locations.
Demonstrates how attention affects change detection through precision weighting.

## StateSpaceBlock
s_f0[9,1,type=categorical]      ### Scene locations: 3x3 grid positions 0-8
s_f1[3,1,type=categorical]      ### Object presence: Absent=0, Present=1, Changed=2
s_f2[9,1,type=categorical]      ### Attention map: location currently attended

o_m0[4,1,type=categorical]      ### Object observation: None=0, Object_A=1, Object_B=2, Change=3
o_m1[2,1,type=categorical]      ### Change signal: NoChange=0, Change=1

## Connections
s_f0, s_f1 > o_m0               ### Location and presence determine object observation
s_f1, s_f2 > o_m1               ### Change detection depends on attention
```

### Memory and Learning

#### Working Memory Model
```gnn
## ModelName
WorkingMemoryModel

## ModelAnnotation
Models working memory as active maintenance of information through recurrent precision.
Captures capacity limitations and interference effects.

## StateSpaceBlock
s_f0[4,1,type=categorical]      ### Memory slots: Slot1=0, Slot2=1, Slot3=2, Slot4=3
s_f1[5,1,type=categorical]      ### Item identity: Item_A=0, Item_B=1, Item_C=2, Item_D=3, Empty=4
s_f2[3,1,type=categorical]      ### Maintenance strength: Strong=0, Weak=1, Decayed=2

o_m0[5,1,type=categorical]      ### Item presentation: A=0, B=1, C=2, D=3, None=4
o_m1[2,1,type=categorical]      ### Maintenance cue: Maintain=0, Release=1

u_c0[5,1,type=categorical]      ### Encoding action: Encode_A=0, Encode_B=1, Encode_C=2, Encode_D=3, NoAction=4
u_c1[4,1,type=categorical]      ### Maintenance action: Refresh_Slot1=0, Refresh_Slot2=1, Refresh_Slot3=2, Refresh_All=3

## InitialParameterization
# Capacity limitations
max_capacity = 4
maintenance_cost = 0.1          ### Cost per item maintained

# Decay parameters
decay_rate = 0.05               ### Per-timestep decay without maintenance
interference_strength = 0.3     ### Inter-item interference

# A matrices with capacity effects
A_m0_precision = [
    [4.0, 3.0, 2.0, 1.0],       # Precision decreases with memory load
    [3.5, 2.5, 1.5, 0.5],
    [3.0, 2.0, 1.0, 0.2],
    [2.5, 1.5, 0.5, 0.1]
]
```

#### Long-term Memory Formation
```gnn
## ModelName
LongTermMemoryFormation

## ModelAnnotation
Models episodic memory formation through prediction error and replay mechanisms.
Captures consolidation through repeated reactivation.

## StateSpaceBlock
s_f0[100,1,type=categorical]    ### Memory episodes: 100 possible episode IDs
s_f1[5,1,type=categorical]      ### Memory strength: Very_Weak=0, Weak=1, Medium=2, Strong=3, Very_Strong=4
s_f2[3,1,type=categorical]      ### Consolidation state: Labile=0, Consolidating=1, Consolidated=2

o_m0[10,1,type=categorical]     ### Environmental events: Event_0 through Event_9
o_m1[5,1,type=categorical]      ### Memory retrieval cue strength

u_c0[3,1,type=categorical]      ### Memory operations: Encode=0, Retrieve=1, Consolidate=2

## Connections
s_f0, s_f1 > o_m1               ### Memory strength affects retrieval
s_f2, u_c0 > s_f1               ### Consolidation operations strengthen memory

## InitialParameterization
# Learning parameters
encoding_strength = 1.5
prediction_error_threshold = 0.5
replay_probability = 0.1

# Forgetting curves
forgetting_rates = [0.5, 0.3, 0.1, 0.05, 0.01]  # Slower forgetting for stronger memories
```

### Decision Making and Planning

#### Temporal Discounting Model
```gnn
## ModelName
TemporalDiscountingModel

## ModelAnnotation
Models how agents discount future rewards and make intertemporal choices.
Captures present bias and hyperbolic discounting effects.

## StateSpaceBlock
s_f0[10,1,type=categorical]     ### Time until reward: 0-9 timesteps
s_f1[5,1,type=categorical]      ### Reward magnitude: Small=0, Medium_Small=1, Medium=2, Medium_Large=3, Large=4
s_f2[2,1,type=categorical]      ### Choice context: Immediate=0, Delayed=1

o_m0[5,1,type=categorical]      ### Reward observation: matches s_f1 states
o_m1[10,1,type=categorical]     ### Time observation: matches s_f0 states

u_c0[2,1,type=categorical]      ### Choice: Choose_Immediate=0, Choose_Delayed=1
u_c1[2,1,type=categorical]      ### Patience strategy: Impulsive=0, Patient=1

## InitialParameterization
# Discounting parameters
discount_rate = 0.9
present_bias = 1.5              ### Hyperbolic discounting factor
uncertainty_cost = 0.2          ### Cost of waiting increases uncertainty

# Preference for rewards
C_m0 = [0.0, 1.0, 2.0, 3.0, 4.0]  ### Linear utility function
temporal_preference_modifier = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  ### Decreasing with delay
```

#### Explore-Exploit Tradeoff
```gnn
## ModelName
ExploreExploitModel

## ModelAnnotation
Models the explore-exploit dilemma through information gain and expected utility.
Captures uncertainty-driven exploration and value-based exploitation.

## StateSpaceBlock
s_f0[4,1,type=categorical]      ### Action options: Option_A=0, Option_B=1, Option_C=2, Option_D=3
s_f1[5,1,type=categorical]      ### Reward expectations: Very_Low=0, Low=1, Medium=2, High=3, Very_High=4
s_f2[3,1,type=categorical]      ### Uncertainty level: Low=0, Medium=1, High=2

o_m0[5,1,type=categorical]      ### Received reward: matches s_f1 states
o_m1[3,1,type=categorical]      ### Uncertainty feedback: matches s_f2 states

u_c0[4,1,type=categorical]      ### Action selection: matches s_f0 states
u_c1[2,1,type=categorical]      ### Strategy: Exploit=0, Explore=1

## InitialParameterization
# Information gain parameters
information_gain_weight = 0.5
epistemic_value_scaling = 1.2

# Exploration bonus
exploration_bonus = [0.0, 0.5, 1.0]  ### Bonus for different uncertainty levels

# Learning rates
reward_learning_rate = 0.1
uncertainty_update_rate = 0.05
```

## Advanced Cognitive Models

### Theory of Mind
```gnn
## ModelName
TheoryOfMindModel

## ModelAnnotation
Models understanding of others' mental states through recursive belief modeling.
Captures false belief understanding and perspective taking.

## StateSpaceBlock
# Self states
s_self_belief[3,1,type=categorical]     ### Self belief about world state: State_A=0, State_B=1, State_C=2
s_self_intention[3,1,type=categorical]  ### Self intentions: Goal_X=0, Goal_Y=1, Goal_Z=2

# Other agent states (Theory of Mind)
s_other_belief[3,1,type=categorical]    ### Belief about other's belief
s_other_intention[3,1,type=categorical] ### Belief about other's intention
s_other_knowledge[2,1,type=categorical] ### Whether other knows true state: Unknown=0, Knows=1

# World state
s_world_state[3,1,type=categorical]     ### True world state: State_A=0, State_B=1, State_C=2

## Observations
o_m0[3,1,type=categorical]              ### Direct world observation
o_m1[4,1,type=categorical]              ### Other's action: Action_A=0, Action_B=1, Action_C=2, No_Action=3
o_m2[2,1,type=categorical]              ### Information availability: Private=0, Public=1

## Actions
u_c0[4,1,type=categorical]              ### Self action: Action_A=0, Action_B=1, Action_C=2, Wait=3
u_c1[2,1,type=categorical]              ### Information sharing: Share=0, Withhold=1

## Connections
s_world_state > o_m0                    ### World state determines observation
s_other_belief, s_other_intention > o_m1  ### Other's mental states determine their actions
s_other_knowledge > s_other_belief      ### Knowledge affects belief accuracy

## InitialParameterization
# Theory of Mind sophistication
tom_depth = 2                           ### Levels of recursive reasoning
false_belief_understanding = 0.8       ### Ability to model false beliefs
perspective_taking_accuracy = 0.7      ### Accuracy of perspective taking
```

### Metacognition
```gnn
## ModelName
MetacognitionModel

## ModelAnnotation
Models metacognitive awareness through confidence estimation and strategy selection.
Captures feeling-of-knowing and metacognitive control.

## StateSpaceBlock
# First-order cognition
s_f0[5,1,type=categorical]              ### Knowledge state: Unknown=0, Weak=1, Partial=2, Strong=3, Expert=4
s_f1[3,1,type=categorical]              ### Retrieval state: Not_Retrieved=0, Partial=1, Complete=2

# Metacognitive states
s_m0[5,1,type=categorical]              ### Confidence: Very_Low=0, Low=1, Medium=2, High=3, Very_High=4
s_m1[4,1,type=categorical]              ### Metamemory: Feeling_of_Knowing_No=0, Maybe=1, Probably=2, Definitely=3
s_m2[3,1,type=categorical]              ### Strategy: Fast=0, Careful=1, Give_Up=2

## Observations
o_m0[5,1,type=categorical]              ### Retrieval outcome: matches s_f1 with error states
o_m1[4,1,type=categorical]              ### Metacognitive feelings: Fluent=0, Effortful=1, Blocked=2, Uncertain=3

## Actions
u_c0[4,1,type=categorical]              ### Cognitive action: Recall=0, Search=1, Give_Up=2, Ask_Help=3
u_c1[3,1,type=categorical]              ### Metacognitive control: Continue=0, Change_Strategy=1, Stop=2

## InitialParameterization
# Metacognitive sensitivity
confidence_calibration = 0.75          ### How well confidence matches accuracy
metamemory_accuracy = 0.65             ### Accuracy of feeling-of-knowing

# Control parameters
strategy_switching_threshold = 0.3     ### Confidence threshold for strategy change
effort_cost = 0.1                      ### Cost of effortful processing
```

## Implementation Patterns

### Hierarchical Cognitive Architectures

Many cognitive phenomena involve multiple levels of processing:

```gnn
## ModelName
HierarchicalCognitionTemplate

## StateSpaceBlock
# High-level cognitive control
s_high_f0[4,1,type=categorical]         ### Executive control: Focus=0, Switch=1, Inhibit=2, Update=3
s_high_f1[3,1,type=categorical]         ### Goal state: Maintenance=0, Achievement=1, Revision=2

# Mid-level cognitive processes  
s_mid_f0[6,1,type=categorical]          ### Cognitive operation: Encode=0, Retrieve=1, Compare=2, Decide=3, Execute=4, Monitor=5
s_mid_f1[4,1,type=categorical]          ### Processing mode: Automatic=0, Controlled=1, Conflict=2, Error=3

# Low-level perceptual/motor
s_low_f0[8,1,type=categorical]          ### Perceptual features: Feature_A through Feature_H
s_low_f1[4,1,type=categorical]          ### Motor preparation: Prepared=0, Executing=1, Completed=2, Error=3

## Cross-Level Dependencies
s_high_f0 > s_mid_f0                    ### Executive control influences cognitive operations
s_mid_f1 > s_high_f1                    ### Processing outcomes influence goals
s_low_f1 > s_mid_f1                     ### Motor outcomes influence processing mode
```

### Temporal Dynamics in Cognition

Cognitive processes unfold over multiple timescales:

```gnn
## TimeSettings
Dynamic
DiscreteTime = t
FastTimeScale = 1-50ms          ### Neural/perceptual processes
MediumTimeScale = 100-1000ms    ### Cognitive operations  
SlowTimeScale = 1-10s           ### Strategic/metacognitive processes

## TemporalHierarchy
fast_updates: s_low_f0, s_low_f1
medium_updates: s_mid_f0, s_mid_f1  
slow_updates: s_high_f0, s_high_f1
```

### Learning and Adaptation Patterns

Cognitive systems learn at multiple levels:

```gnn
## LearningParameters
# Fast learning (within trial)
fast_learning_rate = 0.5
fast_forgetting_rate = 0.1

# Slow learning (across trials)  
slow_learning_rate = 0.01
slow_forgetting_rate = 0.001

# Meta-learning (strategy adaptation)
meta_learning_rate = 0.005
strategy_persistence = 0.95

## AdaptationMechanisms
prediction_error_threshold = 0.3       ### Trigger for adaptation
uncertainty_threshold = 0.7            ### Trigger for exploration
confidence_threshold = 0.8             ### Trigger for strategy change
```

## Validation and Testing

### Behavioral Validation

Cognitive models should reproduce key behavioral phenomena:

```python
def validate_cognitive_model(model_results, behavioral_data):
    """
    Validate cognitive model against behavioral benchmarks
    """
    validations = {
        'accuracy': validate_accuracy(model_results.accuracy, behavioral_data.accuracy),
        'reaction_time': validate_rt_distribution(model_results.rt, behavioral_data.rt),
        'learning_curve': validate_learning(model_results.learning, behavioral_data.learning),
        'individual_differences': validate_variability(model_results.subjects, behavioral_data.subjects)
    }
    
    return all(validations.values())

def validate_accuracy(model_acc, human_acc, tolerance=0.05):
    """Check if model accuracy matches human performance within tolerance"""
    return abs(model_acc - human_acc) < tolerance

def validate_rt_distribution(model_rt, human_rt):
    """Check if reaction time distributions match"""
    from scipy import stats
    statistic, p_value = stats.ks_2samp(model_rt, human_rt)
    return p_value > 0.05  # Non-significant difference

def validate_learning(model_learning, human_learning):
    """Check if learning curves match"""
    correlation = np.corrcoef(model_learning, human_learning)[0,1]
    return correlation > 0.8  # Strong correlation
```

### Neural Validation

Advanced models can be validated against neural data:

```python
def validate_neural_correspondence(model_states, neural_data):
    """
    Validate model states against neural recordings
    """
    validations = {
        'representational_similarity': validate_rsa(model_states, neural_data),
        'temporal_dynamics': validate_temporal_correlation(model_states, neural_data),
        'information_content': validate_information_similarity(model_states, neural_data)
    }
    
    return validations

def validate_rsa(model_repr, neural_repr):
    """Representational Similarity Analysis"""
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    
    model_rdm = squareform(pdist(model_repr, 'correlation'))
    neural_rdm = squareform(pdist(neural_repr, 'correlation'))
    
    correlation, p_value = spearmanr(model_rdm.flatten(), neural_rdm.flatten())
    return correlation > 0.3 and p_value < 0.05
```

## Research Applications

### Clinical Applications

Cognitive models can inform understanding of clinical conditions:

#### Attention Deficit Models
```gnn
## ModelName
ADHDAttentionModel

## ModelAnnotation
Models ADHD as reduced precision in attention control systems.
Captures distractibility and impulsivity through altered precision parameters.

## ModifiedParameters
attention_precision = 0.5              ### Reduced from normal 2.0
inhibitory_control = 0.3               ### Reduced from normal 1.0  
working_memory_capacity = 2            ### Reduced from normal 4
```

#### Depression Models
```gnn
## ModelName
DepressionModel

## ModelAnnotation
Models depression as altered prior beliefs about reward and agency.
Captures anhedonia and learned helplessness through biased expectations.

## ModifiedParameters
reward_expectations = [-0.5, 0.0, 0.5] ### Shifted negative from [0.0, 1.0, 2.0]
agency_beliefs = 0.3                   ### Reduced from normal 0.8
rumination_probability = 0.7           ### Increased from normal 0.1
```

### Educational Applications

Models can inform learning and instruction:

#### Skill Acquisition Model
```gnn
## ModelName
SkillAcquisitionModel

## ModelAnnotation
Models how complex skills are acquired through practice and feedback.
Captures automatization and expertise development.

## StateSpaceBlock
s_f0[5,1,type=categorical]              ### Skill level: Novice=0, Advanced_Beginner=1, Competent=2, Proficient=3, Expert=4
s_f1[3,1,type=categorical]              ### Processing mode: Controlled=0, Mixed=1, Automatic=2
s_f2[4,1,type=categorical]              ### Error detection: None=0, Self=1, External=2, Predictive=3

## LearningParameters
practice_effect = 0.02                 ### Skill improvement per practice trial
feedback_effectiveness = 0.1           ### Learning rate from feedback
automatization_threshold = 0.8         ### Threshold for automatic processing
```

### Computational Psychiatry

Models can provide quantitative frameworks for understanding mental health:

#### Predictive Processing Disorders
```gnn
## ModelName
PredictiveProcessingDisorderModel

## ModelAnnotation
Models various psychiatric conditions as alterations in predictive processing.
Captures symptom clusters through modified precision and prior parameters.

## DisorderParameters
# Autism: Enhanced precision, reduced flexibility
autism_sensory_precision = 3.0          ### Increased from normal 1.0
autism_prior_flexibility = 0.2          ### Reduced from normal 0.8

# Schizophrenia: Altered precision at different levels
schizophrenia_high_precision = 0.3      ### Reduced high-level precision
schizophrenia_low_precision = 2.5       ### Enhanced low-level precision

# Anxiety: Increased threat detection
anxiety_threat_prior = 0.7              ### Increased from normal 0.1  
anxiety_uncertainty_intolerance = 2.0   ### Increased from normal 1.0
```

## Contributing to Cognitive Phenomena Documentation

### Adding New Phenomena

To add documentation for a new cognitive phenomenon:

1. **Create model specification**: Write GNN model capturing the phenomenon
2. **Provide implementation**: Include working PyMDP/RxInfer code
3. **Validation data**: Include behavioral/neural benchmarks
4. **Usage examples**: Provide clear examples and tutorials
5. **References**: Link to relevant cognitive science literature

### Documentation Standards

Follow these standards for cognitive phenomena documentation:

- **Clear descriptions**: Explain the cognitive phenomenon in accessible terms
- **Model justification**: Explain why the GNN model captures the phenomenon
- **Parameter interpretation**: Explain what each parameter represents cognitively
- **Validation criteria**: Specify how to validate the model
- **Clinical relevance**: Discuss implications for understanding disorders
- **Future directions**: Suggest extensions and improvements

## References

### Core Active Inference Literature
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science
- Hohwy, J. (2013). The predictive mind: Cognitive science meets philosophy of mind

### Computational Cognitive Science
- Anderson, J. R. (2007). How can the human mind occur in the physical universe?
- Griffiths, T. L., et al. (2010). Probabilistic models of cognition: Exploring representations and inductive biases
- Tenenbaum, J. B., et al. (2011). How to grow a mind: Statistics, structure, and abstraction

### Clinical Applications
- Adams, R. A., et al. (2013). The computational anatomy of psychosis
- Fletcher, P. C., & Frith, C. D. (2009). Perceiving is believing: a Bayesian approach to explaining the positive symptoms of schizophrenia
- Barrett, L. F., & Simmons, W. K. (2015). Interoceptive predictions in the brain 