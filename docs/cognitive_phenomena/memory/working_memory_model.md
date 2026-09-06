# Working Memory Model

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: working_memory, capacity_limitation, maintenance_dynamics

## ModelName
WorkingMemoryModel

## ModelAnnotation
Models working memory as active maintenance of information through recurrent precision dynamics.
Captures capacity limitations, interference effects, and attention-mediated control of memory maintenance.
Working memory emerges from sustained high-precision representations that resist decay.

## StateSpaceBlock
# Memory slot states
s_f0[4,1,type=categorical]               ### Memory slots: Slot_1=0, Slot_2=1, Slot_3=2, Slot_4=3
s_f1[8,1,type=categorical]               ### Item identity: Item_A=0, Item_B=1, Item_C=2, Item_D=3, Item_E=4, Item_F=5, Item_G=6, Empty=7
s_f2[4,1,type=categorical]               ### Maintenance strength: Strong=0, Medium=1, Weak=2, Decayed=3

# Memory control states
s_f3[3,1,type=categorical]               ### Maintenance strategy: Active=0, Passive=1, Rehearsal=2
s_f4[5,1,type=categorical]               ### Attention allocation: Slot_1=0, Slot_2=1, Slot_3=2, Slot_4=3, Distributed=4
s_f5[4,1,type=categorical]               ### Memory load: Load_1=0, Load_2=1, Load_3=2, Load_4=3

# Interference and competition
s_f6[3,1,type=categorical]               ### Interference level: Low=0, Medium=1, High=2
s_f7[4,1,type=categorical]               ### Competition state: No_Competition=0, Weak=1, Moderate=2, Strong=3

# Cognitive control
s_f8[3,1,type=categorical]               ### Executive control: Update=0, Maintain=1, Clear=2
s_f9[2,1,type=categorical]               ### Distractor resistance: High=0, Low=1

## Observations
o_m0[8,1,type=categorical]               ### Item presentation: matches s_f1 item identities
o_m1[3,1,type=categorical]               ### Memory cue: Maintain=0, Update=1, Retrieve=2
o_m2[4,1,type=categorical]               ### Distractor presence: None=0, Visual=1, Auditory=2, Cognitive=3
o_m3[2,1,type=categorical]               ### Memory probe: Match=0, NoMatch=1

## Actions
u_c0[8,1,type=categorical]               ### Encoding action: Encode_A=0, Encode_B=1, ... Encode_G=6, NoAction=7
u_c1[4,1,type=categorical]               ### Maintenance action: Refresh_Slot1=0, Refresh_Slot2=1, Refresh_Slot3=2, Refresh_All=3
u_c2[3,1,type=categorical]               ### Attention control: Focus=0, Switch=1, Divide=2
u_c3[2,1,type=categorical]               ### Response action: Same=0, Different=1

## Connections
# Memory encoding and storage
u_c0, s_f0 > s_f1                        ### Encoding actions determine item-slot assignment
s_f1, s_f5 > s_f2                        ### Memory load affects maintenance strength

# Maintenance dynamics
u_c1, s_f4 > s_f2                        ### Maintenance actions and attention affect strength
s_f3 > s_f2                              ### Maintenance strategy affects strength

# Capacity and interference
s_f5 > s_f6                              ### Memory load determines interference level
s_f6, s_f7 > s_f2                        ### Interference and competition reduce maintenance

# Attention and control
s_f8 > s_f4                              ### Executive control determines attention allocation
o_m2, s_f9 > s_f6                        ### Distractors and resistance determine interference

# Memory retrieval
s_f1, s_f2 > o_m3                        ### Item identity and strength determine retrieval success

## InitialParameterization
# Capacity parameters
max_capacity = 4                         ### Miller's 7±2, set to 4 for this model
capacity_cost_exponent = 2.0             ### Quadratic cost increase with load

# Maintenance parameters
maintenance_base_cost = 0.1              ### Base metabolic cost per item
maintenance_decay_rate = 0.05            ### Decay rate without active maintenance
rehearsal_effectiveness = 0.8            ### Effectiveness of rehearsal maintenance

# Interference parameters
inter_item_interference = 0.3            ### Interference between similar items
distractor_interference = [0.0, 0.2, 0.4, 0.6]  ### Interference from different distractor types
capacity_interference_scaling = 1.5      ### Interference increases with memory load

# Attention parameters
attention_precision_boost = 2.0          ### Precision increase for attended items
attention_switching_cost = 0.2           ### Cost of switching attention between slots
divided_attention_penalty = 0.4          ### Reduced effectiveness when dividing attention

# Learning and adaptation
working_memory_learning_rate = 0.02      ### Rate of working memory strategy learning
precision_adaptation_rate = 0.1          ### Rate of precision optimization

# A matrices (maintenance-dependent observation likelihoods)
A_m0_strong_maintenance = [
    [0.95, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005],  # High precision maintenance
    [0.02, 0.95, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005],
    [0.01, 0.01, 0.95, 0.02, 0.005, 0.005, 0.005, 0.005],
    [0.01, 0.01, 0.02, 0.95, 0.005, 0.005, 0.005, 0.005],
    [0.005, 0.005, 0.005, 0.005, 0.95, 0.02, 0.01, 0.01],
    [0.005, 0.005, 0.005, 0.005, 0.02, 0.95, 0.01, 0.01],
    [0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.95, 0.02],
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.86]       # Empty slot
]

A_m0_weak_maintenance = [
    [0.70, 0.08, 0.06, 0.06, 0.025, 0.025, 0.025, 0.025],  # Degraded maintenance
    [0.08, 0.70, 0.06, 0.06, 0.025, 0.025, 0.025, 0.025],
    [0.06, 0.06, 0.70, 0.08, 0.025, 0.025, 0.025, 0.025],
    [0.06, 0.06, 0.08, 0.70, 0.025, 0.025, 0.025, 0.025],
    [0.025, 0.025, 0.025, 0.025, 0.70, 0.08, 0.06, 0.06],
    [0.025, 0.025, 0.025, 0.025, 0.08, 0.70, 0.06, 0.06],
    [0.025, 0.025, 0.025, 0.025, 0.06, 0.06, 0.70, 0.08],
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30]       # Increased forgetting
]

# B matrices (maintenance dynamics)
B_maintenance_strength = [
    [0.9, 0.08, 0.015, 0.005],           # Strong maintenance persists
    [0.2, 0.6, 0.15, 0.05],              # Medium maintenance decays
    [0.1, 0.3, 0.4, 0.2],                # Weak maintenance decays faster
    [0.0, 0.1, 0.2, 0.7]                 # Decayed state is stable
]

# C matrices (preferences for successful memory performance)
C_m3 = [2.0, -2.0]                      ### Strong preference for correct memory responses

# D matrices (initial state priors)
D_f1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]  ### Prior: slots likely to start empty
D_f2 = [0.2, 0.3, 0.3, 0.2]             ### Prior: medium maintenance strength initially
D_f5 = [0.4, 0.3, 0.2, 0.1]             ### Prior: low memory load initially

## Equations
# Working memory capacity
effective_capacity(t) = max_capacity * (1 - interference_penalty(t))

# Maintenance cost function
maintenance_cost(t) = sum(maintenance_base_cost * (memory_load(t)^capacity_cost_exponent))

# Decay function with attention modulation
decay_rate(slot, t) = maintenance_decay_rate * (1 - attention_weight(slot, t)) * (1 + interference_level(t))

# Attention-weighted precision
precision(slot, t) = base_precision + attention_weight(slot, t) * attention_precision_boost

# Inter-item interference
interference(slot_i, slot_j, t) = inter_item_interference * similarity(item_i, item_j) * activation(slot_j, t)

# Capacity-limited attention allocation
attention_allocation(t) = softmax(attention_demands(t) / temperature(memory_load(t)))

# Memory retrieval probability
retrieval_probability(slot, t) = sigmoid(maintenance_strength(slot, t) * precision(slot, t) - interference(slot, t))

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 50

## ActInfOntologyAnnotation
WorkingMemory: active_maintenance
CapacityLimitation: resource_constraint
AttentionControl: precision_allocation
MemoryMaintenance: decay_resistance

## Footer
This model demonstrates working memory through Active Inference mechanisms.
Memory maintenance emerges from sustained high-precision representations.
Model captures capacity limitations, interference, and attention-control processes.

## Signature
Model: WorkingMemoryModel
Framework: Active Inference
Domain: Cognitive Phenomena - Memory
Created: 2025
Validated: N-back task, change detection, digit span paradigms 