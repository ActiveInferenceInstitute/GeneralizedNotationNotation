# GNNVersionAndFlags

GNN Version: 1.0
Processing Flags: ParseMath=True, ValidateTypes=True, GenerateCode=True, CreateDiagrams=True

# ModelName

Executive Control: Task Switching and Cognitive Flexibility

# ModelAnnotation

This model implements executive control mechanisms in Active Inference, focusing on task switching and cognitive flexibility. The model captures how the brain hierarchically controls lower-level cognitive processes by modulating precision weights and selecting appropriate behavioral policies based on goals and context.

Key features:
- Task context representation with switching dynamics
- Working memory maintenance with capacity limitations
- Inhibitory control through precision modulation
- Policy selection based on expected free energy
- Cognitive flexibility through rapid context reconfiguration

The model demonstrates executive control in a dual-task paradigm where an agent must flexibly switch between two tasks (spatial and verbal) based on contextual cues, managing interference and maintaining goal-relevant information.

# StateSpaceBlock

### Hidden States
s_task[2,1,type=categorical] ### Current task context: {0=spatial_task, 1=verbal_task}
s_wm_spatial[4,1,type=categorical] ### Working memory for spatial information: 4 locations
s_wm_verbal[4,1,type=categorical] ### Working memory for verbal information: 4 items
s_attention[3,1,type=categorical] ### Attention state: {0=diffuse, 1=focused_spatial, 2=focused_verbal}
s_conflict[2,1,type=categorical] ### Conflict detection: {0=no_conflict, 1=conflict_detected}

### Observations
o_cue[2,1,type=categorical] ### Task cue: {0=spatial_cue, 1=verbal_cue}
o_stimulus[8,1,type=categorical] ### Stimulus input: 4 spatial + 4 verbal positions
o_performance[3,1,type=categorical] ### Performance feedback: {0=correct, 1=error, 2=timeout}

### Actions
u_response[8,1,type=categorical] ### Response selection: spatial or verbal responses
u_switch[2,1,type=categorical] ### Task switch control: {0=maintain, 1=switch}

# Connections

### Task Context Dependencies
s_task > o_cue ### Task context influences cue processing
s_task > s_attention ### Task context biases attention allocation
s_task > u_switch ### Task context influences switch decisions

### Working Memory Dynamics
s_wm_spatial > s_attention ### Spatial WM content influences attention
s_wm_verbal > s_attention ### Verbal WM content influences attention
s_attention > s_wm_spatial ### Attention modulates spatial WM maintenance
s_attention > s_wm_verbal ### Attention modulates verbal WM maintenance

### Conflict Processing
s_task > s_conflict ### Task demands generate conflict
s_wm_spatial > s_conflict ### Spatial WM interference
s_wm_verbal > s_conflict ### Verbal WM interference
s_conflict > s_attention ### Conflict triggers attentional control

### Stimulus-Response Mapping
o_stimulus > s_wm_spatial ### Spatial stimuli update spatial WM
o_stimulus > s_wm_verbal ### Verbal stimuli update verbal WM
s_wm_spatial > u_response ### Spatial WM drives responses
s_wm_verbal > u_response ### Verbal WM drives responses

### Performance Monitoring
u_response > o_performance ### Actions generate performance outcomes
o_performance > s_conflict ### Performance errors trigger conflict detection

# InitialParameterization

### Observation Model (A matrices)
A_cue = [[0.9, 0.1], [0.1, 0.9]] ### Task cue likelihood given task context
A_stimulus = eye(8) ### Direct stimulus observation
A_performance = [[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.05, 0.15, 0.8]] ### Performance likelihood

### Transition Dynamics (B matrices)
B_task = [[0.95, 0.05], [0.05, 0.95]] ### Task context persistence with switching
B_wm_spatial = eye(4) * 0.9 + 0.025 ### Spatial WM decay with maintenance
B_wm_verbal = eye(4) * 0.9 + 0.025 ### Verbal WM decay with maintenance
B_attention = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] ### Attention state transitions
B_conflict = [[0.85, 0.15], [0.3, 0.7]] ### Conflict detection dynamics

### Preferences (C matrices)
C_performance = [2.0, -2.0, -1.0] ### Strong preference for correct performance
C_cue = [0.0, 0.0] ### No preference for specific cues
C_stimulus = zeros(8) ### No stimulus preferences

### Initial Beliefs (D vectors)
D_task = [0.5, 0.5] ### Uniform initial task context
D_wm_spatial = [0.7, 0.1, 0.1, 0.1] ### Empty spatial WM initially
D_wm_verbal = [0.7, 0.1, 0.1, 0.1] ### Empty verbal WM initially
D_attention = [0.6, 0.2, 0.2] ### Initially diffuse attention
D_conflict = [0.8, 0.2] ### Low initial conflict

### Precision Parameters
γ_task = 2.0 ### Task context precision
γ_wm = 1.5 ### Working memory precision
γ_attention = 3.0 ### Attention precision (high for executive control)
γ_conflict = 2.5 ### Conflict detection precision
α = 16.0 ### Action precision (policy commitment)

# Equations

### Executive Control via Precision Modulation
\\[ \\text{Executive Control} = \\arg\\max_{\\pi} \\mathbb{E}_{Q(s|\\pi)}[\\ln P(o|s) + \\ln P(s|\\pi)] - \\gamma \\cdot D_{KL}[Q(\\pi)||P(\\pi)] \\]

### Working Memory Maintenance
\\[ P(s^{wm}_{t+1}|s^{wm}_t, s^{att}_t) = \\text{Cat}(\\sigma(\\gamma_{att} \\cdot s^{att}_t) \\odot s^{wm}_t) \\]

### Conflict Detection
\\[ P(s^{conflict}_t|s^{task}_t, s^{wm}_t) = \\text{Cat}(\\sigma(\\gamma_{conflict} \\cdot |s^{spatial}_t - s^{verbal}_t|)) \\]

### Task Switching Cost
\\[ \\text{Switch Cost} = \\gamma_{switch} \\cdot |u^{switch}_{t-1} - u^{switch}_t| \\]

### Attention Allocation
\\[ P(s^{att}_t|s^{task}_t, s^{conflict}_t) = \\text{Cat}(\\text{softmax}(\\gamma_{att} \\cdot (\\mathbf{w}_{task} \\cdot s^{task}_t + \\mathbf{w}_{conflict} \\cdot s^{conflict}_t))) \\]

### Policy Prior (Cognitive Control)
\\[ P(\\pi) = \\text{Cat}(\\text{softmax}(\\gamma_{control} \\cdot \\mathbf{w}_{goal} \\cdot G(\\pi))) \\]

where \\( G(\\pi) \\) is the expected free energy under policy \\( \\pi \\).

### Performance Optimization
\\[ Q^*(s,u) = \\arg\\min_{Q} \\mathbb{E}_Q[\\text{Free Energy}] + \\lambda \\cdot \\text{Control Cost} \\]

# Time

Dynamic: True
DiscreteTime: True
ModelTimeHorizon: 20

# ActInfOntologyAnnotation

- Task Context (s_task): Maps to "Cognitive Context" and "Task Representation" in Active Inference Ontology
- Working Memory (s_wm_*): Corresponds to "Active Maintenance" and "Information Buffer" concepts
- Attention (s_attention): Maps to "Attentional Control" and "Precision Allocation" mechanisms
- Conflict Detection (s_conflict): Relates to "Conflict Monitoring" and "Performance Evaluation" processes
- Executive Control: Implemented through hierarchical precision optimization and policy selection
- Cognitive Flexibility: Achieved through rapid reconfiguration of precision weights and context switching

# Footer

Model Type: Executive Control and Cognitive Flexibility
Domain: Cognitive Neuroscience, Computational Psychiatry
Compatible Backends: PyMDP, RxInfer.jl
Complexity: Medium-High (5 hidden states, 3 observations, 2 actions)

# Signature

Generated by: Active Inference Institute - GNN Executive Control Working Group
Date: 2025-01-27
Version: 1.0
Contact: info@activeinference.org
License: Creative Commons Attribution 4.0 International 