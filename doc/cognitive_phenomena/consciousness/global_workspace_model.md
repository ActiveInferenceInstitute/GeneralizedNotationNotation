# GNNVersionAndFlags

GNN Version: 1.0
Processing Flags: ParseMath=True, ValidateTypes=True, GenerateCode=True, CreateDiagrams=True

# ModelName

Global Workspace Consciousness Model

# ModelAnnotation

This model implements consciousness through the Global Workspace Theory in Active Inference. Consciousness emerges from global accessibility of information, where local processors compete to broadcast their contents to a shared workspace that enables coherent, unified conscious experience.

Key features:
- Global workspace with limited capacity and competitive dynamics
- Local specialized processors (visual, auditory, semantic, motor)
- Attention-mediated conscious access and broadcasting
- Self-model and metacognitive awareness
- Hierarchical integration across conscious and unconscious levels
- Phenomenal qualities emerging from precision-weighted prediction

The model captures how consciousness arises from the integration of distributed processing through a global workspace that makes information globally accessible for flexible control and report.

# StateSpaceBlock

### Hidden States
s_workspace[20,1,type=categorical] ### Global workspace contents with limited capacity
s_visual[10,1,type=categorical] ### Visual processing: local visual features and objects
s_auditory[8,1,type=categorical] ### Auditory processing: local auditory features and patterns
s_semantic[12,1,type=categorical] ### Semantic processing: conceptual and linguistic content
s_motor[6,1,type=categorical] ### Motor processing: action planning and preparation
s_access[4,1,type=categorical] ### Conscious access level: {0=unconscious, 1=preconscious, 2=conscious, 3=reflective}
s_attention[5,1,type=categorical] ### Attention focus: {0=visual, 1=auditory, 2=semantic, 3=motor, 4=internal}
s_self[8,1,type=categorical] ### Self-model: body, mind, emotions, thoughts, intentions, beliefs, identity, agency

### Observations
o_visual[10,1,type=categorical] ### Visual sensory input from environment
o_auditory[8,1,type=categorical] ### Auditory sensory input from environment
o_interoceptive[6,1,type=categorical] ### Internal bodily signals and sensations
o_social[4,1,type=categorical] ### Social cues and interpersonal signals

### Actions
u_broadcast[4,1,type=categorical] ### Broadcasting control: {0=inhibit, 1=local, 2=global, 3=amplify}
u_attention[5,1,type=categorical] ### Attention deployment: matches s_attention states
u_report[3,1,type=categorical] ### Conscious report: {0=silent, 1=verbal, 2=behavioral}

# Connections

### Local to Global Broadcasting
s_visual > s_workspace ### Visual processors broadcast to global workspace
s_auditory > s_workspace ### Auditory processors broadcast to global workspace
s_semantic > s_workspace ### Semantic processors broadcast to global workspace
s_motor > s_workspace ### Motor processors broadcast to global workspace

### Attention-Mediated Access
s_attention > s_access ### Attention determines conscious access level
s_attention > s_visual ### Attention modulates visual processing
s_attention > s_auditory ### Attention modulates auditory processing
s_attention > s_semantic ### Attention modulates semantic processing

### Global to Local Feedback
s_workspace > s_visual ### Global workspace influences local visual processing
s_workspace > s_auditory ### Global workspace influences local auditory processing
s_workspace > s_semantic ### Global workspace influences local semantic processing
s_workspace > s_motor ### Global workspace influences motor planning

### Self-Model Integration
s_workspace > s_self ### Workspace contents update self-model
s_self > s_attention ### Self-model influences attention allocation
s_self > s_access ### Self-awareness affects conscious access

### Sensory Processing
o_visual > s_visual ### Visual input drives visual processing
o_auditory > s_auditory ### Auditory input drives auditory processing
o_interoceptive > s_self ### Interoceptive signals inform self-model
o_social > s_semantic ### Social cues drive semantic processing

### Action and Control
s_workspace > u_broadcast ### Workspace state controls broadcasting
s_access > u_report ### Conscious access enables reporting
s_attention > u_attention ### Attention state drives attention deployment

# InitialParameterization

### Observation Model (A matrices)
A_visual = eye(10) * 0.85 + 0.015 ### Visual observation with noise
A_auditory = eye(8) * 0.85 + 0.019 ### Auditory observation with noise
A_interoceptive = eye(6) * 0.7 + 0.05 ### Noisy interoceptive signals
A_social = eye(4) * 0.8 + 0.067 ### Social signal observation

### Transition Dynamics (B matrices)
B_workspace = eye(20) * 0.6 + 0.02 ### Workspace with fast dynamics and competition
B_visual = eye(10) * 0.8 + 0.02 ### Visual processing with moderate persistence
B_auditory = eye(8) * 0.8 + 0.025 ### Auditory processing with moderate persistence
B_semantic = eye(12) * 0.85 + 0.013 ### Semantic processing with higher persistence
B_motor = eye(6) * 0.75 + 0.042 ### Motor processing with moderate persistence
B_access = [[0.7, 0.2, 0.08, 0.02], [0.15, 0.65, 0.15, 0.05], [0.05, 0.2, 0.65, 0.1], [0.02, 0.05, 0.23, 0.7]] ### Access level transitions
B_attention = [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.1, 0.6, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]] ### Attention switching
B_self = eye(8) * 0.9 + 0.014 ### Self-model with high persistence

### Preferences (C matrices)
C_visual = zeros(10) ### No specific visual preferences
C_auditory = zeros(8) ### No specific auditory preferences
C_interoceptive = [1.0, 0.8, 0.5, 0.0, -0.5, -1.0] ### Preference for positive interoceptive states
C_social = [1.0, 0.5, 0.0, -0.5] ### Mild preference for positive social signals

### Initial Beliefs (D vectors)
D_workspace = normalize(ones(20) * 0.05) ### Initially sparse workspace
D_visual = normalize(ones(10)) ### Uniform initial visual states
D_auditory = normalize(ones(8)) ### Uniform initial auditory states
D_semantic = normalize(ones(12)) ### Uniform initial semantic states
D_motor = normalize(ones(6)) ### Uniform initial motor states
D_access = [0.5, 0.3, 0.15, 0.05] ### Initially mostly unconscious
D_attention = [0.25, 0.25, 0.25, 0.15, 0.1] ### Distributed initial attention
D_self = normalize(ones(8)) ### Uniform initial self-model

### Precision Parameters
γ_workspace = 1.8 ### Moderate workspace precision for competitive dynamics
γ_visual = 2.0 ### High visual precision
γ_auditory = 1.8 ### High auditory precision
γ_semantic = 1.5 ### Moderate semantic precision for flexibility
γ_motor = 1.6 ### Moderate motor precision
γ_access = 2.5 ### High precision for conscious access
γ_attention = 3.0 ### Very high attention precision
γ_self = 1.4 ### Moderate self-model precision
α = 16.0 ### Action precision for decisive broadcasting and reporting

# Equations

### Global Workspace Competition
\\[ P(s^{workspace}_t|s^{local}_t) = \\text{Cat}(\\text{softmax}(\\gamma_{workspace} \\cdot \\sum_{i} w_i \\cdot \\text{broadcast}_i(s^{local}_{i,t}))) \\]

### Conscious Access Threshold
\\[ P(s^{access}_t|s^{workspace}_t, s^{attention}_t) = \\text{Cat}(\\text{softmax}(\\gamma_{access} \\cdot (\\alpha \\cdot ||s^{workspace}_t|| + \\beta \\cdot s^{attention}_t))) \\]

### Broadcasting Threshold
\\[ \\text{broadcast}_i(s_t) = \\begin{cases} 
s_t & \\text{if } ||s_t|| > \\theta_{broadcast} \\\\
0 & \\text{otherwise}
\\end{cases} \\]

### Attention-Modulated Processing
\\[ P(s^{local}_{t+1}|s^{local}_t, s^{attention}_t) = \\text{Cat}(\\text{softmax}(\\gamma_{local} \\cdot \\text{attention\\_weight}(s^{attention}_t) \\odot s^{local}_t)) \\]

### Self-Model Update
\\[ s^{self}_{t+1} = s^{self}_t + \\eta_{self} \\cdot (\\text{integrated\\_info}(s^{workspace}_t) - s^{self}_t) \\]

### Conscious Report Generation
\\[ P(u^{report}_t|s^{access}_t, s^{workspace}_t) = \\text{Cat}(\\text{softmax}(\\alpha \\cdot \\mathbf{W}_{report} \\cdot [s^{access}_t; s^{workspace}_t])) \\]

### Phenomenal Quality (Qualia) Generation
\\[ \\text{qualia}_t = \\gamma_{phenom} \\cdot s^{attention}_t \\odot \\epsilon^{sensory}_t \\odot ||s^{workspace}_t|| \\]

where \\( \\epsilon^{sensory}_t \\) are sensory prediction errors and \\( \\odot \\) is element-wise multiplication.

### Information Integration Measure
\\[ \\Phi(s^{workspace}_t) = \\sum_{i,j} \\text{MI}(s^{workspace}_{i,t}, s^{workspace}_{j,t}) - \\sum_k \\text{MI}(\\text{partition}_k) \\]

where \\( \\Phi \\) measures integrated information in the workspace.

# Time

Dynamic: True
DiscreteTime: True
ModelTimeHorizon: 100

# ActInfOntologyAnnotation

- Global Workspace (s_workspace): Maps to "Conscious Access" and "Information Broadcasting" in Active Inference Ontology
- Local Processors (s_visual, s_auditory, etc.): Correspond to "Specialized Processing" and "Modular Computation" concepts
- Conscious Access (s_access): Maps to "Awareness Level" and "Reportability" mechanisms
- Attention (s_attention): Relates to "Selective Processing" and "Precision Allocation" processes
- Self-Model (s_self): Corresponds to "Self-Representation" and "Metacognitive Awareness" concepts
- Broadcasting: Implemented through global information sharing and competitive workspace dynamics
- Consciousness: Emerges from integrated information processing and global accessibility

# Footer

Model Type: Consciousness and Global Workspace Theory
Domain: Cognitive Neuroscience, Consciousness Studies, Computational Psychology
Compatible Backends: PyMDP, RxInfer.jl
Complexity: Very High (8 hidden states, 4 observations, 3 actions)

# Signature

Generated by: Active Inference Institute - GNN Consciousness Research Group
Date: 2025-01-27
Version: 1.0
Contact: info@activeinference.org
License: Creative Commons Attribution 4.0 International 