# GNN Quick Start Tutorial

**Version**: v1.1.0  
**Last Updated**: February 9, 2026  
**Status**: ✅ Production Ready  
**Test Count**: 1,127 Tests Passing  

**Goal**: Create and run your first GNN model in 15 minutes, no prior Active Inference knowledge required.

## 🎯 What You'll Build

A simple navigation agent that learns to find a goal location in a 2x2 grid world.

```
[Start] [    ]
[    ] [Goal]
```

## ⚡ Prerequisites

- Basic programming knowledge (any language)
- Python 3.8+ installed
- 15 minutes of focused time

> 💡 **No Active Inference background needed!** This tutorial explains concepts as we go.

## 📥 Step 1: Setup (2 minutes)

### Install the GNN toolkit

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using UV (recommended)
uv sync

# Test the installation
python src/main.py --help
```

### Create your workspace

```bash
# Create a folder for your first model
mkdir my_first_gnn_model
cd my_first_gnn_model
```

## 🧱 Step 2: Understanding the Basics (3 minutes)

### What is Active Inference?

**Simple version**: A mathematical framework where agents:

1. Have **beliefs** about the world (hidden states)
2. Make **observations** about what they can see
3. Take **actions** to achieve their **preferences**

### What does a GNN model specify?

1. **States**: What the agent needs to track (e.g., position)
2. **Observations**: What the agent can see (e.g., visual input)
3. **Actions**: What the agent can do (e.g., move)
4. **Preferences**: What the agent wants (e.g., reach goal)

### Our Grid World Model

- **States**: 4 positions (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
- **Observations**: Current position (can see where it is)
- **Actions**: 4 movements (Up, Down, Left, Right)
- **Preference**: Be at the goal (Bottom-Right)

## 📝 Step 3: Write Your First GNN Model (5 minutes)

Create a file called `grid_agent.gnn`:

```gnn
## GNNVersionAndFlags
GNN v1

## ModelName
Simple Grid Navigation Agent v1.0

## ModelAnnotation
A 2x2 grid navigation agent that learns to reach a goal.
The agent can observe its current position and can move in 4 directions.
Goal is to reach the bottom-right corner.

## StateSpaceBlock
# Hidden State Factor: Agent's position
s_f0[4,1,type=int]   # Position (0:TopLeft, 1:TopRight, 2:BottomLeft, 3:BottomRight/Goal)

# Observation: What the agent sees (its current position)
o_m0[4,1,type=int]   # Observed position (same as true position)

# Control: Agent's actions
pi_c0[4,type=float]  # Policy over movement actions  
u_c0[1,type=int]     # Chosen action (0:Up, 1:Down, 2:Left, 3:Right)

# Model matrices
A_m0[4,4,type=float] # Likelihood: P(observation | position)
B_f0[4,4,4,type=float] # Transition: P(next_position | current_position, action)
C_m0[4,type=float]   # Preferences over observations
D_f0[4,type=float]   # Prior beliefs about starting position

# Expected Free Energy and time
G[1,type=float]      # Expected Free Energy for action selection
t[1,type=int]        # Time step

## Connections
# Prior influences initial state
(D_f0) -> (s_f0)

# Position determines what agent observes
(s_f0) -> (A_m0)
(A_m0) -> (o_m0)

# Position and action determine next position
(s_f0, u_c0) -> (B_f0)
(B_f0) -> s_f0_next

# Preferences and expected outcomes influence action selection
(C_m0, A_m0, B_f0, s_f0) > G
G > pi_c0
(pi_c0) -> u_c0

## InitialParameterization
# A_m0: Agent can perfectly observe its position (identity matrix)
A_m0={
  ((1.0, 0.0, 0.0, 0.0),   # If at TopLeft(0), observe TopLeft
   (0.0, 1.0, 0.0, 0.0),   # If at TopRight(1), observe TopRight  
   (0.0, 0.0, 1.0, 0.0),   # If at BottomLeft(2), observe BottomLeft
   (0.0, 0.0, 0.0, 1.0))   # If at BottomRight(3), observe BottomRight
}

# B_f0: Movement transitions [next_pos, current_pos, action]
# Actions: 0:Up, 1:Down, 2:Left, 3:Right
B_f0={
  # next_position = TopLeft(0)
  (((1.0, 0.0, 1.0, 0.0),   # From positions 0,1,2,3 with action Up(0)
    (1.0, 0.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Down(1)
    (1.0, 1.0, 1.0, 1.0),   # From positions 0,1,2,3 with action Left(2) 
    (0.0, 0.0, 0.0, 0.0))), # From positions 0,1,2,3 with action Right(3)
    
  # next_position = TopRight(1)  
  (((0.0, 1.0, 0.0, 1.0),   # From positions 0,1,2,3 with action Up(0)
    (0.0, 1.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Down(1)
    (0.0, 0.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Left(2)
    (1.0, 1.0, 1.0, 1.0))), # From positions 0,1,2,3 with action Right(3)
    
  # next_position = BottomLeft(2)
  (((0.0, 0.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Up(0)
    (0.0, 0.0, 1.0, 0.0),   # From positions 0,1,2,3 with action Down(1)
    (0.0, 0.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Left(2)
    (0.0, 0.0, 0.0, 0.0))), # From positions 0,1,2,3 with action Right(3)
    
  # next_position = BottomRight(3) - GOAL
  (((0.0, 0.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Up(0)
    (0.0, 0.0, 0.0, 1.0),   # From positions 0,1,2,3 with action Down(1)
    (0.0, 0.0, 0.0, 0.0),   # From positions 0,1,2,3 with action Left(2)
    (0.0, 0.0, 0.0, 0.0)))  # From positions 0,1,2,3 with action Right(3)
}

# C_m0: Preferences (higher values = more preferred)
C_m0={(-1.0, -1.0, -1.0, 2.0)}  # Strongly prefer goal position (BottomRight)

# D_f0: Start at TopLeft with certainty
D_f0={(1.0, 0.0, 0.0, 0.0)}

## Equations
# Standard Active Inference equations for policy selection:
# G(π) = E_q[ln q(o,s|π) - ln P(o,s|π) - ln C(o)]
# P(π) = softmax(-G(π))

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=5

## ActInfOntologyAnnotation
s_f0=HiddenStatePosition
o_m0=ObservationPosition  
pi_c0=PolicyMovement
u_c0=ActionMovement
A_m0=LikelihoodMatrixPosition
B_f0=TransitionMatrixMovement
C_m0=PreferenceVector
D_f0=PriorBelief
G=ExpectedFreeEnergy
t=TimeStep

## Footer
Simple Grid Navigation Agent v1.0

## Signature
Creator: GNN Tutorial
Date: 2024
Status: Tutorial Example
```

Save this as `grid_agent.gnn` in your `my_first_gnn_model` folder.

## ✅ Step 4: Validate Your Model (2 minutes)

Check if your model is correct:

```bash
# Run the GNN type checker (Step 5)
python src/5_type_checker.py --target-dir my_first_gnn_model/ --verbose

# If successful, you should see:
# ✅ grid_agent.gnn: Valid GNN model
# 📊 Resource estimation: [details]
```

**If you see errors**: Check the [Common Errors Guide](../troubleshooting/common_errors.md) or compare with the template above.

For more information on the type checker, see **[src/type_checker/AGENTS.md](../../src/type_checker/AGENTS.md)**.

## 🚀 Step 5: Generate Runnable Code (3 minutes)

Convert your GNN model to executable Python code:

```bash
# Generate PyMDP code (Steps 3, 11, 12)
python src/main.py --only-steps "3,11,12" --target-dir my_first_gnn_model/ --output-dir output/my_first_model/ --verbose

# This creates several outputs:
# - output/11_render_output/ (executable code for PyMDP, RxInfer, etc.)
# - output/8_visualization_output/ (model diagrams)
# - output/7_export_output/ (JSON, XML formats)
```

For more details on code generation, see:

- **[src/render/AGENTS.md](../../src/render/AGENTS.md)**: Code rendering module documentation  
- **[src/execute/AGENTS.md](../../src/execute/AGENTS.md)**: Execution module documentation

### Test the generated code

```bash
# Navigate to rendered output
cd output/11_render_output/

# Run the PyMDP simulation
python grid_agent_pymdp.py

# You should see the agent's behavior:
# Time 0: Position=TopLeft, Action=Right
# Time 1: Position=TopRight, Action=Down  
# Time 2: Position=BottomRight, Action=Stay (GOAL REACHED!)
```

## 🎉 Congratulations

You've just:

1. ✅ Written your first GNN model
2. ✅ Validated it with the type checker
3. ✅ Generated executable code
4. ✅ Run a working Active Inference agent

## 🔄 What Just Happened?

Your agent:

1. **Started** with belief it's at TopLeft
2. **Observed** its true position  
3. **Planned** actions to reach the goal (BottomRight)
4. **Selected** actions based on expected free energy minimization
5. **Learned** to navigate optimally

## 🎯 Next Steps

### Immediate Experiments (5 minutes each)

1. **Change the goal**: Modify `C_m0` to prefer TopRight instead
2. **Add uncertainty**: Make observations noisy by modifying `A_m0`
3. **Bigger world**: Extend to a 3x3 grid (requires updating all matrices)

### Deeper Learning

1. **Understand the math**: Read [Active Inference basics](about_gnn.md)
2. **Try examples**: Explore [more complex models](gnn_examples_doc.md)
3. **Different domains**: Navigation → Perception → Decision making
4. **Advanced features**: Multi-agent, learning, hierarchical models
5. **Pipeline architecture**: See **[src/AGENTS.md](../../src/AGENTS.md)** for complete module documentation
6. **Pipeline safety**: Read **[src/README.md](../../src/README.md)** for architecture patterns

### Build Your Own Model

1. **Start with the template**: Use [`templates/basic_gnn_template.md`](../templates/basic_gnn_template.md)
2. **Model your domain**: What states, observations, actions make sense?
3. **Get help**: Check [FAQ](../troubleshooting/faq.md) and [community discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
4. **Process with pipeline**: Use `src/main.py` to run complete workflow

## 🧠 Key Concepts You've Learned

| Concept | What It Does | In Our Example |
|---------|--------------|----------------|
| **Hidden States** (`s_f0`) | What the agent tracks internally | 4 grid positions |
| **Observations** (`o_m0`) | What the agent can perceive | Current position |
| **Actions** (`u_c0`) | What the agent can do | 4 movement directions |
| **Likelihood** (`A_m0`) | How states relate to observations | Perfect position sensing |
| **Transitions** (`B_f0`) | How actions change states | Movement rules |
| **Preferences** (`C_m0`) | What the agent wants | Reach bottom-right |
| **Expected Free Energy** (`G`) | How the agent chooses actions | Minimize surprise, maximize reward |

## 🛠️ Common Issues and Quick Fixes

### "Parser Error: Invalid syntax"

- Check section headers: Use `## StateSpaceBlock` not `## StateSpace`
- Check variable names: Use `s_f0` not `s f0` (underscores, not spaces)

### "Dimension mismatch"

- Ensure matrix sizes match variable definitions
- `A_m0[4,4]` means 4 observations × 4 states

### "Probabilities don't sum to 1"

- Each column in B_f0 must sum to 1.0
- Each column in A_m0 must sum to 1.0

### "My agent doesn't reach the goal"

- Check `C_m0`: Higher values should be at preferred states
- Check `B_f0`: Ensure movement logic is correct
- Try `ModelTimeHorizon=10` for longer planning

## 📚 Resources

- **Pipeline Documentation**:
  - [src/AGENTS.md](../../src/AGENTS.md): Complete module registry
  - [src/README.md](../../src/README.md): Pipeline architecture and safety
- **Documentation**: [Full GNN guide](about_gnn.md) and [GNN Overview](gnn_overview.md)
- **Examples**: [Model gallery](gnn_examples_doc.md)
- **Help**:
  - [FAQ](../troubleshooting/README.md)
  - [Error guide](../troubleshooting/api_error_reference.md)
  - [Support](../SUPPORT.md)
- **Community**: [GitHub Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)

---

**🎓 You're now a GNN practitioner!** Ready to model complex cognitive agents and contribute to Active Inference research.

**Time taken**: ~15 minutes  
**Achievement unlocked**: First working GNN model ✨
