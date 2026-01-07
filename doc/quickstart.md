# GNN Quick Start Guide

> **📋 Document Metadata**  
> **Type**: Quick Start Guide | **Audience**: All Users | **Complexity**: Beginner  
> **Cross-References**: [Learning Paths](learning_paths.md) | [Main Documentation](README.md) | [Setup Guide](SETUP.md)

## ⚡ 5-Minute Demo

**Experience GNN instantly without installation!**

### **🎯 What You'll See**
- A working Active Inference agent in action
- Real-time model visualization
- Code generation across multiple frameworks
- Interactive model exploration

### **🚀 Option 1: Online Demo (Recommended)**
```bash
# Use our pre-configured cloud environment
curl -s https://demo.gnn-ai.org/quick-demo | bash

# Or via web interface:
# Visit: https://demo.gnn-ai.org/interactive
```

### **🖥️ Option 2: Local Quick Demo**
```bash
# One-command demo (no installation required)
docker run --rm -it gnn-ai/quickdemo:latest

# Or with existing Python:
curl -s https://raw.githubusercontent.com/ActiveInferenceInstitute/GeneralizedNotationNotation/main/scripts/quick_demo.py | python3
```

### **📊 What the Demo Shows**

#### **Model Creation (30 seconds)**
Watch as we create a simple navigation agent:
```gnn
## ModelName
NavigationAgent

## StateSpaceBlock
s_f0[2,1,type=categorical]  # Position: [left, right]
o_m0[2,1,type=categorical]  # Observation: [wall, open]
u_c0[2,1,type=categorical]  # Action: [left, right]

## Connections
s_f0 > o_m0                 # Position determines observation
s_f0, u_c0 > s_f0           # Position and action determine next position
```

#### **Real-Time Processing (2 minutes)**
See the GNN pipeline in action:
1. **✅ Parsing**: Extract model structure
2. **✅ Validation**: Check syntax and semantics  
3. **✅ Visualization**: Generate interactive diagrams
4. **✅ Code Generation**: Create PyMDP and RxInfer implementations
5. **✅ Simulation**: Run the agent and see results

#### **Results Exploration (2.5 minutes)**
Interactive exploration of outputs:
- **📊 Model Visualization**: Click-through network diagrams
- **🐍 Generated Python Code**: Working PyMDP implementation
- **🔢 Julia Code**: RxInfer.jl probabilistic programming
- **📈 Simulation Results**: Agent behavior over time
- **🎯 Performance Metrics**: Computational requirements

### **🎮 Interactive Features**

**Model Modifications**: Try these instant edits:
- Change preferences: `C_m0 = [1.0, 0.0]` → `C_m0 = [0.0, 1.0]`
- Add uncertainty: `A_m0 = [[1.0, 0.0], [0.0, 1.0]]` → `A_m0 = [[0.9, 0.1], [0.1, 0.9]]`
- Increase complexity: Add third position with `s_f0[3,1,type=categorical]`

**Real-Time Updates**: Watch how changes affect:
- Agent behavior patterns
- Computational complexity
- Generated code structure

### **📋 Demo Checklist**

After the 5-minute demo, you'll have seen:
- [ ] ✅ **GNN Syntax**: How models are specified
- [ ] ✅ **Validation**: Automatic error checking  
- [ ] ✅ **Multi-Framework**: Code for PyMDP, RxInfer, DisCoPy
- [ ] ✅ **Visualization**: Network diagrams and matrix heatmaps
- [ ] ✅ **Simulation**: Working Active Inference agent
- [ ] ✅ **Performance**: Resource estimation and optimization

### **🎯 Choose Your Next Step**

Based on what interested you most:

**🔬 "I want to understand the theory"** → [Research Learning Path](learning_paths.md#research-focused-path)
- Deep dive into Active Inference mathematics
- Explore cognitive modeling applications
- Review research methodology integration

**💻 "I want to build something"** → [Developer Learning Path](learning_paths.md#developer-focused-path)  
- Technical setup and integration
- Production deployment patterns
- Custom framework development

**🎓 "I want structured learning"** → [Academic Learning Path](learning_paths.md#academic-learning-path)
- Comprehensive curriculum
- Hands-on exercises and assessments
- Progressive skill building

**⚡ "I want to explore more examples"** → [Quick Exploration Path](learning_paths.md#quick-exploration-path)
- Gallery of pre-built models
- Interactive model browser
- Comparison with other approaches

---

## Get up and running with Generalized Notation Notation (GNN) in 10 minutes.

## What is GNN?

GNN is a text-based language for standardizing Active Inference generative models. It enables:

- **Model Specification**: Define cognitive models using clear, standardized notation
- **Cross-Platform Generation**: Automatically generate code for PyMDP, RxInfer.jl, and other frameworks  
- **Visualization**: Create interactive diagrams and categorical representations
- **Validation**: Check model consistency and estimate computational requirements
- **Documentation**: Generate comprehensive documentation and reports

## 🚀 Quick Installation

### Prerequisites
- Python 3.8+ 
- Git

### Install GNN
```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python src/main.py --help
```

## 📝 Your First GNN Model

### 1. Create a Simple Model

Create a file called `my_first_model.md`:

```gnn
## GNNVersionAndFlags
GNN v1.0
ProcessingFlags: active_inference

## ModelName
SimpleAgent

## ModelAnnotation
A basic Active Inference agent that navigates a simple environment.
The agent has 2 hidden states (locations), observes 2 possible observations,
and can take 2 actions (move left/right).

## StateSpaceBlock
### Hidden States
s_f0[2,1,type=categorical]      ### Agent position: Left=0, Right=1

### Observations  
o_m0[2,1,type=categorical]      ### What agent sees: Wall=0, Open=1

### Actions
u_c0[2,1,type=categorical]      ### Agent movement: Left=0, Right=1

## Connections
### Observations depend on position
s_f0 > o_m0                     ### Position determines what is observed

### Position changes based on action
s_f0, u_c0 > s_f0               ### New position depends on current position and action

## InitialParameterization
### Observation model (A matrix): P(observation|position)
A_m0 = [[0.9, 0.1], [0.1, 0.9]]

### Transition model (B matrix): P(new_position|old_position, action)
B_f0 = [
    [[0.8, 0.2], [0.3, 0.7]],   # Action 0 (Left): mostly stay, some move
    [[0.2, 0.8], [0.7, 0.3]]    # Action 1 (Right): mostly move, some stay
]

### Preferences (C vector): log preferences over observations
C_m0 = [0.0, 1.0]               ### Prefer open spaces over walls

### Prior beliefs (D vector): initial position beliefs
D_f0 = [0.5, 0.5]               ### Equal probability of starting left or right

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 10

## ActInfOntologyAnnotation
s_f0 = HiddenStateFactor_Position
o_m0 = ObservationModality_Vision  
u_c0 = ControlFactor_Movement
A_m0 = LikelihoodMatrix_Vision
B_f0 = TransitionMatrix_Position
C_m0 = PreferenceVector_Vision
D_f0 = PriorBelief_Position

## Footer
Created: 2024-01-15
LastModified: 2024-01-15
Version: 1.0

## Signature
ModelCreator: Your Name
Institution: Your Institution
License: MIT
```

### 2. Run the Pipeline

```bash
# Process your model through the complete pipeline
python src/main.py --target-dir my_first_model.md

# Or run specific steps
python src/main.py --target-dir my_first_model.md --steps 1 2 3 4 5
```

### 3. Check the Results

After running, you'll find:

```
output/
├── gnn_parsing/           # Parsed model structure
├── validation/            # Validation reports  
├── pymdp/                # Generated PyMDP code
├── rxinfer/              # Generated RxInfer.jl code
├── visualization/        # Model diagrams
├── discopy/              # Categorical diagrams
└── site/                 # Documentation website
    └── report/               # Comprehensive analysis reports

---

## 🏎️ Quick Path to First Inference

If you have GNN installed and want to run a model *right now*, follow these 3 steps:

### **1. Copy this GNN Snippet**
Save this as `fast_agent.md`:
```gnn
## ModelName
FastAgent

## StateSpaceBlock
s_f0[2,1,type=categorical]  ### State: [A, B]
o_m0[2,1,type=categorical]  ### Obs: [Red, Green]
u_c0[2,1,type=categorical]  ### Act: [Stay, Switch]

## Connections
s_f0 > o_m0
s_f0, u_c0 > s_f0
```

### **2. Generate and Execute in One Command**
```bash
python src/main.py --target-dir fast_agent.md --steps 1-10 --frameworks pymdp
```

### **3. Inspect the Logic**
Check `output/pymdp/fast_agent.py` to see the generated belief update logic. You can run it directly:
```bash
python output/pymdp/fast_agent.py
```

---

## 📊 Quick Examples

### View Generated PyMDP Code

```python
# output/pymdp/simple_agent.py
import numpy as np
from pymdp import Agent
from pymdp import utils

# A matrix (observations given states)
A = utils.obj_array(1)
A[0] = np.array([[0.9, 0.1], [0.1, 0.9]])

# B matrix (transitions given states and actions)  
B = utils.obj_array(1)
B[0] = np.array([[[0.8, 0.2], [0.3, 0.7]], 
                 [[0.2, 0.8], [0.7, 0.3]]])

# C vector (preferences)
C = utils.obj_array([np.array([0.0, 1.0])])

# D vector (priors)
D = utils.obj_array([np.array([0.5, 0.5])])

# Create agent
agent = Agent(A=A, B=B, C=C, D=D)

# Example usage
if __name__ == "__main__":
    obs = [1]  # Observe open space
    qs = agent.infer_states(obs)
    action = agent.sample_action()
    print(f"Belief: {qs[0]}")
    print(f"Action: {action[0]}")
```

### Run the Generated Code

```bash
# Run PyMDP simulation
cd output/pymdp
python simple_agent.py

# Run RxInfer simulation  
cd ../rxinfer
julia simple_agent.jl
```

## 🔧 Common Pipeline Commands

### Development Workflow
```bash
# Quick validation only
python src/main.py my_model.md --steps 1 2 3 4

# Generate code for specific framework
python src/main.py my_model.md --steps 1 2 4 5 9 --target pymdp

# Include visualization
python src/main.py my_model.md --steps 1 2 4 5 6

# Full pipeline with documentation
python src/main.py my_model.md --steps 1-13
```

### Batch Processing
```bash
# Process all models in a directory
python src/main.py --target-dir ./examples/

# Process specific pattern
python src/main.py --target-dir ./models/ --pattern "*.md"

# Parallel processing
python src/main.py --target-dir ./models/ --parallel --workers 4
```

### Configuration
```bash
# Use custom configuration
python src/main.py my_model.md --config config/development.yaml

# Set output directory
python src/main.py my_model.md --output-dir ./my_results/

# Debug mode
python src/main.py my_model.md --debug --verbose
```

## 📚 Learning Path

### 1. Start with Examples (5 minutes)
```bash
# Explore provided examples
ls src/gnn/gnn_examples/
python src/main.py --target-dir src/gnn/gnn_examples/actinf_pomdp_agent.md
```

### 2. Learn GNN Syntax (15 minutes)
- Read: [GNN Syntax Guide](gnn/gnn_syntax.md)
- Practice: Modify the example models
- Validate: Use the type checker to check your syntax

### 3. Try Different Frameworks (10 minutes)
```bash
# Generate PyMDP code
python src/main.py my_model.md --steps 1 2 4 5 9

# Generate RxInfer code  
python src/main.py my_model.md --steps 1 2 4 5 9 --target rxinfer

# Create visualizations
python src/main.py my_model.md --steps 1 2 4 5 6
```

### 4. Advanced Features (30 minutes)
- **Templates**: Use `doc/templates/` for common patterns
- **LLM Integration**: Try AI-enhanced analysis
- **Categorical Diagrams**: Explore DisCoPy translation
- **Multi-agent Systems**: Model agent interactions

## ⚡ Quick Troubleshooting

### Model Won't Parse
```bash
# Check syntax with detailed errors
python src/4_gnn_type_checker.py my_model.md --verbose

# Common issues:
# - Missing required sections (ModelName, StateSpaceBlock, etc.)
# - Incorrect variable naming (use s_f0, o_m0, u_c0 format)
# - Matrix dimension mismatches
```

### Generated Code Doesn't Run
```bash
# Validate matrices
python src/4_gnn_type_checker.py my_model.md --check-matrices

# Common issues:
# - Matrices don't sum to 1 (for probability matrices)
# - Dimension mismatches between A, B, C, D matrices
# - Invalid probability values (negative or > 1)
```

### Missing Dependencies
```bash
# Check what's missing
python src/1_setup.py --check-dependencies

# Install optional dependencies
pip install torch  # For PyTorch integration
pip install jax    # For JAX acceleration
```

## 🏗️ Templates for Common Models

### Use Ready-Made Templates

```bash
# Copy a template
cp doc/templates/basic_gnn_template.md my_new_model.md

# Available templates:
# - basic_gnn_template.md: Simple starting point
# - pomdp_template.md: Partially observable environments
# - multiagent_template.md: Multiple interacting agents
# - hierarchical_template.md: Multi-level architectures
```

### Customize Templates
1. Replace placeholder values with your specifics
2. Modify state spaces for your domain
3. Adjust matrices for your dynamics
4. Update preferences for your objectives

## 🚀 Next Steps

### Beginner Path
1. **Follow Tutorials**: Work through [tutorials/README.md](tutorials/README.md)
2. **Read Documentation**: Study [gnn/about_gnn.md](gnn/about_gnn.md)
3. **Join Community**: Participate in [discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)

### Intermediate Path  
1. **Framework Deep-Dive**: Master [PyMDP](pymdp/gnn_pymdp.md) or [RxInfer](rxinfer/gnn_rxinfer.md)
2. **Performance Optimization**: Learn [performance tuning](performance/README.md)
3. **Deploy Models**: Set up [production deployment](deployment/README.md)

### Advanced Path
1. **Categorical Modeling**: Explore [DisCoPy integration](discopy/gnn_discopy.md)
2. **Extend GNN**: Develop new pipeline steps
3. **Research Applications**: Apply to your research domain

## 📖 Essential Documentation

- **[GNN Syntax Reference](gnn/gnn_syntax.md)**: Complete syntax guide
- **[Pipeline Documentation](pipeline/README.md)**: Detailed pipeline steps
- **[API Reference](api/README.md)**: Programming interface
- **[Troubleshooting](troubleshooting/README.md)**: Common problems and solutions

## 🤝 Getting Help

### Community Resources
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
- **Issues**: [Report bugs or request features](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
- **Documentation**: Search the comprehensive docs in `/doc`

### Quick Support
1. **Check Examples**: Look at `src/gnn/gnn_examples/` for similar models
2. **Validate First**: Run step 4 (type checker) to catch common errors
3. **Read Error Messages**: They often contain helpful hints
4. **Use Debug Mode**: Add `--debug --verbose` for detailed output

## 🎯 Quick Reference Card

### Essential Commands
```bash
# Basic processing
python src/main.py model.md

# Just validation  
python src/main.py model.md --steps 1-4

# Generate code
python src/main.py model.md --steps 1-5,9

# Full pipeline
python src/main.py model.md --steps 1-14

# Debug mode
python src/main.py model.md --debug --verbose
```

### Key File Locations
```
src/gnn/gnn_examples/  # Example GNN models
doc/templates/        # Model templates  
doc/gnn/gnn_syntax.md # Syntax reference
output/               # Generated results
config/               # Configuration files
```

### Pipeline Steps
1. **Setup** - Initialize environment and dependencies
2. **GNN Processing** - Read, parse, and validate models
3. **Tests** - Run validation tests (optional)
4. **Type Checking** - Validate syntax and semantics
5. **Export** - Export to standard formats
6. **Visualization** - Generate diagrams and graphs
7. **MCP** - Model Context Protocol integration
8. **Ontology** - Process ontology annotations
9. **Render** - Generate framework code
10. **Execute** - Run generated simulations
11. **LLM** - AI-enhanced analysis
12. **Audio** - Generate audio representations
13. **Website** - Static HTML site generation
14. **Report** - Comprehensive analysis reports

---

**🎉 Congratulations!** You now have a working GNN model. Explore the generated outputs and dive deeper into the [full documentation](README.md) to unlock GNN's full potential for your research and applications.

---

**Status**: Production-Ready Quick Start Guide  
**Next Steps**: [Full Documentation](README.md) | [Learning Paths](learning_paths.md) 