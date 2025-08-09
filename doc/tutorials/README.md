# GNN Tutorials

## Overview
This directory contains step-by-step tutorials for learning and using GeneralizedNotationNotation (GNN), from basic concepts to advanced applications.

## Learning Path

### 🎯 Beginner Tutorials

#### 1. Your First GNN Model
**File**: `01_first_gnn_model.md`
**Duration**: 15 minutes
**Goals**: Create and validate a basic GNN model
**Prerequisites**: None

**What you'll learn**:
- GNN file structure and sections
- Basic variable definitions
- Simple connections
- Running the pipeline

**Tutorial Content**:
```markdown
# Your First GNN Model

## Step 1: Create Basic Model
Create file `my_first_model.md`:

```markdown
# My First GNN Model

## GNNVersionAndFlags
GNN v1.0
ProcessingFlags: default

## ModelName
BasicPerceptionModel

## ModelAnnotation
A simple model with one hidden state and one observation.

## StateSpaceBlock
s_f0[2,1,type=categorical]  ### Hidden state with 2 categories
o_m0[2,1,type=categorical]  ### Observation with 2 categories

## Connections
s_f0 > o_m0  ### Hidden state influences observation

## InitialParameterization
A_m0 = [[0.8, 0.2], [0.3, 0.7]]  ### Likelihood matrix
D_f0 = [0.5, 0.5]  ### Uniform prior over states

## Time
Static
```

## Step 2: Validate Model
```bash
python src/main.py --target-dir path/to/my_first_model.md --only-steps 1,4
```

## Step 3: Visualize Model
```bash
python src/main.py --target-dir path/to/my_first_model.md --only-steps 6
```

Check `output/8_visualization_output/` for generated diagrams.

## Step 4: Export Model
```bash
python src/main.py --target-dir path/to/my_first_model.md --only-steps 5
```

Your model is now exported to JSON, XML, and other formats!
```

#### 2. Understanding GNN Syntax
**File**: `02_gnn_syntax_guide.md`
**Duration**: 20 minutes
**Goals**: Master GNN syntax rules and conventions

#### 3. Variable Types and Dimensions
**File**: `03_variables_and_dimensions.md`
**Duration**: 25 minutes
**Goals**: Learn about different variable types and sizing

### 🚀 Intermediate Tutorials

#### 4. Dynamic Models with Actions
**File**: `04_dynamic_models.md`
**Duration**: 30 minutes
**Goals**: Create time-dependent models with control

#### 5. Multi-Modal Observations
**File**: `05_multimodal_observations.md`
**Duration**: 35 minutes
**Goals**: Handle multiple observation modalities

#### 6. Hierarchical Models
**File**: `06_hierarchical_models.md`
**Duration**: 40 minutes
**Goals**: Build complex nested model structures

### 🎓 Advanced Tutorials

#### 7. Custom Simulation Backends
**File**: `07_custom_backends.md`
**Duration**: 45 minutes
**Goals**: Extend GNN to new simulation environments

#### 8. LLM-Enhanced Analysis
**File**: `08_llm_analysis.md`
**Duration**: 30 minutes
**Goals**: Use AI to analyze and improve models

#### 9. Categorical Diagrams with DisCoPy
**File**: `09_categorical_diagrams.md`
**Duration**: 50 minutes
**Goals**: Translate models to category theory

#### 10. JAX Performance Optimization
**File**: `10_jax_optimization.md`
**Duration**: 45 minutes
**Goals**: High-performance model evaluation

## Specialized Tutorials

### 📊 Domain-Specific Applications

#### Active Inference for Robotics
**File**: `robotics/active_inference_robotics.md`
**Goals**: Apply GNN to robotic control and perception

#### Cognitive Modeling
**File**: `cognitive/cognitive_modeling.md`
**Goals**: Model human cognition and decision-making

#### Multi-Agent Systems
**File**: `multiagent/multiagent_systems.md`
**Goals**: Coordinate multiple Active Inference agents

### 🔧 Developer Tutorials

#### Contributing to GNN
**File**: `development/contributing_tutorial.md`
**Goals**: Learn the development workflow

#### Writing Pipeline Extensions
**File**: `development/pipeline_extensions.md`
**Goals**: Add new pipeline steps

#### Custom Visualization Types
**File**: `development/custom_visualizations.md`
**Goals**: Create domain-specific visualizations

## Quick Start Cheat Sheets

### GNN Syntax Quick Reference
```markdown
# Essential GNN Elements

## Variables
s_f0[dims,type]     # Hidden state factor 0
o_m0[dims,type]     # Observation modality 0
u_c0[dims,type]     # Control factor 0
π_c0[dims,type]     # Policy factor 0

## Connections
s_f0 > o_m0         # Directed edge (causality)
s_f0 - s_f1         # Undirected edge (correlation)

## Matrices
A_m0[obs,state]     # Likelihood matrix
B_f0[state,state,action] # Transition matrix
C_m0[obs]           # Preference vector
D_f0[state]         # Prior vector

## Types
type=categorical    # Discrete categories
type=continuous     # Real-valued
type=binary         # Boolean
```

### Pipeline Commands Quick Reference
```bash
# Basic commands
python src/main.py --target-dir examples/
python src/main.py --only-steps 1,4,6
python src/main.py --skip 11,12,13

# Advanced options
python src/main.py --parallel --conservative
python src/main.py --debug --verbose
python src/main.py --output-dir custom_output/
```

## Interactive Learning

### Jupyter Notebooks
- `notebooks/interactive_gnn_tutorial.ipynb`
- `notebooks/active_inference_primer.ipynb`
- `notebooks/model_comparison_workshop.ipynb`

### Online Sandbox
- Web-based GNN editor (coming soon)
- Real-time syntax validation
- Instant visualization

## Tutorial Assets

### Example Models
Located in `tutorials/assets/`:
- `basic_examples/` - Simple models for learning
- `intermediate_examples/` - More complex scenarios
- `advanced_examples/` - Cutting-edge applications
- `broken_examples/` - Common mistakes to avoid

### Datasets
- `data/simple_observations.csv`
- `data/robotics_sensors.json`
- `data/cognitive_experiment_results.pkl`

### Solutions
- `solutions/` - Complete solutions for tutorial exercises
- `solutions/explained/` - Step-by-step explanations

## Getting Help

### Tutorial-Specific Support
- Check tutorial README files
- Look for `HINTS.md` in tutorial directories
- Compare your work with provided solutions

### Community Learning
- Join tutorial discussion forums
- Participate in weekly GNN workshops
- Share your models with the community

### Troubleshooting
- See `../troubleshooting/README.md` for common issues
- Use `--debug` flag for detailed error messages
- Check tutorial-specific troubleshooting sections

## Tutorial Roadmap

### Coming Soon
- Web-based interactive tutorials
- Video walkthroughs
- Advanced mathematical foundations
- Industry-specific applications
- Integration with popular ML frameworks

### Contribute Tutorials
We welcome tutorial contributions! See `../development/writing_tutorials.md` for guidelines.

## Assessment and Certification

### Tutorial Completion
- Each tutorial includes exercises and solutions
- Self-assessment quizzes
- Practical project assignments

### GNN Proficiency Levels
- **Beginner**: Can create and validate basic models
- **Intermediate**: Handles dynamic and multi-modal models
- **Advanced**: Develops custom extensions and optimizations
- **Expert**: Contributes to GNN development and research

Start with `01_first_gnn_model.md` and work your way up! 