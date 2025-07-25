# PyMDP Gridworld POMDP: Pipeline Integration

⚠️ **NOTICE**: This directory contains reference implementations that have been **integrated into the main GNN pipeline**.

## New Pipeline Integration

The PyMDP simulation capabilities have been moved to the main pipeline structure:

### 📁 **Current Location**: `src/execute/pymdp/`
- **PyMDP Simulation**: `src/execute/pymdp/pymdp_simulation.py`
- **Utilities**: `src/execute/pymdp/pymdp_utils.py` 
- **Visualization**: `src/execute/pymdp/pymdp_visualizer.py`
- **Tests**: `src/execute/pymdp/test_*.py`

### 🔄 **Pipeline Steps Integration**

1. **Step 11 (Render)**: `src/11_render.py` + `src/render/pymdp/`
   - Parses GNN files using `src/gnn/` methods
   - Extracts POMDP state space dimensions and parameters
   - Renders PyMDP simulation code using `src/render/pymdp/pymdp_renderer.py`

2. **Step 12 (Execute)**: `src/12_execute.py` + `src/execute/pymdp/`
   - Executes rendered PyMDP simulations
   - Uses real PyMDP Agent and inference methods
   - Generates comprehensive outputs and visualizations

### 🧩 **GNN Integration Features**

**State Space Configuration from GNN**:
- `num_hidden_states` → A, B, D matrix dimensions
- `num_obs` → A, C matrix dimensions  
- `num_actions` → B matrix action slices
- Initial parameter values from GNN specifications

**Authentic PyMDP Implementation**:
```python
# Real PyMDP usage (not mock)
from pymdp import utils
from pymdp.agent import Agent

# GNN-configured matrices
A = utils.obj_array(1)  # Observation model
B = utils.obj_array(1)  # Transition model  
C = utils.obj_array(1)  # Preferences
D = utils.obj_array(1)  # Priors

# Real inference
agent = Agent(A=A, B=B, C=C, D=D)
qs = agent.infer_states(obs)
action = agent.sample_action()
```

## 📋 **Reference Files (This Directory)**

The files in this directory serve as **reference implementations** that informed the pipeline integration:

### Core Components
- `pymdp_gridworld_simulation.py` - Reference simulation [→ `src/execute/pymdp/pymdp_simulation.py`]
- `pymdp_gridworld_visualizer.py` - Reference visualizer [→ `src/execute/pymdp/pymdp_visualizer.py`]  
- `pymdp_utils.py` - Reference utilities [→ `src/execute/pymdp/pymdp_utils.py`]

### Tests & Documentation  
- `test_visualization.py` - Reference tests [→ `src/execute/pymdp/test_*.py`]
- `test_numpy_serialization.py` - Reference numpy tests [→ integrated]
- `POMDP_EXPLANATION.md` - Technical documentation

### 🎯 **Usage in Pipeline**

**Run GNN → PyMDP Pipeline**:
```bash
# Full pipeline (includes PyMDP rendering and execution)
python3 src/0_template.py  # → ... → 
python3 src/11_render.py   # → parses GNN, renders PyMDP code
python3 src/12_execute.py  # → executes PyMDP simulations

# Or specific steps
python3 src/11_render.py --target_dir input/gnn_files/
python3 src/12_execute.py --target_dir input/gnn_files/
```

**Direct PyMDP Execution**:
```python
from src.execute.pymdp import execute_pymdp_simulation
from src.gnn import parse_gnn_file

# Parse GNN file
gnn_spec = parse_gnn_file('input/gnn_files/actinf_pomdp_agent.md')

# Execute simulation
success, results = execute_pymdp_simulation(
    gnn_spec=gnn_spec.to_dict(),
    output_dir=Path('output/pymdp_test')
)
```

## 🔗 **Integration Benefits**

1. **GNN-Driven Configuration**: State spaces automatically extracted from GNN specifications
2. **Real PyMDP Methods**: No mock implementations - uses authentic PyMDP Agent class
3. **Pipeline Compatibility**: Integrates with GNN parsing, validation, and export steps
4. **Comprehensive Output**: JSON traces, performance metrics, visualizations
5. **Modular Design**: Can be used standalone or as part of full pipeline

## 📈 **Next Steps**

For **development** and **execution**, use the integrated pipeline modules:
- Development: `src/execute/pymdp/`
- Testing: `python3 -m src.execute.pymdp.test_*`
- Documentation: This README + `src/render/README.md`

For **reference** and **understanding**, consult the files in this directory.

---

**See Also**: 
- [Pipeline Architecture](../../../doc/pipeline/PIPELINE_ARCHITECTURE.md)
- [Render Module README](../../../src/render/README.md)
- [GNN Examples](../../../input/gnn_files/) 