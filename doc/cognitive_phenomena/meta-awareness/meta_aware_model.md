# Meta-Awareness Model Documentation

> **📋 Document Metadata**  
> **Type**: Research Documentation | **Audience**: Researchers, Cognitive Scientists | **Complexity**: Advanced  
> **Cross-References**: [Meta-Awareness README](README.md) | [Cognitive Phenomena](../README.md) | [Advanced Patterns](../../gnn/advanced_modeling_patterns.md) | [Main Documentation](../../README.md)

## Overview

This document provides comprehensive documentation for the meta-awareness model implementation based on Sandved-Smith et al. (2021). The model implements hierarchical active inference with precision control for modeling meta-awareness and attentional control during mind-wandering.

**Paper**: "Towards a computational phenomenology of mental action: modelling meta-awareness and attentional control with deep parametric active inference"  
**Journal**: *Neuroscience of Consciousness*, 2021(1), niab018  
**DOI**: https://doi.org/10.1093/nc/niab018

## Model Architecture

### Three-Level Hierarchical Structure

The model implements a hierarchical active inference architecture with three levels:

**Level 1 (Perception):**
- Standard vs. oddball stimulus detection
- Bottom-up sensory processing
- Precision-weighted likelihood

**Level 2 (Attention):**
- Focused vs. distracted attentional states
- Attentional control and modulation
- Precision dynamics

**Level 3 (Meta-awareness):**
- High vs. low meta-awareness states
- Meta-cognitive monitoring
- Top-down control

### Key Features

- **Precision Control**: Dynamic precision modulation based on prediction errors
- **Policy Selection**: Active inference for attentional control (stay vs. switch)
- **Hierarchical Message Passing**: Bottom-up precision updating and top-down control
- **Mind-wandering Dynamics**: Naturalistic attentional state transitions

## Implementation

### Core Implementation

The implementation is provided in `sandved_smith_2021.py`:

```python
from sandved_smith_2021 import SandvedSmithModel

# Two-level model (attention only)
model = SandvedSmithModel(T=100, three_level=False, random_seed=42)
results = model.run_simulation()

# Three-level model (with meta-awareness)
model3 = SandvedSmithModel(T=100, three_level=True, random_seed=42)
results3 = model3.run_simulation()
```

### Model Components

**SandvedSmithModel Class:**
- Implements hierarchical active inference
- Supports two-level and three-level models
- Precision control and policy selection
- Mind-wandering dynamics

**Utility Functions:**
- `softmax`: Softmax normalization
- `precision_weighted_likelihood`: Precision-weighted likelihood computation
- `expected_free_energy`: Expected free energy calculation
- `policy_posterior`: Policy posterior computation
- `update_precision_beliefs`: Precision belief updating

**Visualization Functions:**
- `plot_figure_7`: Figure 7 visualization
- `plot_figure_10`: Figure 10 visualization
- `plot_figure_11`: Figure 11 visualization
- `save_all_figures`: Save all figure outputs

## Model Dynamics

### Precision Dynamics

The model implements dynamic precision control:

- **Bottom-Up Precision**: Precision updates based on prediction errors
- **Top-Down Precision**: Precision modulation from higher levels
- **Precision Weighting**: Precision-weighted likelihood computation
- **Precision Beliefs**: Belief updating for precision parameters

### Policy Selection

Attentional control through policy selection:

- **Stay Policy**: Maintain current attentional state
- **Switch Policy**: Change attentional state
- **Policy Posterior**: Bayesian policy selection
- **Expected Free Energy**: Policy evaluation

### Mind-Wandering Dynamics

Naturalistic attentional state transitions:

- **Attentional Cycles**: Cyclical attention patterns
- **Mind-Wandering Episodes**: Spontaneous mind-wandering
- **Meta-Awareness Modulation**: Meta-awareness effects on attention
- **State Transitions**: Hierarchical state transitions

## Paper Figures

The implementation reproduces key figures from the paper:

### Figure 7: Influence of Attentional State on Perception

Fixed attentional schedule demonstrating how attentional state affects perception:

```python
from sandved_smith_2021 import run_figure_7_simulation

results_fig7 = run_figure_7_simulation()
```

### Figure 10: Two-Level Model with Attentional Cycles

Two-level model showing attentional cycles and mind-wandering:

```python
from sandved_smith_2021 import run_figure_10_simulation

results_fig10 = run_figure_10_simulation()
```

### Figure 11: Three-Level Model with Meta-Awareness

Three-level model demonstrating meta-awareness effects:

```python
from sandved_smith_2021 import run_figure_11_simulation

results_fig11 = run_figure_11_simulation()
```

## Usage Examples

### Basic Simulation

```python
from sandved_smith_2021 import SandvedSmithModel
from visualizations import save_all_figures, display_results_summary

# Create and run three-level model
model = SandvedSmithModel(T=100, three_level=True, random_seed=42)
results = model.run_simulation()

# Save visualizations
save_all_figures(results, "output_directory")

# Display summary
display_results_summary(results)
```

### Custom Simulations

```python
# Custom model parameters
model = SandvedSmithModel(
    T=200,                    # Simulation length
    three_level=True,         # Use three-level model
    random_seed=42,          # Reproducibility
    precision_params={...}    # Custom precision parameters
)

results = model.run_simulation()
```

## Testing

The implementation includes comprehensive tests:

```bash
python test_implementation.py
```

**Test Coverage:**
- Mathematical utility functions
- Model consistency across runs
- Three-level vs two-level differences
- Mind-wandering dynamics
- Precision dynamics
- Figure mode behaviors

## Research Applications

### Cognitive Science

- **Mind-Wandering Research**: Understanding mind-wandering dynamics
- **Attention Research**: Attentional control mechanisms
- **Meta-Cognition**: Meta-awareness and meta-cognitive monitoring
- **Consciousness Research**: Computational models of consciousness

### Clinical Applications

- **ADHD Research**: Attentional control in ADHD
- **Depression Research**: Mind-wandering in depression
- **Anxiety Research**: Attentional biases in anxiety
- **Meditation Research**: Mindfulness and meta-awareness

### Computational Modeling

- **Active Inference**: Hierarchical active inference modeling
- **Precision Control**: Dynamic precision control mechanisms
- **Policy Selection**: Bayesian policy selection
- **Hierarchical Processing**: Multi-level hierarchical processing

## Integration with GNN

The model can be specified using GNN notation:

```gnn
## ModelName
MetaAwarenessModel

## ModelAnnotation
Three-level hierarchical active inference model for meta-awareness and attentional control.

## StateSpaceBlock
s_f0[2,1,type=categorical]      ### Perception: Standard=0, Oddball=1
s_f1[2,1,type=categorical]      ### Attention: Focused=0, Distracted=1
s_f2[2,1,type=categorical]      ### Meta-awareness: High=0, Low=1

o_m0[2,1,type=categorical]      ### Observation: Standard=0, Oddball=1

u_c0[2,1,type=categorical]      ### Policy: Stay=0, Switch=1

## Connections
s_f0 > o_m0                     ### Perception affects observation
s_f1 > s_f0                     ### Attention affects perception
s_f2 > s_f1                     ### Meta-awareness affects attention
u_c0 > s_f1                     ### Policy affects attention
```

## Related Documentation

- **[Meta-Awareness README](README.md)**: Implementation overview
- **[Cognitive Phenomena](../README.md)**: Cognitive phenomena modeling
- **[Advanced Patterns](../../gnn/advanced_modeling_patterns.md)**: Advanced modeling techniques
- **[GNN Overview](../../gnn/gnn_overview.md)**: GNN framework overview

## See Also

- **[Cognitive Phenomena](../README.md)**: Cognitive phenomena documentation
- **[Advanced Patterns](../../gnn/advanced_modeling_patterns.md)**: Advanced modeling patterns
- **[Main Documentation](../../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Last Updated**: 2025-12-30  
**Version**: 1.0.0  
**Implementation**: Complete and validated


