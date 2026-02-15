# Meta-Aware-2: GNN-Configurable Meta-Awareness Simulation

A modular, standalone, and fully GNN-configurable implementation of the computational phenomenology of meta-awareness and attentional control based on Sandved-Smith et al. (2021).

## Overview

Meta-Aware-2 is the "golden spike" GNN-specified executable implementation that demonstrates hierarchical active inference for meta-awareness phenomena. It supports arbitrary dimensionality, is fully configurable through GNN specification files, and provides comprehensive simulation, analysis, and visualization capabilities.

### Key Features

- **Fully GNN-Configurable**: All model parameters specified via TOML configuration files
- **Generic Dimensionality**: Supports arbitrary state space, observation, and action dimensions
- **Modular Architecture**: Cleanly separated components for easy extension and modification
- **Comprehensive Logging**: Detailed simulation tracking and performance monitoring
- **Publication-Quality Figures**: Reproduces all figures from Sandved-Smith et al. (2021)
- **Complete Test Suite**: Unit tests, integration tests, and validation against paper results
- **Standalone Implementation**: No external dependencies on other GNN pipeline components

## Quick Start

### Basic Usage

```bash
# Run with default configuration
python run_meta_awareness.py config/meta_awareness_gnn.toml

# Run specific simulation modes
python run_meta_awareness.py config/meta_awareness_gnn.toml -m figure_10 figure_11

# Run with custom settings
python run_meta_awareness.py config/meta_awareness_gnn.toml -o ./my_results -s 42 -l DEBUG
```

### Show Configuration

```bash
# Display configuration details
python run_meta_awareness.py config/meta_awareness_gnn.toml --show-config
```

### Run Tests

```bash
# Run comprehensive test suite
python run_meta_awareness.py --test
```

## Architecture

### Directory Structure

```
meta-aware-2/
├── config/                    # GNN configuration files
│   ├── gnn_parser.py         # Configuration parser
│   └── meta_awareness_gnn.toml # Main configuration
├── core/                     # Core simulation model
│   └── meta_awareness_model.py # Main model implementation
├── utils/                    # Mathematical utilities
│   └── math_utils.py         # Generic math functions
├── simulation_logging/       # Logging system
│   └── simulation_logger.py  # Comprehensive logging
├── visualization/            # Figure generation
│   └── figure_generator.py   # Publication figures
├── execution/                # Simulation runner
│   └── simulation_runner.py  # Main execution pipeline
├── tests/                    # Test suite
│   └── test_simulation.py    # Comprehensive tests
├── run_meta_awareness.py     # Main executable ("golden spike")
└── README.md                 # This file
```

### Component Overview

#### 1. GNN Configuration System (`config/`)

- **`gnn_parser.py`**: Parses TOML configuration files into Python dataclasses
- **`meta_awareness_gnn.toml`**: Main configuration specifying all model parameters
- Supports arbitrary dimensionality and hierarchical structure specification

#### 2. Core Model (`core/`)

- **`meta_awareness_model.py`**: Main meta-awareness simulation model
- Implements hierarchical active inference with generic state spaces
- Supports both 2-level and 3-level model configurations
- Fully driven by GNN configuration parameters

#### 3. Mathematical Utilities (`utils/`)

- **`math_utils.py`**: Generic mathematical operations
- Softmax, normalization, entropy, KL divergence, Bayesian model averaging
- Precision weighting, attentional charge computation, free energy calculations
- All functions support arbitrary dimensionality

#### 4. Logging System (`simulation_logging/`)

- **`simulation_logger.py`**: Comprehensive simulation tracking
- Performance monitoring, error tracking, metrics collection
- Structured logging with correlation contexts
- Automatic log file management and metrics export

#### 5. Visualization (`visualization/`)

- **`figure_generator.py`**: Publication-quality figure generation
- Reproduces Figures 7, 10, and 11 from Sandved-Smith et al. (2021)
- Additional analysis figures for precision, free energy, and comparative analysis
- Supports multiple output formats (PNG, PDF, SVG)

#### 6. Execution Pipeline (`execution/`)

- **`simulation_runner.py`**: Main simulation orchestration
- Integrates all components into complete analysis pipeline
- Supports multiple simulation modes and comparative analysis
- Automatic result saving and figure generation

#### 7. Test Suite (`tests/`)

- **`test_simulation.py`**: Comprehensive test coverage
- Unit tests, integration tests, performance benchmarks
- Validation against paper results and expected behaviors
- Reproducibility testing with random seeds

## Configuration

### GNN Configuration File Format

The simulation is fully configured through TOML files. Here's the basic structure:

```toml
[model]
name = "meta_awareness_model"
description = "Meta-awareness with hierarchical active inference"
num_levels = 3
level_names = ["perception", "attention", "meta_awareness"]
time_steps = 100
oddball_pattern = "default"

[levels.perception]
state_dim = 2
obs_dim = 2
action_dim = 0

[levels.attention]
state_dim = 2
obs_dim = 2
action_dim = 2

[levels.meta_awareness]
state_dim = 2
obs_dim = 2
action_dim = 2

[precision_bounds]
perception = [0.5, 2.0]
attention = [2.0, 4.0]

[policy_precision]
2_level = 2.0
3_level = 4.0

[simulation_modes]
default = "natural_dynamics"
figure_7 = "fixed_attention_schedule"
figure_10 = "two_level_mind_wandering"
figure_11 = "three_level_meta_awareness"
```

### Key Configuration Sections

#### Model Configuration

- `name`: Model identifier
- `num_levels`: Number of hierarchical levels (2 or 3)
- `level_names`: Names for each hierarchical level
- `time_steps`: Simulation duration
- `oddball_pattern`: Stimulus sequence pattern

#### Level Configuration

For each level, specify:

- `state_dim`: Dimension of state space
- `obs_dim`: Dimension of observation space
- `action_dim`: Dimension of action space (0 for no actions)

#### Precision Parameters

- `precision_bounds`: [min, max] precision values for each level
- `policy_precision`: Policy selection precision parameters

#### Simulation Modes

Define different simulation scenarios:

- `figure_7`: Fixed attention schedule (reproduces Figure 7)
- `figure_10`: Two-level mind-wandering dynamics (reproduces Figure 10)
- `figure_11`: Three-level meta-awareness (reproduces Figure 11)

## Simulation Modes

### Figure 7: Fixed Attention Schedule

- Models with predetermined attentional states
- Demonstrates precision dynamics under controlled conditions
- Reproduces Figure 7 from Sandved-Smith et al. (2021)

### Figure 10: Two-Level Mind-Wandering

- Two-level hierarchical model (perception + attention)
- Natural mind-wandering dynamics through policy selection
- Shows attention switching and precision modulation
- Reproduces Figure 10 from the paper

### Figure 11: Three-Level Meta-Awareness

- Three-level model (perception + attention + meta-awareness)
- Meta-cognitive control of attentional states
- Demonstrates meta-awareness influencing attention regulation
- Reproduces Figure 11 from the paper

## Results and Output

### Output Directory Structure

```
output/
├── results/                  # Simulation data
│   ├── mode_results.json     # Human-readable results
│   └── mode_results.pkl      # Full numpy results
├── figures/                  # Generated figures
│   ├── figure_7_*.png        # Figure 7 reproductions
│   ├── figure_10_*.png       # Figure 10 reproductions
│   ├── figure_11_*.png       # Figure 11 reproductions
│   └── analysis_*.png        # Additional analysis figures
└── logs/                     # Simulation logs
    ├── sim_*.log             # Main simulation logs
    ├── sim_*_performance.log # Performance tracking
    ├── sim_*_errors.log      # Error logs
    └── sim_*_metrics.json    # Quantitative metrics
```

### Analysis Metrics

The simulation automatically computes comprehensive analysis including:

#### Mind-Wandering Analysis

- Focused vs. distracted percentages
- Transition counts and episode lengths
- Attentional stability metrics

#### Precision Dynamics

- Mean, standard deviation, range of precision values
- Precision change rates and variability coefficients
- Cross-level precision correlations

#### Free Energy Analysis

- Expected and variational free energy statistics
- Policy evaluation metrics
- Free energy distributions and trends

#### Behavioral Patterns

- Stimulus-response relationships
- Oddball detection performance
- Response timing and accuracy

## Mathematical Framework

### Hierarchical Active Inference

The model implements hierarchical active inference with:

1. **State Estimation**: Bayesian belief updating at each hierarchical level
2. **Precision Weighting**: Dynamic precision modulation based on attention
3. **Policy Selection**: Expected free energy minimization for action selection
4. **Attentional Charge**: Cross-level prediction error propagation

### Key Equations

#### Precision-Weighted Likelihood

```
A_bar = softmax(γ * log(A))
```

#### Attentional Charge

```
AtC = sum(|O - A_bar * X| * A)
```

#### Expected Free Energy

```
G = -sum(P(o) * log(P(o))) + sum(P(s) * H(A))
```

#### Policy Posterior

```
π = softmax(-γ_G * G + log(π_prior))
```

## Testing and Validation

### Test Suite Coverage

- **Configuration Loading**: TOML parsing and validation
- **Model Initialization**: Proper setup with arbitrary dimensions
- **Simulation Execution**: Complete simulation runs
- **Mathematical Operations**: All utility function correctness
- **Reproducibility**: Consistent results with same random seeds
- **Performance**: Simulation speed and memory usage
- **Paper Validation**: Results matching Sandved-Smith et al. (2021)

### Running Tests

```bash
# Run all tests
python run_meta_awareness.py --test

# Run specific test module
python -m pytest tests/test_simulation.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Extending the Model

### Adding New Levels

1. Update configuration to include new level:

```toml
[levels.new_level]
state_dim = 3
obs_dim = 3
action_dim = 2
```

1. Add precision bounds:

```toml
[precision_bounds]
new_level = [1.0, 5.0]
```

1. The model automatically handles arbitrary numbers of levels

### Custom Simulation Modes

1. Define mode in configuration:

```toml
[simulation_modes]
custom_mode = "custom_dynamics"
```

1. Implement mode logic in `meta_awareness_model.py`:

```python
def _setup_custom_dynamics(self, mode: str):
    # Custom mode implementation
    pass
```

### Adding Analysis Metrics

1. Extend `_compute_additional_analysis()` in `simulation_runner.py`
2. Add visualization in `figure_generator.py`
3. Update test validation in `test_simulation.py`

## Dependencies

### Required Packages

- `numpy`: Numerical computations
- `matplotlib`: Figure generation
- `seaborn`: Statistical visualization
- `toml`: Configuration file parsing

### Installation

```bash
uv pip install numpy matplotlib seaborn toml
```

## Performance Considerations

### Simulation Speed

- Typical simulation (100 time steps): < 1 second
- Memory usage scales linearly with time steps and dimensions
- Performance logging tracks computational bottlenecks

### Optimization Tips

- Reduce time steps for faster testing
- Use WARNING log level for production runs
- Enable only necessary simulation modes
- Use pickle format for large result datasets

## Troubleshooting

### Common Issues

#### Configuration Errors

```
Error loading configuration: [details]
```

- Check TOML syntax with `--show-config`
- Validate all required sections are present
- Ensure dimensions are positive integers

#### Numerical Issues

```
Warning: Numerical issue (underflow): [details]
```

- Check precision bounds are reasonable
- Verify matrix initialization
- Adjust tolerance in validation config

#### Memory Issues

```
MemoryError: Unable to allocate array
```

- Reduce time steps or state dimensions
- Use smaller simulation modes for testing
- Monitor memory usage in performance logs

### Debug Mode

```bash
python run_meta_awareness.py config.toml -l DEBUG
```

Provides detailed execution information and error traces.

## Scientific Validation

### Paper Reproduction

The implementation reproduces key results from Sandved-Smith et al. (2021):

- **Figure 7**: Fixed attention schedule with precision dynamics
- **Figure 10**: Two-level mind-wandering with ~65% distracted time
- **Figure 11**: Three-level meta-awareness control
- **Quantitative Metrics**: Mind-wandering percentages, precision ranges, transition counts

### Validation Criteria

- Mind-wandering percentage: 30-90% (paper: ~65%)
- Precision range: 0.5-2.0 with meaningful variation
- Transition counts: 50-80 per 100 time steps
- Numerical stability: No NaN or infinite values
- Reproducibility: Identical results with same random seed

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{sandved-smith2021,
  title={A computational phenomenology of mental action},
  author={Sandved-Smith, Lars and Hesp, Casper and Mattout, J{\'e}r{\'e}mie and Friston, Karl and Lutz, Antoine and Ramstead, Maxwell JD},
  journal={Journal of Mathematical Psychology},
  volume={105},
  pages={102607},
  year={2021},
  publisher={Elsevier}
}
```

## License

This implementation is part of the GeneralizedNotationNotation (GNN) project and follows the same license terms.

## Contributing

Contributions are welcome! Please:

1. Run the test suite before submitting changes
2. Add tests for new functionality
3. Update documentation for new features
4. Follow the existing code style and organization
5. Validate scientific accuracy against paper results

## Support

For questions or issues:

1. Check this README and configuration examples
2. Run the test suite to validate your setup
3. Use debug mode for detailed error information
4. Review the comprehensive logs for troubleshooting

---

**Meta-Aware-2**: A comprehensive, modular, and scientifically validated implementation of computational phenomenology for meta-awareness research.
