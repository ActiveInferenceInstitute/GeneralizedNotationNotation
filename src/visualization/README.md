# GNN Visualization Module

This module provides tools for visualizing Generalized Notation Notation (GNN) models. It processes GNN example files and generates comprehensive state-space visualizations that represent all variables and the complete model structure.

## Features

- Parse GNN files in both CSV and Markdown formats
- Generate state-space visualizations showing all variables and their dimensions
- Visualize connections between variables in the model
- Create combined visualizations that represent the full model
- Save all visualizations to a time-stamped output folder

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can use the GNN visualization module in several ways:

### Command-Line Interface

Process a single GNN file:

```bash
python -m gnn.visualization path/to/gnn_file.md
```

Process all GNN files in a directory:

```bash
python -m gnn.visualization path/to/gnn_examples/
```

Specify an output directory:

```bash
python -m gnn.visualization path/to/gnn_file.md -o path/to/output
```

Process directories recursively:

```bash
python -m gnn.visualization path/to/gnn_examples/ --recursive
```

### Top-level Script

For convenience, you can also use the top-level script:

```bash
python visualize_gnn.py path/to/gnn_file.md
```

### From Python Code

```python
from gnn.visualization import GNNVisualizer

# Create a visualizer
visualizer = GNNVisualizer()

# Process a single file
output_dir = visualizer.visualize_file("path/to/gnn_file.md")
print(f"Visualizations saved to {output_dir}")

# Process a directory
output_dir = visualizer.visualize_directory("path/to/gnn_examples/")
print(f"Visualizations saved to {output_dir}")
```

## Output

The module creates a time-stamped directory (e.g., `gnn_visualization_20230615_120000/`) with subdirectories for each processed file. Each subdirectory contains:

- `state_space.png`: Visualization of all state-space variables
- `connections.png`: Visualization of connections between variables
- `combined_visualization.png`: Combined visualization of the model
- `model_metadata.json`: Extracted metadata from the GNN file
- `full_model_data.json`: Complete parsed data from the GNN file 