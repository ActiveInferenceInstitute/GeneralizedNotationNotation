#!/usr/bin/env python3
"""
Standalone DisCoPy script template emitters for GNN Step 11.

Extracted from ``render.discopy.translator``.
"""

from typing import (
    Any,
    Dict,
)


def gnn_spec_to_discopy_code(gnn_spec: Dict[str, Any]) -> str:
    """
    Generate Python code that creates and draws a DisCoPy categorical diagram from GNN spec.

    Args:
        gnn_spec: The GNN specification as a Python dictionary

    Returns:
        Python code string that creates and visualizes the diagram
    """
    model_name = gnn_spec.get("name", "gnn_model")

    code = f'''#!/usr/bin/env python3
"""
DisCoPy Categorical Diagram for GNN Model: {model_name}

Generated from GNN specification.
This script creates and visualizes a DisCoPy categorical diagram.
"""

try:
    import discopy
    from discopy.tensor import Dim, Box, Diagram, Id
    from discopy.monoidal import Ty
    import matplotlib.pyplot as plt
    print("✓ DisCoPy and matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Import error: {{e}}")
    print("Please install required packages:")
    print("uv pip install discopy matplotlib")
    exit(1)

def create_gnn_diagram():
    """Create DisCoPy diagram from GNN specification."""
    
    # Extract model information
    model_name = "{model_name}"
    variables = {gnn_spec.get("variables", [])}
    connections = {gnn_spec.get("connections", [])}
    
    print(f"Creating diagram for model: {{model_name}}")
    print(f"Variables: {{len(variables)}}")
    print(f"Connections: {{len(connections)}}")
    
    # Create basic diagram structure
    # Create a simple diagram with core components.
    
    # Define basic types
    state_type = Ty("State")
    obs_type = Ty("Observation") 
    action_type = Ty("Action")
    
    # Create core boxes.
    transition_box = Box("Transition", state_type, state_type)
    observation_box = Box("Observation", state_type, obs_type)
    action_box = Box("Action", state_type, action_type)
    
    # Create simple diagram: State -> Transition -> State -> Observation
    diagram = Id(state_type) >> transition_box >> observation_box
    
    return diagram

def visualize_diagram(diagram):
    """Visualize the DisCoPy diagram."""
    try:
        # Draw the diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        diagram.draw(ax=ax)
        ax.set_title(f"DisCoPy Diagram: {model_name}")
        plt.tight_layout()
        
        # Save the diagram
        output_file = f"{{model_name}}_discopy_diagram.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Diagram saved as: {{output_file}}")
        
        # Show the diagram
        plt.show()
        
    except Exception as e:
        print(f"✗ Visualization error: {{e}}")
        print("Diagram structure:")
        print(diagram)

if __name__ == "__main__":
    print("=" * 60)
    print("DisCoPy Categorical Diagram Generator")
    print("=" * 60)
    
    # Create the diagram
    diagram = create_gnn_diagram()
    
    # Visualize the diagram
    visualize_diagram(diagram)
    
    print("\\n✓ DisCoPy diagram generation completed!")
'''

    return code


def gnn_spec_to_discopy_jax_code(gnn_spec: Dict[str, Any], seed: int = 0) -> str:
    """
    Generate Python code that creates and evaluates a DisCoPy matrix diagram with JAX.

    Args:
        gnn_spec: The GNN specification as a Python dictionary
        seed: Random seed for JAX operations

    Returns:
        Python code string that creates and evaluates the matrix diagram
    """
    model_name = gnn_spec.get("name", "gnn_model")

    code = f'''#!/usr/bin/env python3
"""
DisCoPy Matrix Diagram with JAX for GNN Model: {model_name}

Generated from GNN specification.
This script creates and evaluates a DisCoPy matrix diagram using JAX.
"""

try:
    import discopy
    from discopy.tensor import Dim, Box, Diagram, Id
    from discopy.matrix import Matrix
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    print("✓ DisCoPy, JAX, and matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Import error: {{e}}")
    print("Please install required packages:")
    print("uv pip install discopy jax jaxlib matplotlib")
    exit(1)

# Set JAX random seed
jax.random.PRNGKey({seed})

def create_matrix_diagram():
    """Create DisCoPy matrix diagram from GNN specification."""
    
    # Extract model information
    model_name = "{model_name}"
    variables = {gnn_spec.get("variables", [])}
    connections = {gnn_spec.get("connections", [])}
    
    print(f"Creating matrix diagram for model: {{model_name}}")
    print(f"Variables: {{len(variables)}}")
    print(f"Connections: {{len(connections)}}")
    
    # Create matrix diagram with JAX arrays
    # For demonstration, create simple matrices
    
    # Create random matrices for demonstration
    A_matrix = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
    B_matrix = jax.random.normal(jax.random.PRNGKey(2), (3, 3))
    
    # Create DisCoPy matrix boxes
    A_box = Matrix("A", A_matrix)
    B_box = Matrix("B", B_matrix)
    
    # Create diagram: A >> B
    diagram = A_box >> B_box
    
    return diagram, {{"A": A_matrix, "B": B_matrix}}

def evaluate_diagram(diagram, matrices):
    """Evaluate the matrix diagram."""
    try:
        # Evaluate the diagram
        result = diagram.eval()
        print(f"✓ Diagram evaluation successful")
        print(f"Result shape: {{result.shape}}")
        print(f"Result matrix:\\n{{result}}")
        
        # Show individual matrices
        for name, matrix in matrices.items():
            print(f"\\n{{name}} matrix:")
            print(f"Shape: {{matrix.shape}}")
            print(f"Values:\\n{{matrix}}")
        
        return result
        
    except Exception as e:
        print(f"✗ Evaluation error: {{e}}")
        return None

def visualize_matrices(matrices):
    """Visualize the matrices."""
    try:
        n_matrices = len(matrices)
        fig, axes = plt.subplots(1, n_matrices, figsize=(5*n_matrices, 4))
        
        if n_matrices == 1:
            axes = [axes]
        
        for i, (name, matrix) in enumerate(matrices.items()):
            im = axes[i].imshow(matrix, cmap='viridis')
            axes[i].set_title(f"{{name}} Matrix")
            axes[i].set_xlabel("Columns")
            axes[i].set_ylabel("Rows")
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = f"{{model_name}}_matrix_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Matrix visualization saved as: {{output_file}}")
        
        # Show the visualization
        plt.show()
        
    except Exception as e:
        print(f"✗ Visualization error: {{e}}")

if __name__ == "__main__":
    print("=" * 60)
    print("DisCoPy Matrix Diagram with JAX")
    print("=" * 60)
    
    # Create the matrix diagram
    diagram, matrices = create_matrix_diagram()
    
    # Evaluate the diagram
    result = evaluate_diagram(diagram, matrices)
    
    # Visualize the matrices
    visualize_matrices(matrices)
    
    print("\\n✓ DisCoPy matrix diagram evaluation completed!")
'''

    return code
