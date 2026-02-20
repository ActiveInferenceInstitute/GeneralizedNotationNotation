"""
Statistical and analytical visualization functions for advanced visualization.

Provides statistical plots, matrix correlation heatmaps, POMDP transition
analysis, policy visualization, and interactive Plotly dashboards.

Extracted from processor.py for maintainability.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any

# Import matplotlib for plotting (with fallback for headless environments)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

# Seaborn availability
SEABORN_AVAILABLE = False
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None

from ._shared import AdvancedVisualizationAttempt


def _generate_statistical_plots(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate statistical analysis visualizations"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="statistical",
        model_name=model_name,
        status="in_progress"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not np:
            attempt.status = "skipped"
            attempt.error_message = "matplotlib/numpy not available"
            return attempt

        output_files = []

        # Extract data
        variables = model_data.get("variables", [])
        parameters = model_data.get("parameters", [])

        # 1. Variable type distribution
        if variables:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Variable type distribution
            var_types = {}
            for var in variables:
                if isinstance(var, dict):
                    vtype = var.get("var_type", "unknown")
                    var_types[vtype] = var_types.get(vtype, 0) + 1

            if var_types:
                axes[0, 0].pie(var_types.values(), labels=var_types.keys(), autopct='%1.1f%%')
                axes[0, 0].set_title("Variable Type Distribution")

            # Variable dimension distribution
            dim_counts = {}
            for var in variables:
                if isinstance(var, dict):
                    dims = var.get("dimensions", [])
                    dim_str = str(dims) if dims else "scalar"
                    dim_counts[dim_str] = dim_counts.get(dim_str, 0) + 1

            if dim_counts:
                axes[0, 1].bar(range(len(dim_counts)), list(dim_counts.values()))
                axes[0, 1].set_xticks(range(len(dim_counts)))
                axes[0, 1].set_xticklabels(list(dim_counts.keys()), rotation=45, ha='right')
                axes[0, 1].set_title("Variable Dimension Distribution")
                axes[0, 1].set_ylabel("Count")

            # Parameter value distribution (for scalar parameters)
            scalar_values = []
            for param in parameters:
                if isinstance(param, dict):
                    value = param.get("value")
                    if isinstance(value, (int, float)):
                        scalar_values.append(value)

            if scalar_values:
                axes[1, 0].hist(scalar_values, bins=min(20, len(scalar_values)), alpha=0.7)
                axes[1, 0].set_title("Scalar Parameter Distribution")
                axes[1, 0].set_xlabel("Value")
                axes[1, 0].set_ylabel("Frequency")

            # Matrix size distribution
            matrix_sizes = []
            for param in parameters:
                if isinstance(param, dict):
                    value = param.get("value")
                    if isinstance(value, (list, tuple)):
                        try:
                            arr = np.array(value)
                            matrix_sizes.append(arr.size)
                        except Exception:
                            pass

            if matrix_sizes:
                axes[1, 1].hist(matrix_sizes, bins=min(15, len(matrix_sizes)), alpha=0.7, color='green')
                axes[1, 1].set_title("Matrix Size Distribution")
                axes[1, 1].set_xlabel("Matrix Size (elements)")
                axes[1, 1].set_ylabel("Frequency")

            plt.suptitle(f"Statistical Analysis: {model_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            output_file = output_dir / f"{model_name}_statistical_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            output_files.append(str(output_file))

        attempt.status = "success"
        attempt.output_files = output_files

    except Exception as e:
        logger.error(f"Failed to generate statistical plots for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_matrix_correlations(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate matrix correlation heatmaps"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="matrix_correlations",
        model_name=model_name,
        status="in_progress"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not np:
            attempt.status = "skipped"
            attempt.error_message = "matplotlib/numpy not available"
            return attempt

        try:
            from visualization.matrix_visualizer import MatrixVisualizer
        except ImportError:
            try:
                from src.visualization.matrix_visualizer import MatrixVisualizer
            except ImportError:
                attempt.status = "failed"
                attempt.error_message = "MatrixVisualizer not available"
                return attempt
        mv = MatrixVisualizer()

        # Extract matrices
        parameters = model_data.get("parameters", [])
        matrices = mv.extract_matrix_data_from_parameters(parameters)

        if len(matrices) < 2:
            attempt.status = "skipped"
            attempt.error_message = "Need at least 2 matrices for correlation"
            return attempt

        # Flatten matrices and compute correlations
        matrix_names = list(matrices.keys())
        matrix_vectors = []

        for name in matrix_names:
            matrix = matrices[name]
            # Flatten to 1D
            flat = matrix.flatten()
            # Normalize
            if flat.max() > flat.min():
                flat = (flat - flat.min()) / (flat.max() - flat.min())
            matrix_vectors.append(flat)

        # Pad to same length
        max_len = max(len(v) for v in matrix_vectors)
        for i, vec in enumerate(matrix_vectors):
            if len(vec) < max_len:
                padded = np.zeros(max_len)
                padded[:len(vec)] = vec
                matrix_vectors[i] = padded

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(matrix_vectors)

        # Create heatmap
        plt.figure(figsize=(10, 8))

        if SEABORN_AVAILABLE and sns:
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                       xticklabels=matrix_names, yticklabels=matrix_names,
                       cmap='coolwarm', center=0, vmin=-1, vmax=1)
        else:
            im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.xticks(range(len(matrix_names)), matrix_names, rotation=45, ha='right')
            plt.yticks(range(len(matrix_names)), matrix_names)
            # Add text annotations
            for i in range(len(matrix_names)):
                for j in range(len(matrix_names)):
                    plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

        plt.title(f"Matrix Correlation Heatmap: {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = output_dir / f"{model_name}_matrix_correlations.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        attempt.status = "success"
        attempt.output_files = [str(output_file)]

    except Exception as e:
        logger.error(f"Failed to generate matrix correlations for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_interactive_plotly_dashboard(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate full interactive Plotly dashboard"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="interactive_dashboard",
        model_name=model_name,
        status="in_progress"
    )

    start_time = time.time()

    try:
        # Check for plotly
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            plotly_available = True
        except ImportError:
            plotly_available = False

        if not plotly_available:
            attempt.status = "skipped"
            attempt.error_message = "plotly not available"
            return attempt

        # Extract data
        variables = model_data.get("variables", [])
        parameters = model_data.get("parameters", [])
        connections = model_data.get("connections", [])

        from visualization.matrix_visualizer import MatrixVisualizer
        mv = MatrixVisualizer()
        matrices = mv.extract_matrix_data_from_parameters(parameters)

        # Create dashboard with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Variable Types", "Matrix Overview", "Network Graph", "Model Statistics"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )

        # 1. Variable type pie chart
        var_types = {}
        for var in variables:
            if isinstance(var, dict):
                vtype = var.get("var_type", "unknown")
                var_types[vtype] = var_types.get(vtype, 0) + 1

        if var_types:
            fig.add_trace(
                go.Pie(labels=list(var_types.keys()), values=list(var_types.values()), name="Types"),
                row=1, col=1
            )

        # 2. Matrix sizes bar chart
        matrix_names = list(matrices.keys())
        matrix_sizes = [matrices[name].size for name in matrix_names]

        if matrix_names:
            fig.add_trace(
                go.Bar(x=matrix_names, y=matrix_sizes, name="Matrix Sizes"),
                row=1, col=2
            )

        # 3. Network graph (simplified scatter)
        if connections:
            # Extract connection endpoints
            x_coords = []
            y_coords = []
            labels = []

            for i, conn in enumerate(connections[:20]):  # Limit to 20 for performance
                if isinstance(conn, dict):
                    source = str(conn.get("source_variables", [conn.get("source", "")])[0] if conn.get("source_variables") else conn.get("source", ""))
                    target = str(conn.get("target_variables", [conn.get("target", "")])[0] if conn.get("target_variables") else conn.get("target", ""))
                    x_coords.append(i % 5)
                    y_coords.append(i // 5)
                    labels.append(f"{source}\u2192{target}")

            if x_coords:
                fig.add_trace(
                    go.Scatter(x=x_coords, y=y_coords, mode='markers+text',
                             text=labels, textposition="middle center",
                             name="Connections"),
                    row=2, col=1
                )

        # 4. Statistics table
        stats_data = {
            "Metric": ["Variables", "Parameters", "Connections", "Matrices"],
            "Count": [len(variables), len(parameters), len(connections), len(matrices)]
        }

        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_data.keys())),
                cells=dict(values=[stats_data[k] for k in stats_data.keys()])
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text=f"Interactive Dashboard: {model_name}",
            height=800,
            showlegend=True
        )

        # Save as HTML
        if "html" in export_formats:
            output_file = output_dir / f"{model_name}_interactive_dashboard.html"
            fig.write_html(str(output_file))
            attempt.output_files.append(str(output_file))

        # Save as PNG if requested
        if "png" in export_formats:
            output_file = output_dir / f"{model_name}_interactive_dashboard.png"
            fig.write_image(str(output_file), width=1200, height=800)
            attempt.output_files.append(str(output_file))

        attempt.status = "success"

    except Exception as e:
        logger.error(f"Failed to generate interactive dashboard for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt
