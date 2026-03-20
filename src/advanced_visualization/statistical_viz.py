"""
Statistical and analytical visualization functions for advanced visualization.

Provides statistical plots, matrix correlation heatmaps, POMDP transition
analysis, policy visualization, and interactive Plotly dashboards.

Extracted from processor.py for maintainability.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

_module_logger = logging.getLogger(__name__)

from ._shared import (
    AdvancedVisualizationAttempt,
    MATPLOTLIB_AVAILABLE,
    SEABORN_AVAILABLE,
    np,
    plt,
    sns,
)

try:
    from visualization.matrix_visualizer import MatrixVisualizer as _MatrixVisualizer
except ImportError:
    try:
        from src.visualization.matrix_visualizer import MatrixVisualizer as _MatrixVisualizer
    except ImportError:
        _MatrixVisualizer = None


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
                        except (ValueError, TypeError) as e:
                            logger.debug("Skipping non-numeric matrix data: %s", e)

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

        if _MatrixVisualizer is None:
            attempt.status = "failed"
            attempt.error_message = "MatrixVisualizer not available"
            return attempt
        mv = _MatrixVisualizer()

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

        # Compute correlation matrix (suppress warnings from constant columns)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_matrix = np.corrcoef(matrix_vectors)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

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


# Re-exported from interactive_viz — canonical definition lives there
from .interactive_viz import _generate_interactive_plotly_dashboard  # noqa: E402
