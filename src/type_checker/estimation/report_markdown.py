"""
Markdown report generation for the resource estimator.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union


def generate_markdown_report(results: Dict[str, Any], output_path: Path, actual_project_root: Optional[Path] = None) -> str:
    """Generate a markdown report of the resource estimates."""
    
    total_files = len(results)
    valid_results = [r for r in results.values() if "error" not in r and r["memory_estimate"] is not None]

    if valid_results:
        avg_memory = sum(r["memory_estimate"] for r in valid_results) / len(valid_results)
        avg_inference = sum(r["inference_estimate"] for r in valid_results) / len(valid_results)
        avg_storage = sum(r["storage_estimate"] for r in valid_results) / len(valid_results)

        report_content = ["# GNN Resource Estimation Report", ""]
        report_content.append(f"Analyzed {total_files} files")
        report_content.append(f"Average Memory Usage: {avg_memory:.2f} KB")
        report_content.append(f"Average Inference Time: {avg_inference:.2f} units")
        report_content.append(f"Average Storage: {avg_storage:.2f} KB")
        report_content.append("")
    else:
        report_content = ["# GNN Resource Estimation Report", ""]
        report_content.append(f"Analyzed {total_files} files, but no valid results were obtained.")
        report_content.append("Check for errors in the analysis.")
        report_content.append("")

    for file_path_str, res in results.items():
        if res.get("error"):
            report_content.append(f"## {Path(file_path_str).name}")
            report_content.append(f"Path: {file_path_str}")
            report_content.append(f"Error: {res['error']}")
            report_content.append("")
            continue

        file_path_obj = Path(file_path_str).resolve()
        display_path = file_path_str
        if actual_project_root:
            try:
                display_path = str(file_path_obj.relative_to(actual_project_root))
            except ValueError:
                display_path = file_path_obj.name

        report_content.append(f"## {file_path_obj.name}")
        report_content.append(f"Path: {display_path}")
        report_content.append(f"Memory Estimate: {res['memory_estimate']:.2f} KB")
        report_content.append(f"Inference Estimate: {res['inference_estimate']:.2f} units")
        report_content.append(f"Storage Estimate: {res['storage_estimate']:.2f} KB")
        report_content.append("")

        report_content.append("### Model Info")
        for key, value in res["model_info"].items():
            report_content.append(f"- {key}: {value}")

        report_content.append("")

        report_content.append("### Complexity Metrics")
        for key, value in res["complexity"].items():
            if isinstance(value, (int, float)):
                report_content.append(f"- {key}: {value:.4f}")
            else:
                report_content.append(f"- {key}: {value}")

        report_content.append("")

    report_content.append("# Metric Definitions")
    report_content.append("")
    report_content.append("## General Metrics")
    report_content.append("- **Memory Estimate (KB):** Estimated RAM required to hold the model's variables and data structures in memory. Calculated based on variable dimensions and data types (e.g., float: 4 bytes, int: 4 bytes).")
    report_content.append("- **Inference Estimate (units):** A relative, abstract measure of computational cost for a single inference pass. It is derived from factors like model type (Static, Dynamic, Hierarchical), the number and type of variables, the complexity of connections (edges), and the operations defined in equations. Higher values indicate a more computationally intensive model. These units are not tied to a specific hardware time (e.g., milliseconds) but allow for comparison between different GNN models.")
    report_content.append("- **Storage Estimate (KB):** Estimated disk space required to store the model file. This includes the memory footprint of the data plus overhead for the GNN textual representation, metadata, comments, and equations.")
    report_content.append("")
    report_content.append("## Complexity Metrics (scores are generally relative; higher often means more complex)")
    report_content.append("- **state_space_complexity:** Logarithmic measure of the total dimensionality of all variables (sum of the product of dimensions for each variable). Represents the model's theoretical information capacity or the size of its state space.")
    report_content.append("- **graph_density:** Ratio of actual edges to the maximum possible edges in the model graph. A value of 0 indicates no connections, while 1 would mean a fully connected graph. Measures how interconnected the variables are.")
    report_content.append("- **avg_in_degree:** Average number of incoming connections (edges) per variable.")
    report_content.append("- **avg_out_degree:** Average number of outgoing connections (edges) per variable.")
    report_content.append("- **max_in_degree:** Maximum number of incoming connections for any single variable in the model.")
    report_content.append("- **max_out_degree:** Maximum number of outgoing connections for any single variable in the model.")
    report_content.append("- **cyclic_complexity:** A score indicating the presence and extent of cyclic patterns or feedback loops in the graph. Approximated based on the ratio of edges to variables; higher values suggest more complex recurrent interactions.")
    report_content.append("- **temporal_complexity:** Proportion of edges that involve time dependencies (e.g., connecting a variable at time `t` to one at `t+1`). Indicates the degree to which the model's behavior depends on past states or sequences.")
    report_content.append("- **equation_complexity:** A measure based on the average length, number, and types of mathematical operators (e.g., +, *, log, softmax) used in the model's equations. Higher values suggest more intricate mathematical relationships between variables.")
    report_content.append("- **overall_complexity:** A weighted composite score (typically scaled, e.g., 0-10) that combines state space size, graph structure (density, cyclicity), temporal aspects, and equation complexity to provide a single, holistic measure of the model's intricacy.")
    report_content.append("")

    report = "\n".join(report_content)

    # Save text report
    report_path = output_path / "resource_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
        
    return report
