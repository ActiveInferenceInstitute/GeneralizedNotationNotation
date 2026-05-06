"""
GNN Resource Estimator core engine.

Analyzes GNN models and estimates computational resources needed for:
- Memory usage
- Inference time
- Storage requirements
"""

import json
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional, Union

from .strategies import (
    calculate_complexity as _calc_complexity,
    detailed_memory_breakdown as _est_memory_breakdown,
    estimate_batched_inference as _est_batched_inference,
    estimate_flops as _est_flops,
    estimate_inference as _est_inference,
    estimate_inference_time as _est_inference_time,
    estimate_matrix_operation_costs as _est_matrix_ops,
    estimate_memory as _est_memory,
    estimate_model_overhead as _est_model_overhead,
    estimate_storage as _est_storage,
)
from .report_html import generate_html_report
from .report_markdown import generate_markdown_report

logger = logging.getLogger(__name__)


class GNNResourceEstimator:
    """Estimates computational resources required for GNN models."""

    # Default computational cost factors (realistic units)
    MEMORY_FACTORS = MappingProxyType({
        'float': 4,    # bytes per float (single precision)
        'double': 8,   # bytes per double (double precision)
        'int': 4,      # bytes per int (32-bit)
        'long': 8,     # bytes per long (64-bit)
        'bool': 1,     # bytes per bool
        'string': 16,  # average bytes per string reference
        'categorical': 4  # bytes per category (int encoding)
    })

    # Operation cost factors in FLOPS
    OPERATION_COSTS = MappingProxyType({
        'matrix_multiply': 2,  # 2 FLOPS per element (multiply and add)
        'scalar_multiply': 1,  # 1 FLOP per element
        'addition': 1,        # 1 FLOP per element
        'division': 4,        # ~4 FLOPS per element
        'exp': 20,            # ~20 FLOPS for exponential
        'log': 20,            # ~20 FLOPS for logarithm
        'softmax': 30,        # ~30 FLOPS per element (exp, sum, div)
        'sigmoid': 25,        # ~25 FLOPS per element
        'tanh': 30            # ~30 FLOPS per element
    })

    # Inference speed factors (relative to static models)
    INFERENCE_FACTORS = MappingProxyType({
        'Static': 1.0,          # Base reference
        'Dynamic': 2.5,         # Dynamic models ~2.5x more expensive
        'Hierarchical': 3.5,    # Hierarchical models ~3.5x more expensive
        'float': 1.0,           # Base reference
        'double': 1.8,          # Double precision ~1.8x slower
        'int': 0.8,             # Integers ~0.8x of float cost
        'bool': 0.5,            # Booleans ~0.5x of float cost
        'string': 1.2,          # String ops ~1.2x of float cost
        'categorical': 1.1      # Categorical ~1.1x of float cost
    })

    # Hardware specifications (representative values)
    HARDWARE_SPECS = MappingProxyType({
        'cpu_flops_per_second': 50e9,   # 50 GFLOPS for typical CPU
        'memory_bandwidth': 25e9,       # 25 GB/s memory bandwidth
        'disk_read_speed': 500e6,       # 500 MB/s disk read
        'disk_write_speed': 450e6,      # 450 MB/s disk write
    })

    def __init__(self, type_check_data: Optional[str] = None):
        """
        Initialize the resource estimator.
        
        Args:
            type_check_data: Path to JSON data from type checker
        """
        self.results = {}
        self.detailed_metrics = {}

        if type_check_data:
            try:
                with open(type_check_data, 'r') as f:
                    self.type_check_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load type check data: {e}")
                self.type_check_data = None
        else:
            self.type_check_data = None

    def estimate_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Estimate resources for a single GNN file.
        
        Args:
            file_path: Path to GNN file
            
        Returns:
            Dictionary with resource estimates
        """
        from type_checker.checking import extract_gnn_dimensions
        
        try:
            with open(file_path, 'r') as f:
                content_str = f.read()
            
            # For backward compatibility, simulate parser dictionary
            variables_with_dims = extract_gnn_dimensions(content_str)
            vars_map = {}
            for k, v in variables_with_dims.items():
                vars_map[k] = {"dimensions": v, "type": "float"}
                
            import re
            directed = re.findall(r'(\w+)\s*>\s*(\w+)', content_str)
            undirected = re.findall(r'(\w+)\s*-\s*(\w+)', content_str)
            edges = [{"source": u, "target": v, "type": "directed"} for u, v in directed]
            edges.extend([{"source": u, "target": v, "type": "undirected"} for u, v in undirected])
            
            equations = "\n".join([f"{u}={v}" for u, v in re.findall(r'(\w+)\s*=\s*(.+)', content_str)])
            time_spec = 'Dynamic' if 't' in content_str else 'Static'
            
            content_dict = {
                'Variables': vars_map,
                'Edges': edges,
                'Equations': equations,
                'Time': time_spec,
                'ModelName': Path(file_path).stem
            }
            
            return self._analyze_model(content_dict, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            return {
                "file": file_path,
                "error": str(e),
                "memory_estimate": None,
                "inference_estimate": None,
                "storage_estimate": None
            }

    def estimate_from_directory(self, dir_path: str, recursive: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Estimate resources for all GNN files in a directory.
        
        Args:
            dir_path: Path to directory with GNN files
            recursive: Whether to recursively process subdirectories
            
        Returns:
            Dictionary mapping file paths to resource estimates
        """
        path = Path(dir_path)
        results = {}

        pattern = "**/*.md" if recursive else "*.md"

        for file_path in path.glob(pattern):
            file_str = str(file_path)
            results[file_str] = self.estimate_from_file(file_str)

        self.results = results
        return results

    def _analyze_model(self, content: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Analyze a GNN model and estimate resources.
        
        Args:
            content: Parsed GNN content
            file_path: Path to source file
            
        Returns:
            Dictionary with resource estimates
        """
        variables = content.get('Variables', {})
        time_spec = content.get('Time', 'Static').split('\n')[0].strip()
        edges = content.get('Edges', [])
        equations = content.get('Equations', '')

        model_name = content.get('ModelName', os.path.basename(file_path))

        is_hierarchical = any('hierarchical' in key.lower() for key in content)
        if is_hierarchical:
            model_type = 'Hierarchical'
        else:
            model_type = time_spec

        memory_estimate = _est_memory(variables, self.MEMORY_FACTORS)
        inference_estimate = _est_inference(variables, model_type, edges, equations, self.INFERENCE_FACTORS)
        storage_estimate = _est_storage(variables, edges, equations, self.MEMORY_FACTORS)

        flops_estimate = _est_flops(variables, edges, equations, model_type, self.OPERATION_COSTS)
        inference_time_estimate = _est_inference_time(flops_estimate, self.HARDWARE_SPECS)
        batched_inference_estimate = _est_batched_inference(variables, model_type, flops_estimate, self.HARDWARE_SPECS)
        model_overhead = _est_model_overhead(variables, edges, equations)

        matrix_operation_costs = _est_matrix_ops(variables, edges, equations, self.OPERATION_COSTS)
        memory_breakdown = _est_memory_breakdown(variables, self.MEMORY_FACTORS)
        complexity = _calc_complexity(variables, edges, equations)

        self.detailed_metrics[file_path] = {
            "flops_estimate": flops_estimate,
            "inference_time_estimate": inference_time_estimate,
            "batched_inference_estimate": batched_inference_estimate,
            "model_overhead": model_overhead,
            "matrix_operation_costs": matrix_operation_costs,
            "memory_breakdown": memory_breakdown,
            "model_type": model_type
        }

        return {
            "file": file_path,
            "model_name": model_name,
            "memory_estimate": memory_estimate,
            "inference_estimate": inference_estimate,
            "storage_estimate": storage_estimate,
            "flops_estimate": flops_estimate,
            "inference_time_estimate": inference_time_estimate,
            "batched_inference_estimate": batched_inference_estimate,
            "model_overhead": model_overhead,
            "complexity": complexity,
            "model_info": {
                "variables_count": len(variables),
                "edges_count": len(edges),
                "time_spec": time_spec,
                "equation_count": len(equations.split('\n'))
            }
        }

    def generate_html_report(self, output_dir: Optional[str] = None) -> str:
        """Generate HTML report."""
        output_path = output_dir if output_dir else "resource_estimates"
        return generate_html_report(self.results, self.detailed_metrics, output_path)

    def generate_report(self, output_dir: Optional[str] = None, project_root_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a markdown report and JSON data."""
        if not self.results:
            return "No GNN models processed for resource estimation."

        output_path = Path(output_dir if output_dir else "resource_estimates").resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        actual_project_root = Path(project_root_path).resolve() if project_root_path else None

        # Save JSON data
        json_path = output_path / "resource_data.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Generate HTML
        self.generate_html_report(str(output_path))
            
        # Generate Markdown
        return generate_markdown_report(self.results, output_path, actual_project_root)
