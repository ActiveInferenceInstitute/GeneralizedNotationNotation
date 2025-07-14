#!/usr/bin/env python3
"""
GNN Resource Estimator

Analyzes GNN models and estimates computational resources needed for:
- Memory usage
- Inference time
- Storage requirements
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


class GNNResourceEstimator:
    """
    Estimates computational resources required for GNN models.
    """
    
    # Default computational cost factors (realistic units)
    MEMORY_FACTORS = {
        'float': 4,    # bytes per float (single precision)
        'double': 8,   # bytes per double (double precision)
        'int': 4,      # bytes per int (32-bit)
        'long': 8,     # bytes per long (64-bit)
        'bool': 1,     # bytes per bool
        'string': 16,  # average bytes per string reference
        'categorical': 4  # bytes per category (int encoding)
    }
    
    # Operation cost factors in FLOPS
    OPERATION_COSTS = {
        'matrix_multiply': 2,  # 2 FLOPS per element (multiply and add)
        'scalar_multiply': 1,  # 1 FLOP per element
        'addition': 1,        # 1 FLOP per element
        'division': 4,        # ~4 FLOPS per element
        'exp': 20,            # ~20 FLOPS for exponential
        'log': 20,            # ~20 FLOPS for logarithm
        'softmax': 30,        # ~30 FLOPS per element (exp, sum, div)
        'sigmoid': 25,        # ~25 FLOPS per element
        'tanh': 30            # ~30 FLOPS per element
    }
    
    # Inference speed factors (relative to static models)
    INFERENCE_FACTORS = {
        'Static': 1.0,          # Base reference
        'Dynamic': 2.5,         # Dynamic models ~2.5x more expensive
        'Hierarchical': 3.5,    # Hierarchical models ~3.5x more expensive
        'float': 1.0,           # Base reference
        'double': 1.8,          # Double precision ~1.8x slower
        'int': 0.8,             # Integers ~0.8x of float cost
        'bool': 0.5,            # Booleans ~0.5x of float cost
        'string': 1.2,          # String ops ~1.2x of float cost
        'categorical': 1.1      # Categorical ~1.1x of float cost
    }
    
    # Hardware specifications (representative values)
    HARDWARE_SPECS = {
        'cpu_flops_per_second': 50e9,   # 50 GFLOPS for typical CPU
        'memory_bandwidth': 25e9,       # 25 GB/s memory bandwidth
        'disk_read_speed': 500e6,       # 500 MB/s disk read
        'disk_write_speed': 450e6,      # 450 MB/s disk write
    }
    
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
                print(f"Warning: Could not load type check data: {e}")
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
        from visualization.parser import GNNParser
        
        parser = GNNParser()
        
        try:
            # Parse the file
            content = parser.parse_file(file_path)
            return self._analyze_model(content, file_path)
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
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
        
        # Define pattern for GNN files
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
        # Extract key model information
        variables = content.get('Variables', {})
        time_spec = content.get('Time', 'Static').split('\n')[0].strip()
        edges = content.get('Edges', [])
        equations = content.get('Equations', '')
        
        # Extract model name
        model_name = content.get('ModelName', os.path.basename(file_path))
        
        # Determine if model is hierarchical
        is_hierarchical = any('hierarchical' in key.lower() for key in content.keys())
        if is_hierarchical:
            model_type = 'Hierarchical'
        else:
            model_type = time_spec
        
        # Basic estimates
        memory_estimate = self._estimate_memory(variables)
        inference_estimate = self._estimate_inference(variables, model_type, edges, equations)
        storage_estimate = self._estimate_storage(variables, edges, equations)
        
        # Advanced estimates
        flops_estimate = self._estimate_flops(variables, edges, equations, model_type)
        inference_time_estimate = self._estimate_inference_time(flops_estimate)
        batched_inference_estimate = self._estimate_batched_inference(variables, model_type, flops_estimate)
        model_overhead = self._estimate_model_overhead(variables, edges, equations)
        
        # More detailed matrix operation estimates
        matrix_operation_costs = self._estimate_matrix_operation_costs(variables, edges, equations)
        
        # Detailed memory breakdowns
        memory_breakdown = self._detailed_memory_breakdown(variables)
        
        # Calculate complexity metrics
        complexity = self._calculate_complexity(variables, edges, equations)
        
        # Store detailed metrics for HTML report
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
    
    def _estimate_memory(self, variables: Dict[str, Any]) -> float:
        """
        Estimate memory requirements based on variables.
        
        Args:
            variables: Dictionary of model variables
            
        Returns:
            Memory estimate in KB
        """
        total_memory = 0.0
        
        for var_name, var_info in variables.items():
            var_type = var_info.get('type', 'float')
            dims = var_info.get('dimensions', [1])
            
            # Calculate size of this variable
            size_factor = self.MEMORY_FACTORS.get(var_type, self.MEMORY_FACTORS['float'])
            
            # Process dimensions with caution - handle symbolic dimensions
            try:
                dimension_values = []
                for d in dims:
                    if isinstance(d, (int, float)):
                        dimension_values.append(d)
                    elif isinstance(d, str) and d.isdigit():
                        dimension_values.append(int(d))
                    elif isinstance(d, str) and 'len' in d and 'π' in d:
                        # Approximate the size for dynamic dimensions referencing policy length
                        dimension_values.append(3)  # Reasonable default for policy length
                    elif isinstance(d, str) and d.startswith('='):
                        # Handle '=[2]' or '=[2,1]' format by extracting numbers
                        import re
                        matches = re.findall(r'\d+', d)
                        if matches:
                            dimension_values.append(int(matches[0]))
                        else:
                            dimension_values.append(1)
                    else:
                        dimension_values.append(1)  # Default for unparseable dimensions
                
                total_size = size_factor * math.prod(dimension_values)
            except Exception as e:
                print(f"Warning: Error calculating size for variable {var_name}: {e}")
                # Use a default size based on variable type
                total_size = size_factor * 2  # Assume small dimensions as fallback
            
            total_memory += total_size
        
        # Convert to KB
        return total_memory / 1024.0
    
    def _detailed_memory_breakdown(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed memory breakdown by variable and type.
        
        Args:
            variables: Dictionary of model variables
            
        Returns:
            Dictionary with detailed memory breakdown
        """
        breakdown = {
            "by_variable": {},
            "by_type": {t: 0 for t in self.MEMORY_FACTORS.keys()},
            "total_bytes": 0,
            "representation_overhead": 0  # Additional overhead for representation
        }
        
        # Fixed overhead for model structure
        breakdown["representation_overhead"] = 1024  # Approx 1KB for basic model structure
        
        for var_name, var_info in variables.items():
            var_type = var_info.get('type', 'float')
            dims = var_info.get('dimensions', [1])
            
            # Calculate size of this variable
            size_factor = self.MEMORY_FACTORS.get(var_type, self.MEMORY_FACTORS['float'])
            
            # Process dimensions
            try:
                dimension_values = []
                for d in dims:
                    if isinstance(d, (int, float)):
                        dimension_values.append(d)
                    elif isinstance(d, str) and d.isdigit():
                        dimension_values.append(int(d))
                    elif isinstance(d, str) and ('len' in d or 'π' in d):
                        dimension_values.append(3)
                    elif isinstance(d, str) and d.startswith('='):
                        import re
                        matches = re.findall(r'\d+', d)
                        if matches:
                            dimension_values.append(int(matches[0]))
                        else:
                            dimension_values.append(1)
                    else:
                        dimension_values.append(1)
                
                element_count = math.prod(dimension_values)
                var_size = size_factor * element_count
                
                # Additional overhead for variable names and metadata
                var_overhead = len(var_name) + 24  # ~24 bytes overhead per variable
                
                breakdown["by_variable"][var_name] = {
                    "size_bytes": var_size,
                    "elements": element_count,
                    "dimensions": dimension_values,
                    "type": var_type,
                    "overhead_bytes": var_overhead,
                    "total_bytes": var_size + var_overhead
                }
                
                # Add to type totals
                breakdown["by_type"][var_type] = breakdown["by_type"].get(var_type, 0) + var_size
                
                # Add to total bytes
                breakdown["total_bytes"] += var_size + var_overhead
                
                # Add to representation overhead
                breakdown["representation_overhead"] += var_overhead
                
            except Exception as e:
                print(f"Warning: Error in memory breakdown for {var_name}: {e}")
        
        # Convert totals to KB for convenience
        breakdown["total_kb"] = breakdown["total_bytes"] / 1024.0
        breakdown["overhead_kb"] = breakdown["representation_overhead"] / 1024.0
        
        return breakdown
    
    def _estimate_flops(self, variables: Dict[str, Any], edges: List[Dict[str, Any]], 
                       equations: str, model_type: str) -> Dict[str, Any]:
        """
        Estimate floating-point operations (FLOPS) required for inference.
        
        Args:
            variables: Dictionary of model variables
            edges: List of edges in the model
            equations: Equations in the model
            model_type: Type of model (Static, Dynamic, Hierarchical)
            
        Returns:
            Dictionary with FLOPS estimates
        """
        flops_estimate = {
            "total_flops": 0,
            "matrix_operations": 0,
            "element_operations": 0,
            "nonlinear_operations": 0
        }
        
        # Count matrices and their dimensions
        matrices = {}
        for var_name, var_info in variables.items():
            dims = var_info.get('dimensions', [1])
            if len(dims) >= 2:  # It's a matrix
                matrices[var_name] = dims
        
        # Estimate matrix multiplication costs (dominant operation)
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            if source in matrices and target in matrices:
                source_dims = matrices[source]
                target_dims = matrices[target]
                
                # Matrix multiply cost: m×n×p operations for m×n * n×p matrices
                if len(source_dims) >= 2 and len(target_dims) >= 2:
                    try:
                        # Extract dimensions as integers when possible
                        m = int(source_dims[0]) if isinstance(source_dims[0], (int, str)) and (isinstance(source_dims[0], int) or source_dims[0].isdigit()) else 2
                        n = int(source_dims[1]) if isinstance(source_dims[1], (int, str)) and (isinstance(source_dims[1], int) or source_dims[1].isdigit()) else 2
                        p = int(target_dims[1]) if len(target_dims) > 1 and isinstance(target_dims[1], (int, str)) and (isinstance(target_dims[1], int) or target_dims[1].isdigit()) else 2
                        
                        # 2 FLOPS per element (multiply and add)
                        flops = m * n * p * self.OPERATION_COSTS['matrix_multiply']
                        flops_estimate["matrix_operations"] += flops
                        flops_estimate["total_flops"] += flops
                    except (ValueError, TypeError) as e:
                        # Default estimation if conversion fails
                        flops_estimate["matrix_operations"] += 100  # Assume small matrices
                        flops_estimate["total_flops"] += 100
        
        # Estimate element-wise operations from equations
        eq_lines = equations.split('\n')
        for line in eq_lines:
            element_ops = 0
            nonlinear_ops = 0
            
            # Count arithmetic operations
            element_ops += line.count('+') * self.OPERATION_COSTS['addition']
            element_ops += line.count('*') * self.OPERATION_COSTS['scalar_multiply']
            element_ops += line.count('/') * self.OPERATION_COSTS['division']
            
            # Count nonlinear operations (approximate)
            nonlinear_ops += line.count('exp') * self.OPERATION_COSTS['exp']
            nonlinear_ops += line.count('log') * self.OPERATION_COSTS['log']
            nonlinear_ops += line.count('softmax') * self.OPERATION_COSTS['softmax']
            nonlinear_ops += line.count('sigma') * self.OPERATION_COSTS['sigmoid']
            
            flops_estimate["element_operations"] += element_ops
            flops_estimate["nonlinear_operations"] += nonlinear_ops
            flops_estimate["total_flops"] += element_ops + nonlinear_ops
        
        # Apply model type multiplier
        if model_type == 'Dynamic':
            flops_estimate["total_flops"] *= 2.5  # Dynamic models more expensive
        elif model_type == 'Hierarchical':
            flops_estimate["total_flops"] *= 3.5  # Hierarchical models most expensive
        
        # If no specific operations detected, estimate based on variable count and model type
        if flops_estimate["total_flops"] == 0:
            base_flops = len(variables) * 20  # 20 FLOPS per variable as baseline
            if model_type == 'Static':
                flops_estimate["total_flops"] = base_flops
            elif model_type == 'Dynamic':
                flops_estimate["total_flops"] = base_flops * 2.5
            else:  # Hierarchical
                flops_estimate["total_flops"] = base_flops * 3.5
        
        return flops_estimate
    
    def _estimate_inference_time(self, flops_estimate: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate inference time based on FLOPS and hardware specs.
        
        Args:
            flops_estimate: Dictionary with FLOPS estimates
            
        Returns:
            Dictionary with inference time estimates in various units
        """
        total_flops = flops_estimate["total_flops"]
        
        # Calculate time based on hardware specs
        cpu_time_seconds = total_flops / self.HARDWARE_SPECS["cpu_flops_per_second"]
        
        return {
            "cpu_time_seconds": cpu_time_seconds,
            "cpu_time_ms": cpu_time_seconds * 1000,
            "cpu_time_us": cpu_time_seconds * 1_000_000
        }
    
    def _estimate_batched_inference(self, variables: Dict[str, Any], model_type: str, 
                                   flops_estimate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate batched inference performance.
        
        Args:
            variables: Dictionary of model variables
            model_type: Type of model (Static, Dynamic, Hierarchical)
            flops_estimate: Dictionary with FLOPS estimates
            
        Returns:
            Dictionary with batched inference estimates
        """
        total_flops = flops_estimate["total_flops"]
        
        # Batch sizes to estimate
        batch_sizes = [1, 8, 32, 128, 512]
        
        # Estimate batch throughput
        batch_estimates = {}
        for batch_size in batch_sizes:
            # Batched FLOPS (not perfectly linear due to overhead)
            if batch_size == 1:
                batch_flops = total_flops
            else:
                # Diminishing returns with larger batches
                scale_factor = 0.7 + 0.3 / math.log2(batch_size + 1)
                batch_flops = total_flops * batch_size * scale_factor
            
            # Estimate time for batched inference
            time_seconds = batch_flops / self.HARDWARE_SPECS["cpu_flops_per_second"]
            
            # Throughput in samples per second
            throughput = batch_size / time_seconds if time_seconds > 0 else 0
            
            batch_estimates[f"batch_{batch_size}"] = {
                "flops": batch_flops,
                "time_seconds": time_seconds,
                "throughput_per_second": throughput
            }
        
        return batch_estimates

    def _estimate_matrix_operation_costs(self, variables: Dict[str, Any], edges: List[Dict[str, Any]], 
                                      equations: str) -> Dict[str, Any]:
        """
        Provide detailed estimates of matrix operation costs.
        
        Args:
            variables: Dictionary of model variables
            edges: List of edges in the model
            equations: Equations in the model
            
        Returns:
            Dictionary with matrix operation costs
        """
        operation_costs = {
            "matrix_multiply": [],
            "matrix_transpose": [],
            "matrix_inversion": [],
            "element_wise": [],
            "total_matrix_flops": 0
        }
        
        # Identify matrices and their dimensions
        matrices = {}
        for var_name, var_info in variables.items():
            dims = var_info.get('dimensions', [1])
            if len(dims) >= 2:  # It's a matrix
                # Convert dimensions to integers when possible
                int_dims = []
                for d in dims:
                    if isinstance(d, int):
                        int_dims.append(d)
                    elif isinstance(d, str) and d.isdigit():
                        int_dims.append(int(d))
                    else:
                        int_dims.append(2)  # Default dimension
                
                matrices[var_name] = int_dims
        
        # Analyze edge connections for matrix operations
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            if source in matrices and target in matrices:
                source_dims = matrices[source]
                target_dims = matrices[target]
                
                # Matrix multiplication cost
                if len(source_dims) >= 2 and len(target_dims) >= 2:
                    m = source_dims[0]
                    n = source_dims[1] if len(source_dims) > 1 else 1
                    p = target_dims[1] if len(target_dims) > 1 else 1
                    
                    flops = m * n * p * self.OPERATION_COSTS['matrix_multiply']
                    
                    operation_costs["matrix_multiply"].append({
                        "operation": f"{source} × {target}",
                        "dimensions": f"{m}×{n} * {n}×{p}",
                        "flops": flops
                    })
                    
                    operation_costs["total_matrix_flops"] += flops
        
        # Analyze equations for matrix operations
        eq_lines = equations.split('\n')
        for line in eq_lines:
            # Look for matrix transpose operations (A^T or transpose(A))
            if '^T' in line or 'transpose' in line:
                for matrix_name in matrices:
                    if matrix_name in line and (f"{matrix_name}^T" in line or f"transpose({matrix_name})" in line):
                        dims = matrices[matrix_name]
                        m = dims[0]
                        n = dims[1] if len(dims) > 1 else 1
                        
                        # Transpose costs m*n operations
                        flops = m * n
                        
                        operation_costs["matrix_transpose"].append({
                            "operation": f"{matrix_name}^T",
                            "dimensions": f"{m}×{n}",
                            "flops": flops
                        })
                        
                        operation_costs["total_matrix_flops"] += flops
            
            # Look for matrix inversion operations (A^-1 or inv(A))
            if '^-1' in line or 'inv(' in line:
                for matrix_name in matrices:
                    if matrix_name in line and (f"{matrix_name}^-1" in line or f"inv({matrix_name})" in line):
                        dims = matrices[matrix_name]
                        n = dims[0]  # Assume square matrix for inversion
                        
                        # Inversion costs approximately n^3 operations
                        flops = n**3
                        
                        operation_costs["matrix_inversion"].append({
                            "operation": f"{matrix_name}^-1",
                            "dimensions": f"{n}×{n}",
                            "flops": flops
                        })
                        
                        operation_costs["total_matrix_flops"] += flops
        
        return operation_costs

    def _estimate_model_overhead(self, variables: Dict[str, Any], edges: List[Dict[str, Any]], 
                               equations: str) -> Dict[str, Any]:
        """
        Estimate model overhead including compile-time and optimization costs.
        
        Args:
            variables: Dictionary of model variables
            edges: List of edges in the model
            equations: Equations in the model
            
        Returns:
            Dictionary with model overhead estimates
        """
        overhead = {
            "compilation_ms": 0,
            "optimization_ms": 0,
            "memory_overhead_kb": 0
        }
        
        # Estimate compilation time based on model complexity
        var_count = len(variables)
        edge_count = len(edges)
        eq_count = len(equations.split('\n'))
        
        # Simple heuristic: ~10ms base + 2ms per variable + 1ms per edge + 5ms per equation
        compilation_ms = 10 + (var_count * 2) + (edge_count * 1) + (eq_count * 5)
        overhead["compilation_ms"] = compilation_ms
        
        # Optimization costs roughly scale with variables^2
        optimization_ms = 20 + (var_count**2 * 0.5)
        overhead["optimization_ms"] = optimization_ms
        
        # Memory overhead: ~1KB base + 50 bytes per variable + 30 bytes per edge + 100 bytes per equation
        memory_overhead_bytes = 1024 + (var_count * 50) + (edge_count * 30) + (eq_count * 100)
        overhead["memory_overhead_kb"] = memory_overhead_bytes / 1024.0
        
        return overhead
    
    def _estimate_inference(self, variables: Dict[str, Any], model_type: str, 
                            edges: List[Dict[str, Any]], equations: str) -> float:
        """
        Estimate inference time requirements based on model complexity.
        
        Args:
            variables: Dictionary of model variables
            model_type: Type of model (Static, Dynamic, Hierarchical)
            edges: List of edges in the model
            equations: Equations in the model
            
        Returns:
            Inference time estimate (arbitrary units)
        """
        # Base time depends on model type
        base_time = self.INFERENCE_FACTORS.get(model_type, self.INFERENCE_FACTORS['Static'])
        
        # Add time for variable processing - consider variable types
        var_time = 0
        for var_name, var_info in variables.items():
            var_type = var_info.get('type', 'float')
            dims = var_info.get('dimensions', [1])
            
            # Extract dimensions as integers or defaults
            try:
                # Get element count based on dimensions
                element_count = 1
                for d in dims:
                    if isinstance(d, (int, float)):
                        element_count *= int(d)
                    elif isinstance(d, str) and d.isdigit():
                        element_count *= int(d)
                    elif isinstance(d, str) and 'len' in d:
                        element_count *= 3  # Reasonable default
                    elif isinstance(d, str) and d.startswith('='):
                        import re
                        matches = re.findall(r'\d+', d)
                        if matches:
                            element_count *= int(matches[0])
                        else:
                            element_count *= 1
                    else:
                        element_count *= 1
                
                # Scale by type factor and element count
                type_factor = self.INFERENCE_FACTORS.get(var_type, self.INFERENCE_FACTORS['float'])
                var_time += type_factor * math.log2(element_count + 1)  # log scale to avoid explosion
                
            except Exception as e:
                # Use default if calculation fails
                var_time += self.INFERENCE_FACTORS.get(var_type, 1.0)
        
        # Add time for edge traversal - more for complex connections
        # Each edge represents signal propagation with potential transformations
        edge_time = 0
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            # Temporal connections cost more (Dynamic models)
            if '+' in source or '+' in target:
                edge_time += 1.0  # Temporal connection
            else:
                edge_time += 0.5  # Standard connection
        
        # Add time for equation evaluation
        equation_lines = equations.split('\n')
        equation_time = 0
        
        for line in equation_lines:
            # Basic cost per equation
            eq_cost = 2.0
            
            # Additional cost for complex operations
            if 'softmax' in line or 'sigma' in line:
                eq_cost += 1.5  # Nonlinear functions cost more
            if '^' in line:  # Power operations or matrix transposes
                eq_cost += 1.0
            if 'sum' in line or '∑' in line:  # Summation operations
                eq_cost += 1.0
            
            equation_time += eq_cost
        
        # For models with no equations, assume default complexity
        if equation_time == 0 and len(equation_lines) > 0:
            equation_time = len(equation_lines) * 2.0
        
        # Combine factors with weights
        # - Base model type is most important (40%)
        # - Variables and equations matter significantly (25% each)
        # - Edge structure has some impact (10%)
        weighted_time = (
            base_time * 4.0 +
            var_time * 2.5 +
            edge_time * 1.0 +
            equation_time * 2.5
        ) / 10.0
        
        return weighted_time * 10.0  # Scale to get reasonable units
    
    def _estimate_storage(self, variables: Dict[str, Any], edges: List[Dict[str, Any]], equations: str) -> float:
        """
        Estimate storage requirements based on model structure and size.
        
        Args:
            variables: Dictionary of model variables
            edges: List of edges in the model
            equations: Equations in the model
            
        Returns:
            Storage estimate in KB
        """
        # Memory footprint forms the base of our storage estimate
        memory_estimate = self._estimate_memory(variables)
        
        # Calculate structural overhead
        # Model structure, format overhead, metadata, descriptions: ~1KB base + per-item costs
        structural_overhead_kb = 1.0  
        
        # Add overhead for variable names, descriptions, and metadata
        var_overhead_kb = 0.0
        for var_name, var_info in variables.items():
            # Each variable has name, type, dimension info, comments
            var_desc_length = len(var_info.get('comment', ''))
            var_overhead_kb += (len(var_name) + 24 + var_desc_length) / 1024.0
        
        # Add overhead for edge definitions
        edge_overhead_kb = len(edges) * 0.1  # ~100 bytes per edge definition
        
        # Add overhead for equations (consider actual text length)
        equation_overhead_kb = len(equations) * 0.001  # ~1 byte per character
        
        # Textual representation adds overhead on top of binary storage
        format_overhead_kb = 0.5  # GNN format markup, spaces, structure
        
        # Combine all storage components
        total_storage_kb = (
            memory_estimate * 1.2 +  # Binary data storage with padding
            structural_overhead_kb +
            var_overhead_kb +
            edge_overhead_kb +
            equation_overhead_kb +
            format_overhead_kb
        )
        
        # Make sure we don't get unreasonably low estimates
        return max(total_storage_kb, 1.0)
    
    def _calculate_complexity(self, variables: Dict[str, Any], edges: List[Dict[str, Any]], equations: str) -> Dict[str, float]:
        """
        Calculate detailed complexity metrics for the model.
        
        Args:
            variables: Dictionary of model variables
            edges: List of edges in the model
            equations: Equations in the model
            
        Returns:
            Dictionary with complexity metrics
        """
        # Get total dimensionality (state space complexity)
        total_dims = 0
        max_dim = 0
        for var_info in variables.values():
            dims = var_info.get('dimensions', [1])
            # Convert dimensions to integers when possible
            int_dims = []
            for d in dims:
                if isinstance(d, int):
                    int_dims.append(d)
                elif isinstance(d, str) and d.isdigit():
                    int_dims.append(int(d))
                else:
                    int_dims.append(2)  # Default dimension size
            
            # Sum up total dimensionality
            dim_size = math.prod(int_dims)
            total_dims += dim_size
            max_dim = max(max_dim, dim_size)
        
        # Calculate graph metrics
        var_count = len(variables)
        edge_count = len(edges)
        
        # Topological complexity
        # Graph density: 0 (unconnected) to 1 (fully connected)
        density = 0.0
        if var_count > 1:
            max_possible_edges = var_count * (var_count - 1)  # Directed graph
            density = edge_count / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate connectivity patterns
        in_degree = {}
        out_degree = {}
        for edge in edges:
            source = edge.get('source', '').split('+')[0]  # Remove time indices
            target = edge.get('target', '').split('+')[0]
            
            out_degree[source] = out_degree.get(source, 0) + 1
            in_degree[target] = in_degree.get(target, 0) + 1
        
        # Average degrees
        avg_in_degree = sum(in_degree.values()) / max(len(in_degree), 1)
        avg_out_degree = sum(out_degree.values()) / max(len(out_degree), 1)
        
        # Max degrees
        max_in_degree = max(in_degree.values()) if in_degree else 0
        max_out_degree = max(out_degree.values()) if out_degree else 0
        
        # Cyclic complexity - check for cyclic patterns (excluding self-loops)
        # Simple approximation: higher connectivity often means more cycles
        cyclic_score = 0
        if edge_count > var_count:
            cyclic_score = (edge_count - var_count) / max(var_count, 1)
        
        # Temporal complexity (look for time indices in edges)
        temporal_edges = 0
        for edge in edges:
            if '+' in edge.get('source', '') or '+' in edge.get('target', ''):
                temporal_edges += 1
        
        temporal_complexity = temporal_edges / max(edge_count, 1)
        
        # Equation complexity
        eq_lines = equations.split('\n')
        avg_eq_length = sum(len(line) for line in eq_lines) / max(len(eq_lines), 1)
        
        # Count operators in equations as a measure of complexity
        operators = 0
        for line in eq_lines:
            operators += line.count('+') + line.count('-') + line.count('*') + line.count('/') + line.count('^')
        
        # Higher-order operators indicate more complexity
        higher_order_ops = 0
        for line in eq_lines:
            higher_order_ops += line.count('sum') + line.count('prod') + line.count('log') + line.count('exp')
            higher_order_ops += line.count('softmax') + line.count('tanh') + line.count('sigma')
        
        # Combined equation complexity
        equation_complexity = 0
        if eq_lines:
            equation_complexity = (avg_eq_length + operators + 3*higher_order_ops) / len(eq_lines)
        
        # State space complexity (measure of information capacity)
        state_space_complexity = math.log2(total_dims + 1) if total_dims > 0 else 0
        
        # Overall complexity combines several factors:
        # - State space (information capacity)
        # - Connectivity (graph structure)
        # - Temporal aspects (time dependencies)
        # - Algorithmic complexity (equations)
        overall_complexity = (
            state_space_complexity * 0.25 +
            (density + cyclic_score) * 0.25 +
            temporal_complexity * 0.2 +
            equation_complexity * 0.3
        )
        
        # Scale to a reasonable range (0-10)
        overall_complexity = min(10, overall_complexity * 2)
        
        return {
            "state_space_complexity": state_space_complexity,
            "graph_density": density,
            "avg_in_degree": avg_in_degree,
            "avg_out_degree": avg_out_degree,
            "max_in_degree": max_in_degree,
            "max_out_degree": max_out_degree,
            "cyclic_complexity": cyclic_score,
            "temporal_complexity": temporal_complexity,
            "equation_complexity": equation_complexity,
            "overall_complexity": overall_complexity,
            "variable_count": var_count,
            "edge_count": edge_count,
            "total_state_space_dim": total_dims,
            "max_variable_dim": max_dim
        }
    
    def generate_html_report(self, output_dir: Optional[str] = None) -> str:
        """
        Generate a comprehensive HTML report with visualizations and detailed explanations.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Path to the generated HTML report
        """
        import json
        from datetime import datetime
        
        if not self.results:
            return "No results to report. Run estimation first."
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("output/type_checker/resources")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations for HTML embedding
        vis_path = output_path / "html_vis"
        vis_path.mkdir(exist_ok=True)
        self._generate_visualizations_for_html(vis_path)
        
        # Create model comparison arrays for visualizations
        models = []
        memory_values = []
        inference_values = []
        storage_values = []
        flops_values = []
        
        for file_path, result in sorted(self.results.items()):
            if "error" not in result:
                models.append(os.path.basename(file_path).replace(".md", ""))
                memory_values.append(result["memory_estimate"])
                inference_values.append(result["inference_estimate"])
                storage_values.append(result["storage_estimate"])
                if "flops_estimate" in result:
                    flops_values.append(result["flops_estimate"]["total_flops"])
                else:
                    flops_values.append(0)
        
        # Start creating HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Resource Estimation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; margin-top: 20px; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .chart-container {{ width: 100%; height: 400px; margin-bottom: 30px; }}
        .metric-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
        .metric-box {{ flex: 1; min-width: 200px; background-color: #f8f9fa; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .model-card {{ background-color: #fff; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 15px; }}
        .caption {{ font-style: italic; color: #555; margin-top: 5px; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .resource-comparison {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .resource-comparison > div {{ flex: 1; }}
        .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; text-align: center; }}
    </style>
</head>
<body>
    <h1>GNN Resource Estimation Report</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>This report provides detailed resource estimates for {len(models)} GNN models, analyzing their memory usage, computational requirements, and storage needs.</p>
        
        <div class="metric-container">
            <div class="metric-box">
                <h3>Average Memory Usage</h3>
                <div class="metric-value">{sum(memory_values)/len(memory_values):.2f} KB</div>
                <p>RAM required to hold model in memory</p>
            </div>
            <div class="metric-box">
                <h3>Average Inference Time</h3>
                <div class="metric-value">{sum(inference_values)/len(inference_values):.2f} units</div>
                <p>Relative computational cost</p>
            </div>
            <div class="metric-box">
                <h3>Average Storage</h3>
                <div class="metric-value">{sum(storage_values)/len(storage_values):.2f} KB</div>
                <p>Disk space required to store model</p>
            </div>
        </div>
    </div>
    
    <h2>Resource Visualizations</h2>
    <p>The following visualizations provide a comparative analysis of resource requirements across different models.</p>
    
    <h3>Memory Usage Comparison</h3>
    <div class="chart-container">
        <canvas id="memoryChart"></canvas>
    </div>
    <p class="caption">
        <strong>Figure 1:</strong> Memory usage comparison showing RAM requirements (in KB) for each GNN model. 
        Memory usage is determined by the size and number of matrices and variables in the model. 
        Hierarchical models typically require more memory due to their multi-level structure.
    </p>
    
    <h3>Inference Time Comparison</h3>
    <div class="chart-container">
        <canvas id="inferenceChart"></canvas>
    </div>
    <p class="caption">
        <strong>Figure 2:</strong> Inference time comparison showing relative computational cost for each model.
        Higher values indicate more complex computations requiring more CPU/GPU time.
        Dynamic models with temporal dependencies typically have higher inference costs than static models.
    </p>
    
    <h3>Storage Requirements Comparison</h3>
    <div class="chart-container">
        <canvas id="storageChart"></canvas>
    </div>
    <p class="caption">
        <strong>Figure 3:</strong> Storage requirements showing disk space (in KB) needed to store each model.
        Storage includes the base memory requirements plus additional overhead for model structure, 
        metadata, and equation representations.
    </p>
    
    <h3>Normalized Resource Comparison</h3>
    <div class="chart-container">
        <canvas id="comparisonChart"></canvas>
    </div>
    <p class="caption">
        <strong>Figure 4:</strong> Normalized resource comparison showing relative requirements across all resource types.
        Values are normalized to the highest value in each category to allow for direct comparison.
        This visualization helps identify which resource dimension is most constraining for each model.
    </p>
"""

        # Add computational complexity section
        html_content += """
    <h2>Computational Complexity Analysis</h2>
    <p>This section breaks down the computational complexity of each model in terms of operations required and algorithmic efficiency.</p>
    
    <div class="chart-container">
        <canvas id="flopsChart"></canvas>
    </div>
    <p class="caption">
        <strong>Figure 5:</strong> Estimated floating-point operations (FLOPS) required for a single inference pass.
        FLOPS count is calculated based on matrix operations, element-wise operations, and nonlinear function evaluations
        in the model. Higher FLOPS indicate more computationally intensive models.
    </p>
    
    <h3>Matrix Operation Costs</h3>
    <p>Matrix operations typically dominate the computational cost of GNN models. The following table shows estimated costs for key operations.</p>
    
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Matrix Multiplications</th>
                <th>Element-wise Ops</th>
                <th>Nonlinear Ops</th>
                <th>Total FLOPS</th>
            </tr>
        </thead>
        <tbody>
"""

        # Add rows for each model with detailed FLOPS breakdown
        for file_path, result in sorted(self.results.items()):
            if "error" not in result and "flops_estimate" in result:
                model_name = os.path.basename(file_path).replace(".md", "")
                flops = result["flops_estimate"]
                
                html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{flops.get("matrix_operations", 0):.0f}</td>
                <td>{flops.get("element_operations", 0):.0f}</td>
                <td>{flops.get("nonlinear_operations", 0):.0f}</td>
                <td>{flops.get("total_flops", 0):.0f}</td>
            </tr>"""

        html_content += """
        </tbody>
    </table>
    <p class="caption">
        <strong>Table 1:</strong> Breakdown of computational operations by type for each model. 
        Matrix multiplications, element-wise operations, and nonlinear function evaluations contribute to the total FLOPS.
    </p>
"""

        # Add model details section
        html_content += """
    <h2>Individual Model Analysis</h2>
    <p>This section provides detailed resource profiles for each individual model.</p>
    
    <div class="model-container">
"""

        # Add cards for each model with detailed metrics
        for file_path, result in sorted(self.results.items()):
            if "error" not in result:
                model_name = os.path.basename(file_path).replace(".md", "")
                
                # Get model type
                model_type = "Static"
                if file_path in self.detailed_metrics:
                    model_type = self.detailed_metrics[file_path].get("model_type", "Static")
                
                # Get complexity metrics
                complexity = result.get("complexity", {})
                
                # Create memory breakdown
                memory_breakdown = {}
                if file_path in self.detailed_metrics:
                    memory_breakdown = self.detailed_metrics[file_path].get("memory_breakdown", {})
                
                html_content += f"""
        <div class="model-card">
            <h3>{model_name}</h3>
            <p><strong>Type:</strong> {model_type} model</p>
            
            <div class="resource-comparison">
                <div>
                    <h4>Memory Usage</h4>
                    <div class="metric-value">{result["memory_estimate"]:.2f} KB</div>
                    <p>Variables: {result["model_info"]["variables_count"]}</p>
                </div>
                <div>
                    <h4>Inference Time</h4>
                    <div class="metric-value">{result["inference_estimate"]:.2f} units</div>
                    <p>Edges: {result["model_info"]["edges_count"]}</p>
                </div>
                <div>
                    <h4>Storage</h4>
                    <div class="metric-value">{result["storage_estimate"]:.2f} KB</div>
                    <p>Equations: {result["model_info"]["equation_count"]}</p>
                </div>
            </div>
            
            <h4>Complexity Metrics</h4>
            <table>
                <tr>
                    <td>State Space Complexity</td>
                    <td>{complexity.get("state_space_complexity", 0):.2f}</td>
                    <td>Graph Density</td>
                    <td>{complexity.get("graph_density", 0):.2f}</td>
                </tr>
                <tr>
                    <td>Temporal Complexity</td>
                    <td>{complexity.get("temporal_complexity", 0):.2f}</td>
                    <td>Equation Complexity</td>
                    <td>{complexity.get("equation_complexity", 0):.2f}</td>
                </tr>
                <tr>
                    <td>Overall Complexity</td>
                    <td colspan="3">{complexity.get("overall_complexity", 0):.2f} / 10.0</td>
                </tr>
            </table>
        </div>
"""

        # Add explanation section
        html_content += """
    </div>
    
    <h2>Understanding Resource Metrics</h2>
    <p>This section explains how each resource metric is calculated and what it means for model performance.</p>
    
    <h3>Memory Usage</h3>
    <p>Memory usage estimates the amount of RAM required to hold the model in memory during inference. It accounts for:</p>
    <ul>
        <li><strong>Variable Storage:</strong> Size of all matrices and vectors in the model</li>
        <li><strong>Type Considerations:</strong> Different data types (float, int, bool) require different amounts of memory</li>
        <li><strong>Dimension Analysis:</strong> Higher-dimensional state spaces consume more memory</li>
    </ul>
    <p>The memory estimate is calculated based on the dimensions of each variable and its data type, using standard sizes (4 bytes for float/int, 1 byte for bool, etc.).</p>
    
    <h3>Inference Time</h3>
    <p>Inference time estimates the relative computational cost of running the model for a single inference pass. It considers:</p>
    <ul>
        <li><strong>Model Type:</strong> Static vs. Dynamic vs. Hierarchical architecture</li>
        <li><strong>Matrix Operations:</strong> Primarily matrix multiplications which dominate computational cost</li>
        <li><strong>Nonlinear Functions:</strong> Operations like softmax, sigmoid, or tanh which are computationally expensive</li>
        <li><strong>Connectivity:</strong> Edge structure and temporal relationships that affect computation flow</li>
    </ul>
    <p>The inference time is provided in relative units, with higher values indicating more complex computations requiring more processing time.</p>
    
    <h3>Storage Requirements</h3>
    <p>Storage requirements estimate the disk space needed to persist the model. This includes:</p>
    <ul>
        <li><strong>Base Memory Footprint:</strong> Same as memory usage</li>
        <li><strong>Format Overhead:</strong> Additional space for storing the model structure</li>
        <li><strong>Metadata:</strong> Variable names, comments, equations, etc.</li>
    </ul>
    <p>Storage is typically larger than memory usage due to the additional structural information needed to fully represent the model on disk.</p>
    
    <h3>Computational Complexity</h3>
    <p>Computational complexity provides a deeper analysis of the algorithmic efficiency of the model:</p>
    <ul>
        <li><strong>FLOPS (Floating Point Operations):</strong> Count of arithmetic operations needed for inference</li>
        <li><strong>Matrix Operation Costs:</strong> Detailed breakdown of matrix multiplication, transpose, and inversion costs</li>
        <li><strong>State Space Complexity:</strong> Measure of the model's information capacity</li>
        <li><strong>Structural Complexity:</strong> Analysis of the edge structure and connectivity patterns</li>
    </ul>
    <p>These metrics help identify bottlenecks and optimize model architecture for better performance.</p>
"""

        # Add JavaScript for chart rendering
        html_content += f"""
    <script>
        // Memory usage chart
        const memoryCtx = document.getElementById('memoryChart').getContext('2d');
        new Chart(memoryCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(models)},
                datasets: [{{
                    label: 'Memory Usage (KB)',
                    data: {json.dumps(memory_values)},
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Memory Usage (KB)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Model'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // Inference time chart
        const inferenceCtx = document.getElementById('inferenceChart').getContext('2d');
        new Chart(inferenceCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(models)},
                datasets: [{{
                    label: 'Inference Time (units)',
                    data: {json.dumps(inference_values)},
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Inference Time (arbitrary units)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Model'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // Storage chart
        const storageCtx = document.getElementById('storageChart').getContext('2d');
        new Chart(storageCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(models)},
                datasets: [{{
                    label: 'Storage (KB)',
                    data: {json.dumps(storage_values)},
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Storage Requirements (KB)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Model'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // Normalized comparison chart
        const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
        
        // Normalize data
        const maxMemory = Math.max(...{json.dumps(memory_values)});
        const maxInference = Math.max(...{json.dumps(inference_values)});
        const maxStorage = Math.max(...{json.dumps(storage_values)});
        
        const normMemory = {json.dumps(memory_values)}.map(v => v / maxMemory);
        const normInference = {json.dumps(inference_values)}.map(v => v / maxInference);
        const normStorage = {json.dumps(storage_values)}.map(v => v / maxStorage);
        
        new Chart(comparisonCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(models)},
                datasets: [
                    {{
                        label: 'Memory',
                        data: normMemory,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Inference',
                        data: normInference,
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Storage',
                        data: normStorage,
                        backgroundColor: 'rgba(255, 159, 64, 0.2)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Normalized Resource Usage'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Model'
                        }}
                    }}
                }}
            }}
        }});
        
        // FLOPS chart
        const flopsCtx = document.getElementById('flopsChart').getContext('2d');
        new Chart(flopsCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(models)},
                datasets: [{{
                    label: 'FLOPS',
                    data: {json.dumps(flops_values)},
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Floating Point Operations'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Model'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
    </script>
    
    <div class="footer">
        <p>Generated by GNN Resource Estimator | GNN Type Checker</p>
    </div>
</body>
</html>
"""

        # Save HTML report
        html_report_path = output_path / "resource_report_detailed.html"
        with open(html_report_path, 'w') as f:
            f.write(html_content)
        
        return str(html_report_path)
    
    def _generate_visualizations_for_html(self, output_dir: Path) -> None:
        """
        Generate visualizations specifically for HTML embedding.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.results:
            return
        
        # Extract data for plots
        files = [os.path.basename(file_path) for file_path in self.results.keys()]
        memory_values = [result["memory_estimate"] for result in self.results.values() if "error" not in result and result["memory_estimate"] is not None]
        inference_values = [result["inference_estimate"] for result in self.results.values() if "error" not in result and result["inference_estimate"] is not None]
        storage_values = [result["storage_estimate"] for result in self.results.values() if "error" not in result and result["storage_estimate"] is not None]
        
        # Check if we have any valid data for visualization
        if not memory_values or not inference_values or not storage_values:
            print("Warning: No valid resource estimates available for HTML visualizations")
            return
            
        # Short file names for better display
        short_files = [f[:20] + "..." if len(f) > 20 else f for f in files[:len(memory_values)]]
        
        # Memory usage plot with custom styling for HTML
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bars = plt.bar(short_files, memory_values, color='skyblue')
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.title('Memory Usage Estimates', fontsize=14, fontweight='bold')
        plt.xlabel('Model File', fontsize=12)
        plt.ylabel('Memory Usage (KB)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "memory_usage_html.png", dpi=120, bbox_inches='tight')
        plt.close()
        
        # Inference time plot with custom styling
        plt.figure(figsize=(10, 6))
        bars = plt.bar(short_files, inference_values, color='lightgreen')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.title('Inference Time Estimates', fontsize=14, fontweight='bold')
        plt.xlabel('Model File', fontsize=12)
        plt.ylabel('Inference Time (arbitrary units)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "inference_time_html.png", dpi=120, bbox_inches='tight')
        plt.close()
        
        # Storage requirements plot with custom styling
        plt.figure(figsize=(10, 6))
        bars = plt.bar(short_files, storage_values, color='salmon')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.title('Storage Requirements Estimates', fontsize=14, fontweight='bold')
        plt.xlabel('Model File', fontsize=12)
        plt.ylabel('Storage (KB)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "storage_requirements_html.png", dpi=120, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_dir: Optional[str] = None, project_root_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a markdown report and JSON data of the resource estimates.
        
        Args:
            output_dir: Directory to save the report and JSON data. 
                        If None, defaults to a subdirectory 'resource_estimates' in the current working directory.
            project_root_path: Optional path to the project root for making file paths relative in the report.

        Returns:
            String summary of the report.
        """
        if not self.results:
            return "No GNN models processed for resource estimation."

        output_path = Path(output_dir if output_dir else "resource_estimates").resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / "resource_report.md"
        json_file = output_path / "resource_data.json"

        # Resolve project_root once
        actual_project_root = None
        if project_root_path:
            actual_project_root = Path(project_root_path).resolve()

        # Calculate averages
        total_files = len(self.results)
        valid_results = [r for r in self.results.values() if "error" not in r and r["memory_estimate"] is not None]
        
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

        for file_path_str, res in self.results.items():
            if res.get("error"):
                report_content.append(f"## {Path(file_path_str).name}")
                report_content.append(f"Path: {file_path_str}") # Keep original path if error occurred before resolving
                report_content.append(f"Error: {res['error']}")
                report_content.append("")
                continue
            
            file_path_obj = Path(file_path_str).resolve()
            display_path = file_path_str
            if actual_project_root:
                try:
                    display_path = str(file_path_obj.relative_to(actual_project_root))
                except ValueError:
                    display_path = file_path_obj.name # Fallback

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
        
        # Generate visualizations
        self._generate_visualizations_for_html(output_path)
        
        # Generate HTML report with detailed explanations
        html_report_path = self.generate_html_report(str(output_path))
        
        # Save JSON data
        json_path = output_path / "resource_data.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return report

    def _generate_visualizations(self, output_dir: Path) -> None:
        """
        Generate visualizations of resource estimates.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # For backward compatibility, just call the new method
        self._generate_visualizations_for_html(output_dir)


def main():
    """
    Main function to run the resource estimator from command line.
    """
    parser = argparse.ArgumentParser(description="GNN Resource Estimator")
    parser.add_argument("input_path", help="Path to GNN file or directory")
    parser.add_argument("-t", "--type-check-data", help="Path to type check JSON data")
    parser.add_argument("-o", "--output-dir", help="Directory to save resource reports")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--html-only", action="store_true", help="Generate only HTML report with visualizations")
    
    args = parser.parse_args()
    
    estimator = GNNResourceEstimator(args.type_check_data)
    
    input_path = args.input_path
    path = Path(input_path)
    
    if path.is_file():
        # Estimate single file
        result = estimator.estimate_from_file(str(path))
        estimator.results = {str(path): result}
    else:
        # Estimate directory
        estimator.estimate_from_directory(str(path), recursive=args.recursive)
    
    # Generate and display report
    report = estimator.generate_report(args.output_dir)
    
    if not args.html_only:
        print(report)
    else:
        # When HTML only mode is selected, just print a simple summary and HTML location
        output_dir = args.output_dir if args.output_dir else "output/type_checker/resources"
        html_path = os.path.join(output_dir, "resource_report_detailed.html")
        print(f"Generated HTML resource report at: {html_path}")
        print(f"Analyzed {len(estimator.results)} files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 