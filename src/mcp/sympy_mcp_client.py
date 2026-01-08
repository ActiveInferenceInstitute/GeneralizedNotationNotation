#!/usr/bin/env python3
"""
SymPy MCP Client for GNN Integration

This module provides a client interface to connect to the SymPy MCP server
and integrate symbolic mathematics capabilities into the GNN pipeline.
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import sys

# Try to import httpx for HTTP client functionality
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    # Create a dummy httpx module for type checking
    class _DummyHttpx:
        class AsyncClient:
            def __init__(self, **kwargs): pass
            async def get(self, url: str): raise NotImplementedError()
            async def post(self, url: str, **kwargs): raise NotImplementedError()
            async def aclose(self): pass
        class RequestError(Exception): pass
    httpx = _DummyHttpx()

logger = logging.getLogger(__name__)

class SymPyMCPError(Exception):
    """Base exception for SymPy MCP operations"""
    pass

class SymPyMCPConnectionError(SymPyMCPError):
    """Exception raised when connection to SymPy MCP server fails"""
    pass

class SymPyMCPClient:
    """Client for interacting with SymPy MCP server"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8081", 
                 server_executable: Optional[str] = None,
                 auto_start_server: bool = True):
        """
        Initialize SymPy MCP client.
        
        Args:
            server_url: URL of the SymPy MCP server
            server_executable: Path to SymPy MCP server executable
            auto_start_server: Whether to automatically start server if not running
        """
        self.server_url = server_url
        self.server_executable = server_executable
        self.auto_start_server = auto_start_server
        self.server_process: Optional[subprocess.Popen[str]] = None
        self.session_id = None
        self._client: Optional[httpx.AsyncClient] = None
        
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, SymPy MCP client will have limited functionality")
    
    async def __aenter__(self) -> "SymPyMCPClient":
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Connect to SymPy MCP server, starting it if necessary"""
        if not HTTPX_AVAILABLE:
            raise SymPyMCPConnectionError("httpx not available for HTTP client functionality")
        
        self._client = httpx.AsyncClient(timeout=30.0)
        
        # Check if server is already running
        try:
            response = await self._client.get(f"{self.server_url}/healthcheck")
            if response.status_code == 200:
                logger.info("SymPy MCP server is already running")
                return
        except httpx.RequestError:
            logger.debug("SymPy MCP server not responding, will try to start it")
        
        # Try to start server if configured to do so
        if self.auto_start_server and self.server_executable:
            await self._start_server()
            
            # Wait for server to start
            for attempt in range(10):
                try:
                    if self._client is not None:  # Type guard
                        response = await self._client.get(f"{self.server_url}/healthcheck")
                        if response.status_code == 200:
                            logger.info("SymPy MCP server started successfully")
                            return
                except httpx.RequestError:
                    await asyncio.sleep(1)
            
            raise SymPyMCPConnectionError("Failed to start SymPy MCP server")
        else:
            raise SymPyMCPConnectionError("SymPy MCP server not running and auto-start disabled")
    
    async def disconnect(self):
        """Disconnect from SymPy MCP server"""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
    
    async def _start_server(self):
        """Start the SymPy MCP server"""
        if not self.server_executable:
            raise SymPyMCPConnectionError("Server executable not specified")
        
        cmd = [
            "uv", "run", 
            "--with", "mcp[cli]",
            "--with", "sympy",
            "mcp", "run", 
            str(self.server_executable),
            "--transport", "sse"
        ]
        
        logger.info(f"Starting SymPy MCP server: {' '.join(cmd)}")
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give server time to start
            await asyncio.sleep(2)
            
            if self.server_process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.server_process.communicate()
                raise SymPyMCPConnectionError(
                    f"Server failed to start. stdout: {stdout}, stderr: {stderr}"
                )
        except Exception as e:
            raise SymPyMCPConnectionError(f"Failed to start server: {e}")
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool on the SymPy MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result from the tool execution
        """
        if not self._client:
            raise SymPyMCPConnectionError("Not connected to server")
        
        payload = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            }
        }
        
        try:
            response = await self._client.post(
                f"{self.server_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            if "error" in result:
                raise SymPyMCPError(f"Tool execution failed: {result['error']}")
            
            content = result.get("result", {}).get("content", [{}])
            if content:
                return content[0].get("text", "")
            return ""
        except httpx.RequestError as e:
            raise SymPyMCPConnectionError(f"Request failed: {e}")
        except Exception as e:
            raise SymPyMCPError(f"Unexpected error: {e}")
    
    # Convenience methods for common SymPy operations
    
    async def introduce_variable(self, var_name: str, 
                               pos_assumptions: Optional[List[str]] = None,
                               neg_assumptions: Optional[List[str]] = None) -> str:
        """Introduce a variable with assumptions"""
        return await self.call_tool(
            "intro",
            var_name=var_name,
            pos_assumptions=pos_assumptions or [],
            neg_assumptions=neg_assumptions or []
        )
    
    async def introduce_multiple_variables(self, variables: List[Dict[str, Any]]) -> str:
        """Introduce multiple variables simultaneously"""
        return await self.call_tool("intro_many", variables=variables)
    
    async def introduce_expression(self, expr_str: str, 
                                 canonicalize: bool = True,
                                 expr_var_name: Optional[str] = None) -> str:
        """Parse and introduce a mathematical expression"""
        return await self.call_tool(
            "introduce_expression",
            expr_str=expr_str,
            canonicalize=canonicalize,
            expr_var_name=expr_var_name
        )
    
    async def print_latex_expression(self, expr_key: str) -> str:
        """Get LaTeX representation of an expression"""
        return await self.call_tool("print_latex_expression", expr_key=expr_key)
    
    async def simplify_expression(self, expr_key: str) -> str:
        """Simplify a mathematical expression"""
        return await self.call_tool("simplify_expression", expr_key=expr_key)
    
    async def solve_algebraically(self, expr_key: str, solve_for_var_name: str,
                                domain: str = "COMPLEX") -> str:
        """Solve an equation algebraically"""
        return await self.call_tool(
            "solve_algebraically",
            expr_key=expr_key,
            solve_for_var_name=solve_for_var_name,
            domain=domain
        )
    
    async def create_matrix(self, matrix_data: List[List[Union[int, float, str]]],
                          matrix_var_name: Optional[str] = None) -> str:
        """Create a SymPy matrix"""
        return await self.call_tool(
            "create_matrix",
            matrix_data=matrix_data,
            matrix_var_name=matrix_var_name
        )
    
    async def matrix_determinant(self, matrix_key: str) -> str:
        """Calculate matrix determinant"""
        return await self.call_tool("matrix_determinant", matrix_key=matrix_key)
    
    async def matrix_eigenvalues(self, matrix_key: str) -> str:
        """Calculate matrix eigenvalues"""
        return await self.call_tool("matrix_eigenvalues", matrix_key=matrix_key)
    
    async def matrix_eigenvectors(self, matrix_key: str) -> str:
        """Calculate matrix eigenvectors"""
        return await self.call_tool("matrix_eigenvectors", matrix_key=matrix_key)
    
    async def differentiate_expression(self, expr_key: str, var_name: str,
                                     order: int = 1) -> str:
        """Differentiate an expression"""
        return await self.call_tool(
            "differentiate_expression",
            expr_key=expr_key,
            var_name=var_name,
            order=order
        )
    
    async def integrate_expression(self, expr_key: str, var_name: str,
                                 lower_bound: Optional[str] = None,
                                 upper_bound: Optional[str] = None) -> str:
        """Integrate an expression"""
        return await self.call_tool(
            "integrate_expression",
            expr_key=expr_key,
            var_name=var_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
    
    async def reset_state(self) -> str:
        """Reset the SymPy server state"""
        return await self.call_tool("reset_state")


class GNNSymPyIntegration:
    """Integration layer between GNN and SymPy MCP for mathematical validation and analysis"""
    
    def __init__(self, sympy_client: SymPyMCPClient):
        """
        Initialize GNN-SymPy integration.
        
        Args:
            sympy_client: Connected SymPy MCP client
        """
        self.sympy_client = sympy_client
        self.variable_mapping: Dict[str, str] = {}  # GNN variable names -> SymPy variable keys
        self.expression_cache: Dict[str, str] = {}  # GNN expressions -> SymPy expression keys
        self.matrix_cache: Dict[str, str] = {}      # GNN matrices -> SymPy matrix keys
    
    async def setup_gnn_variables(self, state_space: Dict[str, Any], 
                                observation_space: Dict[str, Any]) -> None:
        """Set up SymPy variables based on GNN state and observation spaces"""
        variables_to_create = []
        
        # Process state space variables
        for var_name, var_spec in state_space.items():
            if var_name.startswith('s_f'):  # State factor
                variables_to_create.append({
                    "var_name": var_name,
                    "pos_assumptions": ["integer", "finite"],
                    "neg_assumptions": []
                })
        
        # Process observation space variables
        for var_name, var_spec in observation_space.items():
            if var_name.startswith('o_m'):  # Observation modality
                variables_to_create.append({
                    "var_name": var_name,
                    "pos_assumptions": ["integer", "finite"],
                    "neg_assumptions": []
                })
        
        # Create all variables at once
        if variables_to_create:
            result = await self.sympy_client.introduce_multiple_variables(variables_to_create)
            logger.debug(f"Created GNN variables in SymPy: {result}")
    
    async def validate_gnn_equation(self, equation_str: str) -> Dict[str, Any]:
        """
        Validate a GNN equation using SymPy.
        
        Args:
            equation_str: Mathematical equation as string
            
        Returns:
            Validation result with original, simplified, and LaTeX forms
        """
        try:
            # Convert GNN syntax to SymPy syntax if needed
            sympy_expr = self._convert_gnn_to_sympy_syntax(equation_str)
            
            # Introduce the expression
            expr_key = await self.sympy_client.introduce_expression(sympy_expr)
            self.expression_cache[equation_str] = expr_key
            
            # Simplify the expression
            simplified_key = await self.sympy_client.simplify_expression(expr_key)
            
            # Get LaTeX representation
            latex_form = await self.sympy_client.print_latex_expression(simplified_key)
            
            return {
                "valid": True,
                "original": equation_str,
                "sympy_expr": sympy_expr,
                "expr_key": expr_key,
                "simplified_key": simplified_key,
                "latex": latex_form,
                "error": None
            }
        except Exception as e:
            logger.error(f"Failed to validate equation '{equation_str}': {e}")
            return {
                "valid": False,
                "original": equation_str,
                "error": str(e)
            }
    
    async def validate_matrix_stochasticity(self, matrix_data: List[List[Any]], 
                                          matrix_type: str = "transition") -> Dict[str, Any]:
        """
        Validate that a matrix satisfies stochasticity constraints.
        
        Args:
            matrix_data: Matrix data as list of lists
            matrix_type: Type of matrix ("transition", "observation")
            
        Returns:
            Validation result including stochasticity check
        """
        try:
            # Create matrix in SymPy
            matrix_key = await self.sympy_client.create_matrix(matrix_data)
            
            # Calculate determinant
            determinant = await self.sympy_client.matrix_determinant(matrix_key)
            
            # Check stochasticity: rows should sum to 1 (for row-stochastic)
            # or columns should sum to 1 (for column-stochastic transition matrices)
            is_stochastic = True
            stochasticity_details = {"row_sums": [], "column_sums": []}
            
            for row in matrix_data:
                row_sum = sum(float(x) if isinstance(x, (int, float)) else 0 for x in row)
                stochasticity_details["row_sums"].append(row_sum)
                if abs(row_sum - 1.0) > 0.01:  # Allow small numerical tolerance
                    is_stochastic = False
            
            # Also check column sums for column-stochastic matrices
            if matrix_data and matrix_data[0]:
                num_cols = len(matrix_data[0])
                for col_idx in range(num_cols):
                    col_sum = sum(float(row[col_idx]) if isinstance(row[col_idx], (int, float)) else 0 
                                  for row in matrix_data)
                    stochasticity_details["column_sums"].append(col_sum)
            
            # Determine type of stochasticity
            row_stochastic = all(abs(s - 1.0) < 0.01 for s in stochasticity_details["row_sums"])
            col_stochastic = all(abs(s - 1.0) < 0.01 for s in stochasticity_details["column_sums"])
            
            stochasticity_type = "none"
            if row_stochastic and col_stochastic:
                stochasticity_type = "doubly_stochastic"
            elif row_stochastic:
                stochasticity_type = "row_stochastic"
            elif col_stochastic:
                stochasticity_type = "column_stochastic"
            
            return {
                "valid": True,
                "matrix_key": matrix_key,
                "determinant": determinant,
                "matrix_type": matrix_type,
                "stochastic": row_stochastic or col_stochastic,
                "stochasticity_type": stochasticity_type,
                "stochasticity_details": stochasticity_details,
                "error": None
            }
        except Exception as e:
            logger.error(f"Failed to validate matrix: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def analyze_system_stability(self, transition_matrices: List[List[List[Any]]]) -> Dict[str, Any]:
        """
        Analyze stability of dynamic system using eigenvalue analysis.
        
        Args:
            transition_matrices: List of transition matrices
            
        Returns:
            Stability analysis results
        """
        stability_results = []
        
        for i, matrix_data in enumerate(transition_matrices):
            try:
                matrix_key = await self.sympy_client.create_matrix(matrix_data)
                eigenvals_key = await self.sympy_client.matrix_eigenvalues(matrix_key)
                eigenvecs_key = await self.sympy_client.matrix_eigenvectors(matrix_key)
                
                # Analyze stability from eigenvalues
                # For discrete-time systems: stable if all |eigenvalue| < 1
                # For continuous-time systems: stable if all real(eigenvalue) < 0
                # We assume discrete-time for transition matrices
                stability_analysis = self._analyze_eigenvalue_stability(matrix_data)
                
                stability_results.append({
                    "matrix_index": i,
                    "matrix_key": matrix_key,
                    "eigenvalues": eigenvals_key,
                    "eigenvectors": eigenvecs_key,
                    "stable": stability_analysis["stable"],
                    "stability_type": stability_analysis["stability_type"],
                    "max_eigenvalue_magnitude": stability_analysis["max_magnitude"],
                    "analysis_notes": stability_analysis["notes"]
                })
            except Exception as e:
                logger.error(f"Failed to analyze matrix {i}: {e}")
                stability_results.append({
                    "matrix_index": i,
                    "error": str(e)
                })
        
        return {
            "matrices_analyzed": len(transition_matrices),
            "results": stability_results
        }
    
    def _analyze_eigenvalue_stability(self, matrix_data: List[List[Any]]) -> Dict[str, Any]:
        """
        Analyze stability from eigenvalues using numpy.
        
        For discrete-time systems (transition matrices): stable if |λ| < 1 for all eigenvalues.
        
        Args:
            matrix_data: Matrix data as list of lists
            
        Returns:
            Stability analysis results
        """
        try:
            import numpy as np
            
            # Convert to numpy array
            matrix = np.array([[float(x) if isinstance(x, (int, float)) else 0 for x in row] 
                               for row in matrix_data])
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(matrix)
            
            # Calculate magnitudes
            magnitudes = np.abs(eigenvalues)
            max_magnitude = float(np.max(magnitudes))
            
            # Stability analysis for discrete-time systems
            # Stable if all eigenvalues have magnitude < 1
            # Marginally stable if largest magnitude == 1
            # Unstable if any eigenvalue has magnitude > 1
            stability_threshold = 1.0
            
            if max_magnitude < stability_threshold - 0.001:
                stability_type = "asymptotically_stable"
                stable = True
                notes = "All eigenvalues inside unit circle"
            elif abs(max_magnitude - stability_threshold) < 0.001:
                stability_type = "marginally_stable"
                stable = True
                notes = "Eigenvalues on unit circle boundary"
            else:
                stability_type = "unstable"
                stable = False
                notes = f"Eigenvalue magnitude {max_magnitude:.4f} exceeds 1"
            
            return {
                "stable": stable,
                "stability_type": stability_type,
                "max_magnitude": max_magnitude,
                "eigenvalue_magnitudes": magnitudes.tolist(),
                "notes": notes
            }
            
        except ImportError:
            return {
                "stable": None,
                "stability_type": "unknown",
                "max_magnitude": None,
                "notes": "numpy not available for eigenvalue analysis"
            }
        except Exception as e:
            return {
                "stable": None,
                "stability_type": "error",
                "max_magnitude": None,
                "notes": f"Analysis failed: {str(e)}"
            }
    
    def _convert_gnn_to_sympy_syntax(self, gnn_expr: str) -> str:
        """
        Convert GNN mathematical syntax to SymPy syntax.
        
        Args:
            gnn_expr: Expression in GNN syntax
            
        Returns:
            Expression in SymPy syntax
        """
        # Basic conversions - extend as needed
        sympy_expr = gnn_expr
        
        # Convert GNN subscript notation s_f0 to s_f0 (already compatible)
        # Convert GNN superscript notation X^Y to X**Y
        # Convert conditional probability P(X|Y) to appropriate form
        
        # Future enhancements:
        # - Handle probability distributions
        # - Convert matrix notation
        # - Handle special Active Inference notation
        
        return sympy_expr
    
    async def cleanup(self):
        """Clean up SymPy session"""
        try:
            await self.sympy_client.reset_state()
            self.variable_mapping.clear()
            self.expression_cache.clear()
            self.matrix_cache.clear()
        except Exception as e:
            logger.error(f"Failed to cleanup SymPy session: {e}")


# Utility function for easy integration with existing GNN pipeline
async def create_sympy_integration(server_executable: Optional[str] = None) -> GNNSymPyIntegration:
    """
    Create and initialize GNN-SymPy integration.
    
    Args:
        server_executable: Path to SymPy MCP server executable
        
    Returns:
        Initialized GNN-SymPy integration instance
    """
    client = SymPyMCPClient(server_executable=server_executable)
    await client.connect()
    return GNNSymPyIntegration(client) 