#!/usr/bin/env python3
"""
MCP Registration for SymPy Integration

This module registers SymPy MCP tools with the GNN MCP system,
providing symbolic mathematics capabilities for GNN model validation and analysis.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Global SymPy integration instance
sympy_integration_instance = None

def register_tools(mcp_instance):
    """
    Register SymPy MCP tools with the main MCP instance.
    
    Args:
        mcp_instance: The main MCP instance to register tools with
    """
    logger.debug("Registering SymPy MCP tools...")
    
    # Register mathematical validation tools
    mcp_instance.register_tool(
        name="sympy_validate_equation",
        func=validate_equation_tool,
        schema={
            "type": "object",
            "properties": {
                "equation": {
                    "type": "string",
                    "description": "Mathematical equation to validate"
                },
                "context": {
                    "type": "object",
                    "description": "GNN context for variable definitions",
                    "default": {}
                }
            },
            "required": ["equation"]
        },
        description="Validate a mathematical equation using SymPy symbolic processing"
    )
    
    mcp_instance.register_tool(
        name="sympy_validate_matrix",
        func=validate_matrix_tool,
        schema={
            "type": "object",
            "properties": {
                "matrix_data": {
                    "type": "array",
                    "description": "Matrix data as array of arrays"
                },
                "matrix_type": {
                    "type": "string",
                    "description": "Type of matrix (transition, observation, etc.)",
                    "default": "transition"
                }
            },
            "required": ["matrix_data"]
        },
        description="Validate matrix properties including stochasticity constraints"
    )
    
    mcp_instance.register_tool(
        name="sympy_analyze_stability",
        func=analyze_stability_tool,
        schema={
            "type": "object",
            "properties": {
                "transition_matrices": {
                    "type": "array",
                    "description": "List of transition matrices to analyze"
                }
            },
            "required": ["transition_matrices"]
        },
        description="Analyze system stability using eigenvalue analysis"
    )
    
    mcp_instance.register_tool(
        name="sympy_simplify_expression",
        func=simplify_expression_tool,
        schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to simplify"
                }
            },
            "required": ["expression"]
        },
        description="Simplify a mathematical expression to canonical form"
    )
    
    mcp_instance.register_tool(
        name="sympy_solve_equation",
        func=solve_equation_tool,
        schema={
            "type": "object",
            "properties": {
                "equation": {
                    "type": "string",
                    "description": "Equation to solve"
                },
                "variable": {
                    "type": "string",
                    "description": "Variable to solve for"
                },
                "domain": {
                    "type": "string",
                    "description": "Solution domain (COMPLEX, REAL, etc.)",
                    "default": "COMPLEX"
                }
            },
            "required": ["equation", "variable"]
        },
        description="Solve an equation algebraically for a specified variable"
    )
    
    mcp_instance.register_tool(
        name="sympy_get_latex",
        func=get_latex_tool,
        schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression to convert to LaTeX"
                }
            },
            "required": ["expression"]
        },
        description="Convert a mathematical expression to LaTeX format"
    )
    
    # Register setup and cleanup tools
    mcp_instance.register_tool(
        name="sympy_initialize",
        func=initialize_sympy_tool,
        schema={
            "type": "object",
            "properties": {
                "server_executable": {
                    "type": "string",
                    "description": "Path to SymPy MCP server executable",
                    "default": None
                }
            }
        },
        description="Initialize SymPy MCP integration"
    )
    
    mcp_instance.register_tool(
        name="sympy_cleanup",
        func=cleanup_sympy_tool,
        schema={
            "type": "object",
            "properties": {}
        },
        description="Clean up SymPy MCP integration and reset state"
    )
    
    logger.info("Successfully registered SymPy MCP tools")


async def get_sympy_integration():
    """Get or create the SymPy integration instance"""
    global sympy_integration_instance
    
    if sympy_integration_instance is None:
        try:
            from .sympy_mcp_client import create_sympy_integration
            # Try to find SymPy MCP server executable
            server_executable = find_sympy_server_executable()
            sympy_integration_instance = await create_sympy_integration(server_executable)
            logger.info("SymPy MCP integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SymPy MCP integration: {e}")
            raise
    
    return sympy_integration_instance


def find_sympy_server_executable() -> Optional[str]:
    """Try to find the SymPy MCP server executable"""
    # Common locations to check
    possible_locations = [
        "server.py",  # Local copy
        "../sympy-mcp/server.py",  # Adjacent directory
        "~/sympy-mcp/server.py",  # Home directory
    ]
    
    for location in possible_locations:
        path = Path(location).expanduser()
        if path.exists():
            logger.debug(f"Found SymPy MCP server at: {path}")
            return str(path)
    
    logger.warning("SymPy MCP server executable not found, will try without auto-start")
    return None


# Tool implementation functions

async def validate_equation_tool(equation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Tool function for equation validation"""
    if context is None:
        context = {}
    try:
        integration = await get_sympy_integration()
        result = await integration.validate_gnn_equation(equation)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error validating equation: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def validate_matrix_tool(matrix_data: List[List[Any]], matrix_type: str = "transition") -> Dict[str, Any]:
    """Tool function for matrix validation"""
    try:
        integration = await get_sympy_integration()
        result = await integration.validate_matrix_stochasticity(matrix_data, matrix_type)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error validating matrix: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def analyze_stability_tool(transition_matrices: List[List[List[Any]]]) -> Dict[str, Any]:
    """Tool function for stability analysis"""
    try:
        integration = await get_sympy_integration()
        result = await integration.analyze_system_stability(transition_matrices)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error analyzing stability: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def simplify_expression_tool(expression: str) -> Dict[str, Any]:
    """Tool function for expression simplification"""
    try:
        integration = await get_sympy_integration()
        
        # Validate and simplify the expression
        validation_result = await integration.validate_gnn_equation(expression)
        
        if validation_result["valid"]:
            return {
                "success": True,
                "original": expression,
                "simplified_latex": validation_result["latex"],
                "result": validation_result
            }
        else:
            return {
                "success": False,
                "error": validation_result.get("error", "Unknown validation error")
            }
    except Exception as e:
        logger.error(f"Error simplifying expression: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def solve_equation_tool(equation: str, variable: str, domain: str = "COMPLEX") -> Dict[str, Any]:
    """Tool function for equation solving"""
    try:
        integration = await get_sympy_integration()
        
        # First validate the equation
        validation_result = await integration.validate_gnn_equation(equation)
        
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": f"Invalid equation: {validation_result.get('error', 'Unknown error')}"
            }
        
        # Solve the equation
        expr_key = validation_result["expr_key"]
        solution = await integration.sympy_client.solve_algebraically(expr_key, variable, domain)
        
        return {
            "success": True,
            "equation": equation,
            "variable": variable,
            "domain": domain,
            "solution": solution
        }
    except Exception as e:
        logger.error(f"Error solving equation: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def get_latex_tool(expression: str) -> Dict[str, Any]:
    """Tool function for LaTeX conversion"""
    try:
        integration = await get_sympy_integration()
        
        # Validate the expression and get LaTeX
        validation_result = await integration.validate_gnn_equation(expression)
        
        if validation_result["valid"]:
            return {
                "success": True,
                "expression": expression,
                "latex": validation_result["latex"]
            }
        else:
            return {
                "success": False,
                "error": validation_result.get("error", "Invalid expression")
            }
    except Exception as e:
        logger.error(f"Error getting LaTeX: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def initialize_sympy_tool(server_executable: Optional[str] = None) -> Dict[str, Any]:
    """Tool function for SymPy initialization"""
    global sympy_integration_instance
    
    try:
        # Clean up existing instance
        if sympy_integration_instance:
            await sympy_integration_instance.cleanup()
            sympy_integration_instance = None
        
        # Create new instance
        from .sympy_mcp_client import create_sympy_integration
        sympy_integration_instance = await create_sympy_integration(server_executable)
        
        return {
            "success": True,
            "message": "SymPy MCP integration initialized successfully"
        }
    except Exception as e:
        logger.error(f"Error initializing SymPy: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def cleanup_sympy_tool() -> Dict[str, Any]:
    """Tool function for SymPy cleanup"""
    global sympy_integration_instance
    
    try:
        if sympy_integration_instance:
            await sympy_integration_instance.cleanup()
            sympy_integration_instance = None
        
        return {
            "success": True,
            "message": "SymPy MCP integration cleaned up successfully"
        }
    except Exception as e:
        logger.error(f"Error cleaning up SymPy: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Synchronous wrapper functions for compatibility with MCP system
def validate_equation_tool_sync(equation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Synchronous wrapper for equation validation"""
    if context is None:
        context = {}
    
    # Store reference to async function to avoid recursion
    async def _async_wrapper():
        return await validate_equation_tool_async(equation, context)
    
    return asyncio.run(_async_wrapper())

def validate_matrix_tool_sync(matrix_data: List[List[Any]], matrix_type: str = "transition") -> Dict[str, Any]:
    """Synchronous wrapper for matrix validation"""
    async def _async_wrapper():
        return await validate_matrix_tool_async(matrix_data, matrix_type)
    
    return asyncio.run(_async_wrapper())

def analyze_stability_tool_sync(transition_matrices: List[List[List[Any]]]) -> Dict[str, Any]:
    """Synchronous wrapper for stability analysis"""
    async def _async_wrapper():
        return await analyze_stability_tool_async(transition_matrices)
    
    return asyncio.run(_async_wrapper())

def simplify_expression_tool_sync(expression: str) -> Dict[str, Any]:
    """Synchronous wrapper for expression simplification"""
    async def _async_wrapper():
        return await simplify_expression_tool_async(expression)
    
    return asyncio.run(_async_wrapper())

def solve_equation_tool_sync(equation: str, variable: str, domain: str = "COMPLEX") -> Dict[str, Any]:
    """Synchronous wrapper for equation solving"""
    async def _async_wrapper():
        return await solve_equation_tool_async(equation, variable, domain)
    
    return asyncio.run(_async_wrapper())

def get_latex_tool_sync(expression: str) -> Dict[str, Any]:
    """Synchronous wrapper for LaTeX conversion"""
    async def _async_wrapper():
        return await get_latex_tool_async(expression)
    
    return asyncio.run(_async_wrapper())

def initialize_sympy_tool_sync(server_executable: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper for SymPy initialization"""
    async def _async_wrapper():
        return await initialize_sympy_tool_async(server_executable)
    
    return asyncio.run(_async_wrapper())

def cleanup_sympy_tool_sync() -> Dict[str, Any]:
    """Synchronous wrapper for SymPy cleanup"""
    async def _async_wrapper():
        return await cleanup_sympy_tool_async()
    
    return asyncio.run(_async_wrapper())


# Rename async functions to avoid conflicts
validate_equation_tool_async = validate_equation_tool
validate_matrix_tool_async = validate_matrix_tool
analyze_stability_tool_async = analyze_stability_tool
simplify_expression_tool_async = simplify_expression_tool
solve_equation_tool_async = solve_equation_tool
get_latex_tool_async = get_latex_tool
initialize_sympy_tool_async = initialize_sympy_tool
cleanup_sympy_tool_async = cleanup_sympy_tool

# Use sync versions for MCP registration
validate_equation_tool = validate_equation_tool_sync
validate_matrix_tool = validate_matrix_tool_sync
analyze_stability_tool = analyze_stability_tool_sync
simplify_expression_tool = simplify_expression_tool_sync
solve_equation_tool = solve_equation_tool_sync
get_latex_tool = get_latex_tool_sync
initialize_sympy_tool = initialize_sympy_tool_sync
cleanup_sympy_tool = cleanup_sympy_tool_sync 