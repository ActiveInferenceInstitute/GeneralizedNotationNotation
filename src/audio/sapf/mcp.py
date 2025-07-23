"""
MCP (Model Context Protocol) Integration for SAPF Module

Provides MCP tools for SAPF-GNN audio generation functionality.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .sapf_gnn_processor import SAPFGNNProcessor, convert_gnn_to_sapf, validate_sapf_code
from .audio_generators import SyntheticAudioGenerator

logger = logging.getLogger(__name__)

def register_sapf_tools() -> List[Dict[str, Any]]:
    """
    Register SAPF-related MCP tools.
    
    Returns:
        List of MCP tool definitions
    """
    return [
        {
            "name": "convert_gnn_to_sapf_audio",
            "description": "Convert a GNN model to SAPF code and generate audio",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "gnn_content": {
                        "type": "string",
                        "description": "GNN model content to convert"
                    },
                    "model_name": {
                        "type": "string", 
                        "description": "Name of the model"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for files"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Audio duration in seconds",
                        "default": 10.0
                    }
                },
                "required": ["gnn_content", "model_name", "output_dir"]
            }
        },
        {
            "name": "generate_sapf_code",
            "description": "Generate SAPF code from GNN model content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "gnn_content": {
                        "type": "string",
                        "description": "GNN model content"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Name of the model"
                    }
                },
                "required": ["gnn_content", "model_name"]
            }
        },
        {
            "name": "validate_sapf_syntax",
            "description": "Validate SAPF code syntax",
            "inputSchema": {
                "type": "object", 
                "properties": {
                    "sapf_code": {
                        "type": "string",
                        "description": "SAPF code to validate"
                    }
                },
                "required": ["sapf_code"]
            }
        },
        {
            "name": "generate_audio_from_sapf",
            "description": "Generate audio file from SAPF code",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sapf_code": {
                        "type": "string",
                        "description": "SAPF code to execute"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output audio file path"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Audio duration in seconds",
                        "default": 10.0
                    }
                },
                "required": ["sapf_code", "output_file"]
            }
        },
        {
            "name": "analyze_gnn_for_audio",
            "description": "Analyze GNN model structure for audio generation parameters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "gnn_content": {
                        "type": "string",
                        "description": "GNN model content to analyze"
                    }
                },
                "required": ["gnn_content"]
            }
        }
    ]

async def handle_convert_gnn_to_sapf_audio(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the convert_gnn_to_sapf_audio MCP request.
    
    Args:
        params: Request parameters
        
    Returns:
        Response dictionary
    """
    try:
        gnn_content = params["gnn_content"]
        model_name = params["model_name"]
        output_dir = Path(params["output_dir"])
        duration = params.get("duration", 10.0)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate SAPF code
        sapf_code = convert_gnn_to_sapf(gnn_content, model_name)
        
        # Save SAPF code
        sapf_file = output_dir / f"{model_name}.sapf"
        with open(sapf_file, 'w') as f:
            f.write(sapf_code)
        
        # Generate audio
        audio_file = output_dir / f"{model_name}_audio.wav"
        generator = SyntheticAudioGenerator()
        success = generator.generate_from_sapf(sapf_code, audio_file, duration)
        
        if success:
            return {
                "success": True,
                "sapf_file": str(sapf_file),
                "audio_file": str(audio_file),
                "duration": duration,
                "sapf_code_lines": len(sapf_code.split('\n'))
            }
        else:
            return {
                "success": False,
                "error": "Audio generation failed",
                "sapf_file": str(sapf_file)
            }
            
    except Exception as e:
        logger.error(f"Error in convert_gnn_to_sapf_audio: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def handle_generate_sapf_code(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the generate_sapf_code MCP request.
    
    Args:
        params: Request parameters
        
    Returns:
        Response dictionary
    """
    try:
        gnn_content = params["gnn_content"]
        model_name = params["model_name"]
        
        sapf_code = convert_gnn_to_sapf(gnn_content, model_name)
        
        return {
            "success": True,
            "sapf_code": sapf_code,
            "model_name": model_name,
            "code_lines": len(sapf_code.split('\n'))
        }
        
    except Exception as e:
        logger.error(f"Error in generate_sapf_code: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def handle_validate_sapf_syntax(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the validate_sapf_syntax MCP request.
    
    Args:
        params: Request parameters
        
    Returns:
        Response dictionary
    """
    try:
        sapf_code = params["sapf_code"]
        
        is_valid, issues = validate_sapf_code(sapf_code)
        
        return {
            "valid": is_valid,
            "issues": issues,
            "code_lines": len(sapf_code.split('\n'))
        }
        
    except Exception as e:
        logger.error(f"Error in validate_sapf_syntax: {e}")
        return {
            "valid": False,
            "issues": [f"Validation error: {str(e)}"]
        }

async def handle_generate_audio_from_sapf(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the generate_audio_from_sapf MCP request.
    
    Args:
        params: Request parameters
        
    Returns:
        Response dictionary
    """
    try:
        sapf_code = params["sapf_code"]
        output_file = Path(params["output_file"])
        duration = params.get("duration", 10.0)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        generator = SyntheticAudioGenerator()
        success = generator.generate_from_sapf(sapf_code, output_file, duration)
        
        if success:
            return {
                "success": True,
                "audio_file": str(output_file),
                "duration": duration,
                "file_size": output_file.stat().st_size if output_file.exists() else 0
            }
        else:
            return {
                "success": False,
                "error": "Audio generation failed"
            }
            
    except Exception as e:
        logger.error(f"Error in generate_audio_from_sapf: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def handle_analyze_gnn_for_audio(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the analyze_gnn_for_audio MCP request.
    
    Args:
        params: Request parameters
        
    Returns:
        Response dictionary
    """
    try:
        gnn_content = params["gnn_content"]
        
        processor = SAPFGNNProcessor()
        sections = processor.parse_gnn_sections(gnn_content)
        
        # Analyze for audio generation parameters
        analysis = {
            "model_sections": list(sections.keys()),
            "state_variables": len(sections.get('StateSpaceBlock', [])),
            "connections": len(sections.get('Connections', [])),
            "has_matrices": bool(sections.get('InitialParameterization', {})),
            "has_time_config": bool(sections.get('Time', {})),
            "estimated_complexity": _estimate_audio_complexity(sections),
            "suggested_duration": _suggest_audio_duration(sections)
        }
        
        # Add detailed breakdown
        if 'StateSpaceBlock' in sections:
            analysis["state_space_detail"] = [
                {
                    "name": state.get('name', 'unknown'),
                    "dimensions": state.get('dimensions', []),
                    "type": state.get('type', 'continuous')
                }
                for state in sections['StateSpaceBlock']
            ]
        
        if 'Connections' in sections:
            analysis["connections_detail"] = [
                {
                    "source": conn.get('source', ''),
                    "target": conn.get('target', ''),
                    "type": conn.get('type', ''),
                    "directed": conn.get('directed', False)
                }
                for conn in sections['Connections']
            ]
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_gnn_for_audio: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def _estimate_audio_complexity(sections: Dict[str, Any]) -> str:
    """Estimate audio complexity based on GNN sections."""
    complexity_score = 0
    
    # State space complexity
    if 'StateSpaceBlock' in sections:
        state_count = len(sections['StateSpaceBlock'])
        complexity_score += state_count
        
        # Check for multi-dimensional states
        for state in sections['StateSpaceBlock']:
            dims = state.get('dimensions', [1])
            if len(dims) > 1:
                complexity_score += sum(dims) / 10
    
    # Connection complexity
    if 'Connections' in sections:
        complexity_score += len(sections['Connections']) * 0.5
    
    # Matrix complexity
    if 'InitialParameterization' in sections:
        matrices = sections['InitialParameterization']
        complexity_score += len(matrices) * 2
    
    if complexity_score < 3:
        return "simple"
    elif complexity_score < 8:
        return "moderate"
    else:
        return "complex"

def _suggest_audio_duration(sections: Dict[str, Any]) -> float:
    """Suggest appropriate audio duration based on model characteristics."""
    base_duration = 10.0
    
    if 'Time' in sections:
        horizon = sections['Time'].get('ModelTimeHorizon', 10)
        # Scale duration based on time horizon
        base_duration = max(5.0, min(30.0, horizon))
    
    # Adjust for complexity
    complexity = _estimate_audio_complexity(sections)
    if complexity == "simple":
        return base_duration * 0.8
    elif complexity == "complex":
        return base_duration * 1.5
    
    return base_duration

# MCP handler mapping
MCP_HANDLERS = {
    "convert_gnn_to_sapf_audio": handle_convert_gnn_to_sapf_audio,
    "generate_sapf_code": handle_generate_sapf_code,
    "validate_sapf_syntax": handle_validate_sapf_syntax,
    "generate_audio_from_sapf": handle_generate_audio_from_sapf,
    "analyze_gnn_for_audio": handle_analyze_gnn_for_audio
} 