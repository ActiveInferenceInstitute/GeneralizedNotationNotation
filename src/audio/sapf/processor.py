#!/usr/bin/env python3
"""
SAPF Processor module for GNN Processing Pipeline.

This module provides SAPF audio processing capabilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def process_gnn_to_audio(gnn_content: str, model_name: str, output_dir: str, 
                        duration: float = 10.0, validate_only: bool = False) -> dict:
    """
    Process GNN content to audio using SAPF.
    
    Args:
        gnn_content: GNN file content
        model_name: Name of the model
        output_dir: Output directory
        duration: Audio duration in seconds
        validate_only: Only validate, don't generate audio
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Processing GNN to audio for model: {model_name}")
        
        # Import SAPF processor
        from .sapf_gnn_processor import SAPFGNNProcessor, convert_gnn_to_sapf
        
        # Create processor
        processor = SAPFGNNProcessor()
        
        # Convert GNN to SAPF
        sapf_code = convert_gnn_to_sapf(gnn_content)
        
        if validate_only:
            # Only validate SAPF code
            validation_result = processor.validate_sapf_code(sapf_code)
            return {
                "success": validation_result["valid"],
                "validation_result": validation_result,
                "model_name": model_name,
                "sapf_code": sapf_code
            }
        
        # Generate audio
        output_path = Path(output_dir) / f"{model_name}_sapf_audio.wav"
        audio_result = processor.generate_audio_from_sapf(sapf_code, str(output_path), duration)
        
        return {
            "success": audio_result["success"],
            "audio_file": str(output_path),
            "model_name": model_name,
            "sapf_code": sapf_code,
            "duration": duration,
            "audio_result": audio_result
        }
        
    except Exception as e:
        logger.error(f"Failed to process GNN to audio: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name
        }

def generate_sapf_audio(sapf_code, output_path, **kwargs):
    """
    Generate audio from SAPF code.
    
    Args:
        sapf_code: SAPF code string
        output_path: Output file path
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with generation results
    """
    try:
        from .sapf_gnn_processor import SAPFGNNProcessor
        
        processor = SAPFGNNProcessor()
        result = processor.generate_audio_from_sapf(sapf_code, output_path, **kwargs)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate SAPF audio: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def create_sapf_visualization(sapf_code, output_path=None):
    """
    Create visualization from SAPF code.
    
    Args:
        sapf_code: SAPF code string
        output_path: Output file path (optional)
        
    Returns:
        Dictionary with visualization results
    """
    try:
        # Basic visualization creation
        visualization_data = {
            "sapf_code": sapf_code,
            "components": [],
            "timeline": [],
            "frequencies": []
        }
        
        # Parse SAPF code for visualization
        lines = sapf_code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('oscillator'):
                # Extract oscillator information
                parts = line.split()
                if len(parts) >= 3:
                    visualization_data["components"].append({
                        "type": "oscillator",
                        "frequency": parts[1],
                        "amplitude": parts[2] if len(parts) > 2 else "1.0"
                    })
            elif line.startswith('envelope'):
                # Extract envelope information
                parts = line.split()
                if len(parts) >= 2:
                    visualization_data["components"].append({
                        "type": "envelope",
                        "shape": parts[1]
                    })
        
        result = {
            "success": True,
            "visualization_data": visualization_data
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(visualization_data, f, indent=2)
            result["output_file"] = output_path
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create SAPF visualization: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_sapf_report(sapf_results, output_path=None):
    """
    Generate report from SAPF results.
    
    Args:
        sapf_results: SAPF processing results
        output_path: Output file path (optional)
        
    Returns:
        Dictionary with report results
    """
    try:
        report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "sapf_results": sapf_results,
            "summary": {
                "success": sapf_results.get("success", False),
                "components_count": len(sapf_results.get("components", [])),
                "duration": sapf_results.get("duration", 0.0)
            }
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            report["output_file"] = output_path
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate SAPF report: {e}")
        return {
            "success": False,
            "error": str(e)
        }
