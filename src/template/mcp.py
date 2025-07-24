"""
Template Step MCP Integration

This module provides Model Context Protocol integration for the template step.
It registers tools that can be used by MCP-enabled applications to interact with the template functionality.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

def register_tools(registry):
    """
    Register all template tools with the MCP registry.
    
    Args:
        registry: The MCP tool registry
    """
    try:
        # Register process_file tool
        registry.register_tool(
            name="template.process_file",
            description="Process a file using the template processor",
            function=process_file,
            parameters=[
                {
                    "name": "file_path",
                    "description": "Path to the file to process",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "output_dir",
                    "description": "Output directory for processed files",
                    "type": "string",
                    "required": False,
                    "default": "output/template"
                },
                {
                    "name": "options",
                    "description": "Processing options",
                    "type": "object",
                    "required": False,
                    "default": {}
                }
            ],
            returns={
                "type": "object",
                "description": "Processing result with status and output paths"
            },
            examples=[
                {
                    "description": "Process a markdown file",
                    "code": 'template.process_file("input/example.md")'
                }
            ]
        )
        
        # Register process_directory tool
        registry.register_tool(
            name="template.process_directory",
            description="Process all files in a directory using the template processor",
            function=process_directory,
            parameters=[
                {
                    "name": "directory_path",
                    "description": "Path to the directory to process",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "recursive",
                    "description": "Whether to process files recursively",
                    "type": "boolean",
                    "required": False,
                    "default": False
                },
                {
                    "name": "output_dir",
                    "description": "Output directory for processed files",
                    "type": "string",
                    "required": False,
                    "default": "output/template"
                },
                {
                    "name": "options",
                    "description": "Processing options",
                    "type": "object",
                    "required": False,
                    "default": {}
                }
            ],
            returns={
                "type": "object",
                "description": "Processing result with status and summary statistics"
            },
            examples=[
                {
                    "description": "Process all files in a directory recursively",
                    "code": 'template.process_directory("input/gnn_files", recursive=True)'
                }
            ]
        )
        
        # Register get_template_info tool
        registry.register_tool(
            name="template.get_info",
            description="Get information about the template step",
            function=get_template_info,
            parameters=[],
            returns={
                "type": "object",
                "description": "Template step information"
            },
            examples=[
                {
                    "description": "Get template step information",
                    "code": 'template.get_info()'
                }
            ]
        )
        
        logger.info("Successfully registered template MCP tools")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register template MCP tools: {e}")
        return False

def process_file(file_path: str, output_dir: str = "output/template", options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a single file using the template processor.
    
    Args:
        file_path: Path to the file to process
        output_dir: Output directory for processed files
        options: Processing options
        
    Returns:
        Processing result with status and output paths
    """
    try:
        # Convert string paths to Path objects
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default options if none provided
        if options is None:
            options = {}
        
        # Import the actual processing function from the template module
        from template.processor import process_single_file
        
        # Process the file
        success = process_single_file(file_path, output_dir, options)
        
        # Generate result
        result = {
            "status": "success" if success else "error",
            "input_file": str(file_path),
            "output_directory": str(output_dir),
            "processing_options": options
        }
        
        # Add output file paths if successful
        if success:
            file_output_dir = output_dir / file_path.stem
            output_file = file_output_dir / f"{file_path.stem}_processed{file_path.suffix}"
            report_file = file_output_dir / f"{file_path.stem}_report.json"
            
            result["output_file"] = str(output_file)
            result["report_file"] = str(report_file)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "input_file": str(file_path)
        }

def process_directory(directory_path: str, recursive: bool = False, output_dir: str = "output/template", options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process all files in a directory using the template processor.
    
    Args:
        directory_path: Path to the directory to process
        recursive: Whether to process files recursively
        output_dir: Output directory for processed files
        options: Processing options
        
    Returns:
        Processing result with status and summary statistics
    """
    try:
        # Convert string paths to Path objects
        directory_path = Path(directory_path)
        output_dir = Path(output_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default options if none provided
        if options is None:
            options = {}
        
        # Import the template module's main processing function
        from template.processor import process_template_standardized
        
        # Set up a basic logger for this operation
        operation_logger = logging.getLogger("template.mcp.process_directory")
        
        # Process the directory
        success = process_template_standardized(
            target_dir=directory_path,
            output_dir=output_dir,
            logger=operation_logger,
            recursive=recursive,
            verbose=options.get('verbose', False),
            **options
        )
        
        # Generate result
        result = {
            "status": "success" if success else "error",
            "input_directory": str(directory_path),
            "output_directory": str(output_dir),
            "recursive": recursive,
            "processing_options": options
        }
        
        # Add summary file path if it exists
        summary_file = output_dir / "template_processing_summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            result["summary"] = summary
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process directory {directory_path}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "input_directory": str(directory_path)
        }

def get_template_info() -> Dict[str, Any]:
    """
    Get information about the template step.
    
    Returns:
        Template step information
    """
    return {
        "name": "Template Step",
        "description": "Standardized template for all pipeline steps",
        "version": "1.0.0",
        "step_number": 0,
        "capabilities": [
            "File processing",
            "Directory processing",
            "MCP integration",
            "Standardized logging",
            "Error handling",
            "Performance tracking"
        ],
        "input_formats": ["any"],
        "output_formats": ["processed files", "JSON reports"],
        "dependencies": []
    } 