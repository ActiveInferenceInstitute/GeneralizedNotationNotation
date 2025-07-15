from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import time
import json

# Import MCP functionality
try:
    from .mcp_server import MCPServer
    from .tool_registry import ToolRegistry
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False

def process_mcp_operations(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False):
    """Process MCP (Model Context Protocol) operations."""
    log_step_start(logger, "Processing MCP operations")
    
    # Use centralized output directory configuration
    mcp_output_dir = get_output_dir_for_script("7_mcp.py", output_dir)
    mcp_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not MCP_AVAILABLE:
        log_step_error(logger, "MCP functionality not available")
        return False
    
    try:
        # Initialize MCP server and tool registry
        tool_registry = ToolRegistry()
        mcp_server = MCPServer(tool_registry)
        
        # Register tools from all modules
        with performance_tracker.track_operation("register_mcp_tools"):
            # Register tools from each module
            modules_to_register = [
                "gnn", "export", "visualization", "render", 
                "execute", "llm", "site", "sapf", "setup", "tests"
            ]
            
            for module_name in modules_to_register:
                try:
                    tool_registry.register_module_tools(module_name)
                    logger.debug(f"Registered tools from {module_name} module")
                except Exception as e:
                    log_step_warning(logger, f"Failed to register tools from {module_name}: {e}")
        
        # Find GNN files for processing
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True
        
        logger.info(f"Found {len(gnn_files)} GNN files for MCP processing")
        
        # Process files with MCP tools
        successful_operations = 0
        failed_operations = 0
        
        with performance_tracker.track_operation("process_mcp_operations"):
            for gnn_file in gnn_files:
                try:
                    logger.debug(f"Processing {gnn_file.name} with MCP tools")
                    
                    # Create file-specific output directory
                    file_output_dir = mcp_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process with MCP tools
                    results = mcp_server.process_file(
                        file_path=str(gnn_file),
                        output_dir=str(file_output_dir)
                    )
                    
                    if results.get('success', False):
                        successful_operations += 1
                        logger.debug(f"MCP processing completed for {gnn_file.name}")
                    else:
                        failed_operations += 1
                        log_step_warning(logger, f"MCP processing failed for {gnn_file.name}")
                        
                except Exception as e:
                    failed_operations += 1
                    log_step_error(logger, f"Failed to process {gnn_file.name} with MCP: {e}")
        
        # Generate MCP summary report
        summary_file = mcp_output_dir / "mcp_processing_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(gnn_files),
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "available_tools": tool_registry.get_registered_tools(),
            "success_rate": successful_operations / len(gnn_files) * 100 if gnn_files else 0
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log results summary
        if successful_operations == len(gnn_files):
            log_step_success(logger, f"All {len(gnn_files)} files processed successfully with MCP")
            return True
        elif successful_operations > 0:
            log_step_warning(logger, f"Partial success: {successful_operations}/{len(gnn_files)} files processed successfully")
            return True
        else:
            log_step_error(logger, "No files were processed successfully with MCP")
            return False
        
    except Exception as e:
        log_step_error(logger, f"MCP processing failed: {e}")
        return False 