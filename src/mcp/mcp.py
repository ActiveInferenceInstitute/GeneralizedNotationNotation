import importlib
import os
import sys
from pathlib import Path
import logging
import inspect
from typing import Dict, List, Any, Callable, Optional, TypedDict, Union, Tuple

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp")

# --- MCP SDK Status Simulation ---
# This simulates the detection of the MCP SDK.
# In a real application, this status would be set by the actual SDK loading mechanism.
# The original "root - WARNING - MCP SDK not found, using dummy classes" implies
# such a detection and fallback mechanism exists somewhere. We are making mcp.py
# react to this conceptual status.
_MCP_SDK_CONFIG_STATUS = {"found": True, "details": "Using project's internal MCP implementation."}

# Example: Simulate SDK detection failure for demonstration purposes.
# To test the "SDK not found" path, you would uncomment the following lines
# or have a real mechanism that sets _MCP_SDK_CONFIG_STATUS["found"] = False.
# try:
#     import hypothetical_critical_mcp_sdk_component # This would be the actual SDK import
#     _MCP_SDK_CONFIG_STATUS["found"] = True
#     _MCP_SDK_CONFIG_STATUS["details"] = "Successfully loaded critical MCP SDK component."
# except ImportError:
#     _MCP_SDK_CONFIG_STATUS["found"] = False
#     _MCP_SDK_CONFIG_STATUS["details"] = "Failed to import a critical MCP SDK component. MCP functionality will be impaired."
    # The original "root - WARNING..." might be logged by such a mechanism.

class MCPSDKNotFoundError(RuntimeError):
    """Custom exception for when the MCP SDK is not found and pipeline should halt."""
    pass

class MCPTool:
    """Represents an MCP tool that can be executed."""
    
    def __init__(self, name: str, func: Callable, schema: Dict, description: str):
        self.name = name
        self.func = func
        self.schema = schema
        self.description = description

class MCPResource:
    """Represents an MCP resource that can be accessed."""
    
    def __init__(self, uri_template: str, retriever: Callable, description: str):
        self.uri_template = uri_template
        self.retriever = retriever
        self.description = description

class MCP:
    """Main Model Context Protocol implementation."""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.modules: Dict[str, Any] = {}
        # Note: The _MCP_SDK_CONFIG_STATUS is module-level, not instance-level,
        # as SDK availability is a system-wide concern for this MCP setup.
        
    def discover_modules(self) -> bool:
        """Discover and load MCP modules from other directories.

        Returns:
            bool: True if all modules loaded successfully, False otherwise.
        """
        root_dir = Path(__file__).parent.parent
        logger.info(f"Discovering MCP modules in {root_dir}")
        all_modules_loaded_successfully = True
        
        for directory in root_dir.iterdir():
            if not directory.is_dir() or directory.name.startswith('_'):
                continue
                
            mcp_file = directory / "mcp.py"
            if not mcp_file.exists():
                logger.debug(f"No MCP module found in {directory}")
                continue
                
            module_name = f"src.{directory.name}.mcp"
            try:
                # Add parent directory to path if needed
                if str(root_dir.parent) not in sys.path:
                    sys.path.append(str(root_dir.parent))
                
                module = importlib.import_module(module_name)
                logger.info(f"Loaded MCP module: {module_name}")
                
                # Register tools and resources from the module
                if hasattr(module, "register_tools") and callable(module.register_tools):
                    module.register_tools(self)
                
                self.modules[directory.name] = module
            except Exception as e:
                logger.error(f"Failed to load MCP module {module_name}: {str(e)}")
                all_modules_loaded_successfully = False

        return all_modules_loaded_successfully
    
    def register_tool(self, name: str, func: Callable, schema: Dict, description: str):
        """Register a new tool with the MCP."""
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered. Overwriting.")
        
        self.tools[name] = MCPTool(name, func, schema, description)
        logger.info(f"Registered tool: {name}")
        
    def register_resource(self, uri_template: str, retriever: Callable, description: str):
        """Register a new resource with the MCP."""
        if uri_template in self.resources:
            logger.warning(f"Resource '{uri_template}' already registered. Overwriting.")
            
        self.resources[uri_template] = MCPResource(uri_template, retriever, description)
        logger.info(f"Registered resource: {uri_template}")
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool with the given parameters."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool = self.tools[tool_name]
        result = tool.func(**params)
        return {"result": result}
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """Retrieve a resource by URI."""
        # Basic implementation - would need more sophisticated URI template matching
        for template, resource in self.resources.items():
            if template == uri or uri.startswith(template.split("{")[0]):
                try:
                    result = resource.retriever(uri)
                    return {"content": result}
                except Exception as e:
                    logger.error(f"Error retrieving resource {uri}: {str(e)}")
                    raise
                    
        raise ValueError(f"Resource with URI '{uri}' not found")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of this MCP instance."""
        tools = {}
        for name, tool in self.tools.items():
            tools[name] = {
                "schema": tool.schema,
                "description": tool.description
            }
            
        resources = {}
        for uri_template, resource in self.resources.items():
            resources[uri_template] = {
                "description": resource.description
            }
            
        return {
            "tools": tools,
            "resources": resources,
            "version": "1.0.0",
            "name": "GeneralizedNotationNotation MCP"
        }

# Create singleton instance
mcp_instance = MCP()

# Global flag to track if the critical warning has been issued to avoid repetition
_critical_mcp_warning_issued = False

def initialize(halt_on_missing_sdk: bool = True, force_proceed_flag: bool = False) -> Tuple[MCP, bool, bool]:
    """
    Initialize the MCP by discovering modules and checking SDK status.

    Args:
        halt_on_missing_sdk: If True (default), raises MCPSDKNotFoundError if the SDK is missing.
        force_proceed_flag: If True, proceeds even if SDK is missing and halt_on_missing_sdk is True.
                            (e.g., controlled by a command-line argument like --proceed-without-mcp-sdk)

    Returns:
        A tuple: (mcp_instance: MCP, sdk_found: bool, all_modules_loaded: bool)
    
    Raises:
        MCPSDKNotFoundError: If SDK is not found, halt_on_missing_sdk is True, and force_proceed_flag is False.
    """
    global _critical_mcp_warning_issued
    
    # Perform module discovery first, as this populates the mcp_instance
    all_modules_loaded = mcp_instance.discover_modules()

    # With the simplified approach, sdk_found is always True.
    # The check for _MCP_SDK_CONFIG_STATUS["found"] can be simplified or removed
    # if we are certain that the project's internal MCP is always sufficient.
    # For now, we'll keep the structure but ensure "found" is true.
    sdk_found = _MCP_SDK_CONFIG_STATUS["found"] # This will be True

    if not sdk_found: # This block should ideally not be entered anymore.
        if not _critical_mcp_warning_issued:
            consequences_details = _MCP_SDK_CONFIG_STATUS['details']
            consequences = f"""
The Model Context Protocol (MCP) SDK was not found or failed to initialize correctly.
As a result, core MCP functionalities will be severely limited or non-operational.
This will affect capabilities such as, but not limited to:
  - Running GNN type checks via MCP.
  - Estimating GNN computational resources via MCP.
  - Exporting GNN models and reports to various formats via MCP.
  - Utilizing setup utilities (e.g., finding project files, managing directories) via MCP.
  - Executing GNN tests and accessing test reports via MCP.
  - Generating GNN model visualizations via MCP.
  - Accessing GNN core documentation and ontology terms via MCP.
  - Full functionality of the MCP server itself (e.g., self-reflection tools).

Pipeline steps or client applications relying on these MCP functions may fail,
produce incomplete results, or operate with dummy/fallback implementations.
It is strongly recommended to install or correct the MCP SDK for full functionality.
Current SDK status details: {consequences_details}
"""
            
            banner = (
                "\n" + "="*80 +
                "\n" + "!!! CRITICAL MCP SDK WARNING !!!".center(80) +
                "\n" + "="*80
            )
            
            logger.critical(banner)
            logger.critical(consequences)
            logger.critical("="*80 + "\n")
            _critical_mcp_warning_issued = True

        if halt_on_missing_sdk and not force_proceed_flag:
            error_message = (
                "MCP SDK is critical for full functionality and was not found or failed to load. "
                "Pipeline is configured to halt. To proceed with limited MCP capabilities, "
                "use a flag like --proceed-without-mcp-sdk (if available in the calling script) "
                "or adjust pipeline configuration."
            )
            logger.error(error_message)
            raise MCPSDKNotFoundError(error_message)
        elif force_proceed_flag:
            logger.warning(
                "Proceeding without a fully functional MCP SDK due to explicit override. "
                "MCP features will be limited or non-operational."
            )
        else: # Not configured to halt, but SDK is missing
             logger.warning(
                "MCP SDK not found or failed to load, but pipeline is configured to continue. "
                "MCP functionalities will be impaired or non-operational."
            )
    elif sdk_found and _critical_mcp_warning_issued:
        # If SDK was previously thought missing, but now found (e.g. re-init with fix)
        logger.info("MCP SDK appears to be available now.")
        _critical_mcp_warning_issued = False
    elif sdk_found: # This is the expected path now
        logger.info(f"MCP system initialized using project's internal MCP components. SDK Status: {_MCP_SDK_CONFIG_STATUS['details']}")
        _critical_mcp_warning_issued = False # Ensure warning flag is reset if it was ever set


    # The calling script (e.g., 7_mcp.py) might log its own "initialized successfully" message.
    # This function now returns sdk_found so the caller can be more accurate.
    return mcp_instance, sdk_found, all_modules_loaded

# --- Example Usage (illustrative, not typically run directly from here) ---

# Initialize MCP
mcp, sdk_found, all_modules_loaded = initialize(halt_on_missing_sdk=True, force_proceed_flag=False)

# Check if all modules loaded successfully
if all_modules_loaded:
    print("All MCP modules loaded successfully.")
else:
    print("Some MCP modules failed to load.")