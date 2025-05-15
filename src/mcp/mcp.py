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

# --- Custom MCP Exceptions ---
class MCPError(Exception):
    """Base class for MCP related errors."""
    def __init__(self, message, code=-32000, data=None):
        super().__init__(message)
        self.code = code
        self.data = data

class MCPToolNotFoundError(MCPError):
    def __init__(self, tool_name):
        super().__init__(f"Tool '{tool_name}' not found.", code=-32601, data=f"Tool '{tool_name}' not found.")

class MCPResourceNotFoundError(MCPError):
    def __init__(self, uri):
        # Or a custom code, but -32601 (Method not found) can also be used if resources are accessed like methods
        super().__init__(f"Resource '{uri}' not found.", code=-32601, data=f"Resource '{uri}' not found.")

class MCPInvalidParamsError(MCPError):
    def __init__(self, message, details=None):
        super().__init__(message, code=-32602, data=details)

class MCPToolExecutionError(MCPError):
    def __init__(self, tool_name, original_exception):
        super().__init__(f"Error executing tool '{tool_name}': {original_exception}", code=-32000, data=str(original_exception))

class MCPSDKNotFoundError(MCPError): # This was already defined, ensuring it inherits from MCPError
    def __init__(self, message="MCP SDK not found or failed to initialize."):
        super().__init__(message, code=-32001, data=message) # Example custom server error code

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
        self._modules_discovered = False # Flag to track if discovery has run
        # Note: The _MCP_SDK_CONFIG_STATUS is module-level, not instance-level,
        # as SDK availability is a system-wide concern for this MCP setup.
        
    def discover_modules(self) -> bool:
        """Discover and load MCP modules from other directories.

        Returns:
            bool: True if all modules loaded successfully, False otherwise.
        """
        if self._modules_discovered:
            logger.debug("MCP modules already discovered. Skipping redundant discovery.")
            # Return True assuming previous discovery's success state is what matters,
            # or we'd need to store the previous result. For now, if it ran, assume it was handled.
            # To be more precise, one might store the result of the first discovery.
            # However, the function is meant to load modules into self.modules and register tools,
            # which should not be redone.
            return True # Or reflect stored status if available and needed.

        root_dir = Path(__file__).parent.parent
        logger.debug(f"Discovering MCP modules in {root_dir}")
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
                logger.debug(f"Loaded MCP module: {module_name}")
                
                # Special handling for llm module initialization
                if module_name == "src.llm.mcp":
                    if hasattr(module, "initialize_llm_module") and callable(module.initialize_llm_module):
                        logger.debug(f"Calling initialize_llm_module for {module_name}")
                        module.initialize_llm_module(self) # Pass MCP instance
                    else:
                        logger.warning(f"Module {module_name} does not have a callable initialize_llm_module function.")

                # Register tools and resources from the module
                if hasattr(module, "register_tools") and callable(module.register_tools):
                    module.register_tools(self)
                
                self.modules[directory.name] = module
            except Exception as e:
                logger.error(f"Failed to load MCP module {module_name}: {str(e)}")
                all_modules_loaded_successfully = False

        self._modules_discovered = True # Set flag after successful completion of first discovery
        return all_modules_loaded_successfully
    
    def register_tool(self, name: str, func: Callable, schema: Dict, description: str):
        """Register a new tool with the MCP."""
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered. Overwriting.")
        
        self.tools[name] = MCPTool(name, func, schema, description)
        logger.debug(f"Registered tool: {name}")
        
    def register_resource(self, uri_template: str, retriever: Callable, description: str):
        """Register a new resource with the MCP."""
        if uri_template in self.resources:
            logger.warning(f"Resource '{uri_template}' already registered. Overwriting.")
            
        self.resources[uri_template] = MCPResource(uri_template, retriever, description)
        logger.debug(f"Registered resource: {uri_template}")
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool with the given parameters."""
        logger.debug(f"Attempting to execute tool: {tool_name} with params: {params}")
        if tool_name not in self.tools:
            logger.error(f"Tool not found: {tool_name}")
            raise MCPToolNotFoundError(tool_name)
            
        tool = self.tools[tool_name]
        
        # Basic schema validation (can be enhanced with a proper JSON schema validator)
        # For simplicity, this example just checks for required parameters.
        # A real implementation should use a library like jsonschema for full validation.
        if tool.schema and tool.schema.get('properties'):
            required_params = tool.schema.get('required', [])
            for param_name in required_params:
                if param_name not in params:
                    err_msg = f"Missing required parameter for {tool_name}: {param_name}"
                    logger.error(err_msg)
                    raise MCPInvalidParamsError(err_msg, details={"missing_parameter": param_name})
            # Optional: Add type checking here based on schema if not using full jsonschema validation
            for param_name, param_value in params.items():
                if param_name in tool.schema['properties']:
                    expected_type_str = tool.schema['properties'][param_name].get('type')
                    # Basic type mapping - extend as needed
                    type_map = {
                        'string': str,
                        'integer': int,
                        'number': (int, float),
                        'boolean': bool,
                        'array': list,
                        'object': dict
                    }
                    if expected_type_str and expected_type_str in type_map:
                        expected_type = type_map[expected_type_str]
                        if not isinstance(param_value, expected_type):
                            err_msg = f"Invalid type for parameter '{param_name}' in tool '{tool_name}'. Expected {expected_type_str}, got {type(param_value).__name__}."
                            logger.error(err_msg)
                            raise MCPInvalidParamsError(err_msg, details={ "parameter": param_name, "expected_type": expected_type_str, "actual_type": type(param_value).__name__})
        
        try:
            # The actual tool function (tool.func) is responsible for its own logic.
            # It should return a dictionary or JSON-serializable data.
            result_data = tool.func(**params)
            logger.info(f"Tool {tool_name} executed successfully.")
            # The MCP spec usually expects the result of the tool directly.
            # The client then wraps this in a JSON-RPC response if it's an MCP client.
            # If this mcp.py is part of a server that forms the full JSON-RPC response,
            # then it might return just `result_data`.
            # The previous code `return {"result": result_data}` implies this method might be
            # called by something that expects the *full* JSON-RPC `result` field content.
            # However, the method is `execute_tool`, not `handle_execute_tool_rpc_request`.
            # For clarity and adhering to what a tool execution means, it should return the tool's direct output.
            # The JSON-RPC formatting ({"jsonrpc": ..., "result": ..., "id": ...}) should be handled by the transport layer.
            return result_data # Return the direct result of the tool function
        except MCPError: # Re-raise MCP-specific errors directly
            raise
        except Exception as e:
            logger.error(f"Unhandled error during execution of tool {tool_name}: {e}", exc_info=True)
            raise MCPToolExecutionError(tool_name, e)
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """Retrieve a resource by URI."""
        logger.debug(f"Attempting to retrieve resource: {uri}")
        # Basic implementation - would need more sophisticated URI template matching
        for template, resource in self.resources.items():
            # This is a very simplified matching logic. 
            # A robust solution would use URI template libraries or regex.
            # Example: if uri matches template pattern (e.g. using re or a template library)
            if template == uri or (template.endswith('{}') and uri.startswith(template[:-2])) or (template.endswith('{id}') and uri.startswith(template[:-4])) :
                try:
                    # The retriever function should return the resource content directly.
                    resource_content = resource.retriever(uri=uri) # Pass the actual URI to the retriever
                    logger.info(f"Resource {uri} retrieved successfully.")
                    # Similar to execute_tool, return the direct content of the resource.
                    # The JSON-RPC formatting should be handled by the transport layer.
                    return resource_content
                except MCPError: # Re-raise MCP-specific errors
                    raise
                except Exception as e:
                    logger.error(f"Error retrieving resource {uri} via retriever for template {template}: {e}", exc_info=True)
                    # Treat retriever failure like a tool execution failure
                    raise MCPToolExecutionError(f"resource_retriever_for_{template}", e) # Use template as a quasi-toolname
                    
        logger.warning(f"Resource with URI '{uri}' not found after checking all templates.")
        raise MCPResourceNotFoundError(uri)
    
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
# if all_modules_loaded:
# print("All MCP modules loaded successfully.")
# else:
# print("Some MCP modules failed to load.")
# Comment out example usage to prevent execution during import