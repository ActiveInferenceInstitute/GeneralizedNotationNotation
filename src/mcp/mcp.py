import importlib
import os
import sys
from pathlib import Path
import logging
import inspect
from typing import Dict, List, Any, Callable, Optional, TypedDict, Union, Tuple
import importlib.util # For dynamic module loading

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp")

# Global instance of MCP
# This needs to be initialized once.
mcp_instance: Optional['MCP'] = None # Added type hint
_mcp_initialized_globally = False # Global flag for the initialize() function
_sdk_status_globally = False
_all_modules_loaded_globally = False

# --- Custom MCP Exceptions ---
class MCPError(Exception):
    """Base class for MCP related errors."""
    def __init__(self, message, code=-32000, data=None):
        super().__init__(message)
        self.code = code
        self.data = data
        self.message = message # Ensure message is stored

class MCPToolNotFoundError(MCPError):
    """Raised when a specified MCP tool is not found."""
    def __init__(self, tool_name):
        super().__init__(f"Tool '{tool_name}' not found.", code=-32601) # JSON-RPC standard code

class MCPResourceNotFoundError(MCPError):
    """Raised when a specified MCP resource URI does not match any registered retriever."""
    def __init__(self, uri):
        # Or a custom code, but -32601 (Method not found) can also be used if resources are accessed like methods
        super().__init__(f"Resource with URI '{uri}' not found.", code=-32601) 

class MCPInvalidParamsError(MCPError):
    """Raised when parameters provided for an MCP tool are invalid."""
    def __init__(self, message, details=None):
        super().__init__(message, code=-32602, data=details) # JSON-RPC standard code

class MCPToolExecutionError(MCPError):
    """Raised when an MCP tool encounters an error during its execution."""
    def __init__(self, tool_name, original_exception):
        super().__init__(f"Error executing tool '{tool_name}': {original_exception}", code=-32000) # Generic server error
        self.original_exception = original_exception

class MCPSDKNotFoundError(MCPError): # This was already defined, ensuring it inherits from MCPError
    """Raised when a required SDK component for MCP is not found or fails to initialize."""
    def __init__(self, message="MCP SDK not found or failed to initialize."):
        super().__init__(message, code=-32001) # Custom server error code for SDK issues

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
        self.logger = logging.getLogger("mcp") # Standardized logger name
        self.project_root_dir: Optional[Path] = None # Store project root, set by initialize()
        
    def discover_modules(self) -> bool:
        """
        Dynamically discovers and loads MCP integration modules from expected subdirectories
        within the project\'s \'src/\' directory.
        This method is idempotent.
        
        Returns:
            bool: True if all expected modules with mcp.py were loaded successfully, False otherwise.
        """
        if self._modules_discovered:
            self.logger.info("MCP modules already discovered. Skipping redundant discovery.")
            # To be fully idempotent and consistent, we should return the status of the *first* discovery.
            # This requires storing it. For now, assume previous success or rely on _all_modules_loaded_globally.
            global _all_modules_loaded_globally
            return _all_modules_loaded_globally

        if not self.project_root_dir:
            self.logger.error("Project root directory not set in MCP instance. Cannot discover MCP modules.")
            return False
            
        src_dir = self.project_root_dir / "src"
        self.logger.info(f"Discovering MCP modules in {src_dir}")
        
        expected_mcp_module_dirs = [
            "export", "gnn", "gnn_type_checker", "llm", 
            "ontology", "render", "setup", "tests", "visualization",
            "mcp" # For meta_mcp.py
        ]
        
        all_successfully_loaded_this_time = True

        for module_name in expected_mcp_module_dirs:
            module_dir_path = src_dir / module_name
            mcp_py_file = module_dir_path / "mcp.py"
            full_module_dot_path = f"src.{module_name}.mcp"

            if module_name == "mcp": # Special handling for the mcp module itself
                mcp_py_file = module_dir_path / "meta_mcp.py"
                full_module_dot_path = f"src.mcp.meta_mcp" # Correctly target meta_mcp

            if mcp_py_file.exists() and mcp_py_file.is_file():
                try:
                    # Ensure the module (and its parent packages like 'src' and 'src.module_name')
                    # are correctly represented in sys.modules if this is the first time.
                    # This logic assumes \'src\' is a package visible in sys.path.
                    
                    # Create spec
                    spec = importlib.util.spec_from_file_location(full_module_dot_path, mcp_py_file)
                    if not spec or not spec.loader:
                        self.logger.warning(f"Could not create import spec for MCP module: {full_module_dot_path} from {mcp_py_file}")
                        all_successfully_loaded_this_time = False
                        continue
                        
                    # Create and load module
                    module_obj = importlib.util.module_from_spec(spec)
                    sys.modules[full_module_dot_path] = module_obj # Crucial: add to sys.modules BEFORE exec_module
                    spec.loader.exec_module(module_obj)
                    
                    # --- ADDED: Special handling for LLM module initialization ---
                    if full_module_dot_path == "src.llm.mcp" and hasattr(module_obj, "initialize_llm_module"):
                        try:
                            self.logger.info(f"Calling initialize_llm_module for {full_module_dot_path}")
                            # Pass self (the main MCP instance) to the LLM module's initializer
                            module_obj.initialize_llm_module(self) 
                        except Exception as e_init_llm:
                            self.logger.error(f"Error calling initialize_llm_module for {full_module_dot_path}: {e_init_llm}", exc_info=True)
                            # We might want to set all_successfully_loaded_this_time = False if LLM init is critical
                            all_successfully_loaded_this_time = False 
                    # --- END ADDED ---
                    
                    if hasattr(module_obj, "register_tools"):
                        try:
                            # Pass self (the MCP instance) to the module\'s register_tools
                            module_obj.register_tools(self) 
                            self.logger.info(f"Loaded and registered tools from MCP module: {full_module_dot_path}")
                        except Exception as e_reg:
                            self.logger.error(f"Error calling register_tools for {full_module_dot_path}: {e_reg}", exc_info=True)
                            all_successfully_loaded_this_time = False
                    else:
                        self.logger.warning(f"Module {full_module_dot_path} exists but does not have a \'register_tools\' function.")
                        # Depending on policy, this might not set all_successfully_loaded_this_time to False,
                        # if mcp.py without register_tools is permissible for some modules.
                        # For now, let\'s consider it a warning but not a failure of this function.
                        
                except Exception as e_load:
                    self.logger.error(f"Error loading MCP module {full_module_dot_path} from {mcp_py_file}: {e_load}", exc_info=True)
                    all_successfully_loaded_this_time = False
            # else:
                # self.logger.debug(f"No mcp.py found in module directory: {module_dir_path}. Skipping.")
                # This is normal for modules not participating in MCP, so not a warning.

        if all_successfully_loaded_this_time:
            self.logger.info("All discoverable MCP modules processed.")
            self._modules_discovered = True # Mark as discovered only if all attempts were made (even if some failed to load tools)
                                        # The return value indicates if all *loaded successfully*.
        else:
            self.logger.warning("One or more MCP modules encountered issues during loading or tool registration.")
            # Still set _modules_discovered to True because the discovery *process* was completed.
            # The success/failure of loading individual modules is what all_successfully_loaded_this_time tracks.
            self._modules_discovered = True 
            
        return all_successfully_loaded_this_time
    
    def register_tool(self, name: str, func: Callable, schema: Dict, description: str):
        """Register a new tool with the MCP."""
        if name in self.tools:
            self.logger.warning(f"Tool '{name}' is being re-registered. Overwriting existing tool.")
        
        self.tools[name] = MCPTool(name, func, schema, description)
        self.logger.info(f"Registered tool: {name}")
        
    def register_resource(self, uri_template: str, retriever: Callable, description: str):
        """Register a new resource with the MCP."""
        if uri_template in self.resources:
            self.logger.warning(f"Resource retriever for URI template '{uri_template}' is being re-registered.")
            
        self.resources[uri_template] = MCPResource(uri_template, retriever, description)
        self.logger.info(f"Registered resource: {uri_template}")
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool with the given parameters."""
        self.logger.info(f"Executing tool: {tool_name} with params: {params if params is not None else {}}")
        if tool_name not in self.tools:
            self.logger.error(f"Tool not found: {tool_name}")
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
                    self.logger.error(err_msg)
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
                            self.logger.error(err_msg)
                            raise MCPInvalidParamsError(err_msg, details={ "parameter": param_name, "expected_type": expected_type_str, "actual_type": type(param_value).__name__})
        
        try:
            # The actual tool function (tool.func) is responsible for its own logic.
            # It should return a dictionary or JSON-serializable data.
            result_data = tool.func(**params)
            self.logger.info(f"Tool {tool_name} executed successfully.")
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
            self.logger.error(f"Unhandled error during execution of tool {tool_name}: {e}", exc_info=True)
            raise MCPToolExecutionError(tool_name, e)
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """Retrieve a resource by URI."""
        self.logger.info(f"Attempting to retrieve resource: {uri}")
        # Basic implementation - would need more sophisticated URI template matching
        for template, resource in self.resources.items():
            # This is a very simplified matching logic. 
            # A robust solution would use URI template libraries or regex.
            # Example: if uri matches template pattern (e.g. using re or a template library)
            if template == uri or (template.endswith('{}') and uri.startswith(template[:-2])) or (template.endswith('{id}') and uri.startswith(template[:-4])) :
                try:
                    # The retriever function should return the resource content directly.
                    resource_content = resource.retriever(uri=uri) # Pass the actual URI to the retriever
                    self.logger.info(f"Resource {uri} retrieved successfully.")
                    # Similar to execute_tool, return the direct content of the resource.
                    # The JSON-RPC formatting should be handled by the transport layer.
                    return resource_content
                except MCPError: # Re-raise MCP-specific errors
                    raise
                except Exception as e:
                    self.logger.error(f"Error retrieving resource {uri} via retriever for template {template}: {e}", exc_info=True)
                    # Treat retriever failure like a tool execution failure
                    raise MCPToolExecutionError(f"resource_retriever_for_{template}", e) # Use template as a quasi-toolname
                    
        self.logger.warning(f"Resource with URI '{uri}' not found after checking all templates.")
        raise MCPResourceNotFoundError(uri)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of this MCP instance."""
        self.logger.info("Fetching MCP capabilities...")
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
    Initializes the MCP system. This function is now idempotent.
    It discovers and loads tools from all MCP-enabled modules.

    Args:
        halt_on_missing_sdk (bool): If True, raises MCPSDKNotFoundError if essential SDK components are missing.
        force_proceed_flag (bool): If True, attempts to proceed even if SDK components are missing (overrides halt_on_missing_sdk effect).

    Returns:
        Tuple[MCP, bool, bool]: 
            - The MCP instance.
            - SDK status (True if OK or not critical, False if critical and missing).
            - All modules loaded status (True if all discoverable modules loaded tools, False otherwise).
    """
    global mcp_instance, _mcp_initialized_globally, _sdk_status_globally, _all_modules_loaded_globally
    
    logger_init = logging.getLogger("mcp.initialize") # Specific logger for this function

    if _mcp_initialized_globally:
        logger_init.info("MCP system already initialized. Returning existing instance and status.")
        if mcp_instance is None: 
             logger_init.critical("MCP global state inconsistency: _mcp_initialized_globally is True but mcp_instance is None. Reinitializing.")
             # Reset flag to allow re-initialization
             _mcp_initialized_globally = False 
        else:
            return mcp_instance, _sdk_status_globally, _all_modules_loaded_globally

    if mcp_instance is None:
        mcp_instance = MCP()
    
    # Determine and set project_root_dir for mcp_instance
    # This assumes mcp.py is in project_root/src/mcp/
    try:
        current_file_path = Path(__file__).resolve()
        # project_root_dir should be the directory containing \'src\'
        # If mcp.py is src/mcp/mcp.py, then current_file_path.parent is src/mcp, 
        # current_file_path.parent.parent is src, 
        # current_file_path.parent.parent.parent is project_root.
        mcp_instance.project_root_dir = current_file_path.parent.parent.parent
        logger_init.debug(f"MCP Project Root detected as: {mcp_instance.project_root_dir}")
        
        # Add project root to sys.path if not already there, to help with \'from src... import\'
        if str(mcp_instance.project_root_dir) not in sys.path:
            sys.path.insert(0, str(mcp_instance.project_root_dir))
            logger_init.debug(f"Added project root {mcp_instance.project_root_dir} to sys.path for module discovery.")

    except Exception as e:
        logger_init.error(f"Failed to determine project root directory for MCP: {e}. Module discovery may fail.", exc_info=True)
        # If project_root_dir is critical for module discovery, this could be a hard stop or warning.
        # discover_modules() checks for self.project_root_dir.

    # --- SDK Check Placeholder ---
    sdk_found_and_ok = True 
    # Example:
    # try:
    #     # import some_critical_sdk_dependency
    #     logger_init.info("Hypothetical MCP SDK dependency check: OK.")
    # except ImportError:
    #     logger_init.warning("Hypothetical MCP SDK dependency: MISSING.")
    #     sdk_found_and_ok = False
    # --- End SDK Check ---

    if not sdk_found_and_ok:
        sdk_message = "Essential MCP SDK component(s) missing or failed to initialize."
        if halt_on_missing_sdk and not force_proceed_flag:
            logger_init.error(f"{sdk_message} Halting initialization as per configuration.")
            _sdk_status_globally = False
            _all_modules_loaded_globally = False # Modules not loaded due to SDK issue
            _mcp_initialized_globally = True # Mark as "initialization attempted and failed critically"
            if mcp_instance is not None: # Ensure it's returned even on failure
                 return mcp_instance, _sdk_status_globally, _all_modules_loaded_globally
            else: # Should not happen if MCP() was called
                 raise MCPSDKNotFoundError(f"{sdk_message} And mcp_instance is None.")
        else:
            logger_init.warning(f"{sdk_message} Proceeding with MCP initialization as per configuration (halt_on_missing_sdk=False or force_proceed_flag=True).")
    
    # Discover modules - MCP.discover_modules() is now idempotent internally but should only be effectively called once here.
    # The _modules_discovered flag inside MCP instance handles its own idempotency.
    all_modules_loaded = mcp_instance.discover_modules()

    # Log overall status
    sdk_status_msg = "OK" if sdk_found_and_ok else "ISSUES"
    modules_status_msg = "All Loaded" if all_modules_loaded else "Some Failed"
    
    if sdk_found_and_ok and all_modules_loaded:
        logger_init.info(f"MCP system initialized. SDK Status: {sdk_status_msg}. Modules: {modules_status_msg}.")
    else:
        logger_init.warning(f"MCP system initialization ended. SDK Status: {sdk_status_msg}. Modules: {modules_status_msg}. Check logs for details.")

    # Store global status
    _sdk_status_globally = sdk_found_and_ok
    _all_modules_loaded_globally = all_modules_loaded
    _mcp_initialized_globally = True # Mark that the initialization function has completed its first run.

    return mcp_instance, _sdk_status_globally, _all_modules_loaded_globally

# --- Example Usage (illustrative, not typically run directly from here) ---

# Initialize MCP
mcp, sdk_found, all_modules_loaded = initialize(halt_on_missing_sdk=True, force_proceed_flag=False)

# Check if all modules loaded successfully
if all_modules_loaded:
    print("All MCP modules loaded successfully.")
else:
    print("Some MCP modules failed to load.")