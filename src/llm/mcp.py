import logging
from pathlib import Path
from typing import Optional
import inspect

# Removed global imports of mcp_instance, MCPTool, MCPSDKNotFoundError to break circular dependency.
# These will be accessed via mcp_instance_ref or imported locally where appropriate.

# Attempt to import llm_operations.
# MCPTool and MCPSDKNotFoundError will be accessed via mcp_instance_ref or later import if possible.
try:
    from llm import llm_operations
except ImportError:
    try:
        import sys
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.llm import llm_operations
    except ImportError as e_fallback:
        logging.getLogger(__name__).critical(f"Failed to import llm_operations in src/llm/mcp.py: {e_fallback}")
        llm_operations = None

logger = logging.getLogger(__name__)

# Global placeholders, to be resolved by the MCP system providing them or through mcp_instance_ref
MCPTool = None
MCPSDKNotFoundError = None

def initialize_llm_module(mcp_instance_ref):
    """
    Initializes the LLM module, loads API key, and updates MCP status.
    This should be called by the MCP main system after it has initialized mcp_instance.
    """
    global llm_operations, MCPTool, MCPSDKNotFoundError # Allow modification of global placeholders

    # Import MCPTool and MCPSDKNotFoundError directly from src.mcp.mcp
    try:
        from src.mcp.mcp import MCPTool as RealMCPTool, MCPSDKNotFoundError as RealMCPSDKNotFoundError
        MCPTool = RealMCPTool
        MCPSDKNotFoundError = RealMCPSDKNotFoundError
        logger.info("Successfully imported MCPTool and MCPSDKNotFoundError from src.mcp.mcp in initialize_llm_module.")
    except ImportError:
        logger.warning("Could not import MCPTool or MCPSDKNotFoundError from src.mcp.mcp in initialize_llm_module. Tool registration might fail.")
        # Keep them as None or a dummy type if not found
        if MCPTool is None: MCPTool = type('MCPTool', (object,), {}) # Dummy class
        if MCPSDKNotFoundError is None: MCPSDKNotFoundError = type('MCPSDKNotFoundError', (Exception,), {}) # Dummy exception

    if not llm_operations:
        logger.error("LLM operations module not loaded. LLM tools cannot be initialized or registered.")
        if mcp_instance_ref and hasattr(mcp_instance_ref, 'sdk_status'):
            mcp_instance_ref.sdk_status = False
            mcp_instance_ref.sdk_status_message = "LLM operations module not loaded."
        return False

    try:
        llm_operations.load_api_key()
        logger.info("LLM API Key loaded successfully.")
        if mcp_instance_ref and hasattr(mcp_instance_ref, 'sdk_status'):
            mcp_instance_ref.sdk_status = True # Assuming successful load means SDK is ready
            mcp_instance_ref.sdk_status_message = "LLM SDK ready (API Key loaded)."
        return True
    except ValueError as e:
        logger.error(f"MCP for LLM: OpenAI API Key not loaded: {e}. LLM tools will not function.")
        if mcp_instance_ref and hasattr(mcp_instance_ref, 'sdk_status'):
            mcp_instance_ref.sdk_status = False # Indicate SDK (LLM part) is not ready
            mcp_instance_ref.sdk_status_message = f"OpenAI API Key not loaded: {e}. LLM tools will not function."
        return False

# --- Tool Definitions ---

def summarize_gnn_file_content(file_path_str: str, user_prompt_suffix: Optional[str] = None) -> str:
    """
    Reads a GNN file, sends its content to an LLM, and returns a summary.
    An optional user prompt suffix can be added to guide the summary.
    """
    if not llm_operations:
        error_msg = "Error: LLM operations module not loaded."
        logger.error(f"summarize_gnn_file_content: {error_msg}")
        return error_msg

    file_path = Path(file_path_str)
    if not file_path.is_file():
        error_msg = f"File not found at {file_path_str}"
        logger.error(f"summarize_gnn_file_content: {error_msg}")
        return f"Error: {error_msg}"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gnn_content = f.read()
        
        contexts = [f"GNN File Content ({file_path.name}):\n{gnn_content}"]
        task = "Provide a concise summary of the GNN model described in the content above, highlighting its key components (ModelName, primary states/observations, and main connections)."
        if user_prompt_suffix:
            task += f" {user_prompt_suffix}"
            
        prompt = llm_operations.construct_prompt(contexts, task)
        summary = llm_operations.get_llm_response(prompt)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing GNN file {file_path_str}: {e}", exc_info=True)
        return f"Error processing file {file_path_str}: {e}"

def explain_gnn_file_content(file_path_str: str, aspect_to_explain: Optional[str] = None) -> str:
    """
    Reads a GNN file, sends its content to an LLM, and returns an explanation.
    If aspect_to_explain is provided, the explanation focuses on that part.
    """
    if not llm_operations:
        error_msg = "Error: LLM operations module not loaded."
        logger.error(f"explain_gnn_file_content: {error_msg}")
        return error_msg

    file_path = Path(file_path_str)
    if not file_path.is_file():
        error_msg = f"File not found at {file_path_str}"
        logger.error(f"explain_gnn_file_content: {error_msg}")
        return f"Error: {error_msg}"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gnn_content = f.read()

        contexts = [f"GNN File Content ({file_path.name}):\n{gnn_content}"]
        if aspect_to_explain:
            task = f"Explain the following aspect of the GNN model: '{aspect_to_explain}'. Provide a clear and simple explanation suitable for someone familiar with GNNs but perhaps not this specific model."
        else:
            task = "Provide a general explanation of the GNN model described above. Cover its potential purpose, the nature of its state space, and how its components might interact."
        
        prompt = llm_operations.construct_prompt(contexts, task)
        explanation = llm_operations.get_llm_response(prompt)
        return explanation
    except Exception as e:
        logger.error(f"Error explaining GNN file {file_path_str}: {e}", exc_info=True)
        return f"Error processing file {file_path_str}: {e}"

def generate_professional_summary_from_gnn(file_path_str: str, experiment_details: Optional[str] = None, target_audience: str = "fellow researchers") -> str:
    """
    Generates a professional summary of a GNN model and its experimental context.
    Useful for reports or presentations.
    """
    if not llm_operations:
        error_msg = "Error: LLM operations module not loaded."
        logger.error(f"generate_professional_summary_from_gnn: {error_msg}")
        return error_msg

    file_path = Path(file_path_str)
    if not file_path.is_file():
        error_msg = f"File not found at {file_path_str}"
        logger.error(f"generate_professional_summary_from_gnn: {error_msg}")
        return f"Error: {error_msg}"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gnn_content = f.read()
        
        contexts = [f"GNN Model Specification ({file_path.name}):\n{gnn_content}"]
        if experiment_details:
            contexts.append(f"Experimental Context/Results:\n{experiment_details}")
        
        task = f"Generate a professional, publication-quality summary of the GNN model and its experimental context. The summary should be targeted at {target_audience}. It should be well-structured, highlight key findings or model characteristics, and be suitable for inclusion in a research paper or technical report."
        
        prompt = llm_operations.construct_prompt(contexts, task)
        # Potentially use a more capable model or higher token limit for professional summaries
        prof_summary = llm_operations.get_llm_response(prompt, model="gpt-4o-mini")
        return prof_summary
    except Exception as e:
        logger.error(f"Error generating professional summary for {file_path_str}: {e}", exc_info=True)
        return f"Error processing file {file_path_str}: {e}"

# --- Tool Registration (if mcp_instance is available) ---
TOOL_CATEGORY = "LLM Operations"

def register_tools(mcp_instance_ref):
    global MCPTool # Use the MCPTool that initialize_llm_module is responsible for setting

    if not mcp_instance_ref:
        logger.warning("MCP instance not available. LLM tools not registered.")
        return
    
    if MCPTool is None: # This means initialize_llm_module failed to set it (even to a dummy)
        logger.error("MCPTool is None after initialize_llm_module was expected to set it. LLM tools registration aborted.")
        return

    # At this point, MCPTool is either the real MCPTool class from src.mcp.mcp
    # or the dummy class type('MCPTool', (object,), {}) if the import failed in initialize_llm_module.
    # We proceed with the assumption that MCPTool is callable and usable for instantiation.
    # If it's the dummy and incompatible with instantiation or registration,
    # the try-except block below during tool creation/registration will catch it.

    tool_definitions = [
        {
            "name": "llm.summarize_gnn_file",
            "description": "Reads a GNN specification file and uses an LLM to generate a concise summary of its content. Optionally, a user prompt suffix can refine the summary focus.",
            "func": summarize_gnn_file_content,
            "arg_descriptions": {
                "file_path_str": "The absolute or relative path to the GNN file (.md, .gnn.md, .json).",
                "user_prompt_suffix": "(Optional) Additional instructions or focus points for the summary."
            }
        },
        {
            "name": "llm.explain_gnn_file",
            "description": "Reads a GNN specification file and uses an LLM to generate an explanation of its content. Can focus on a specific aspect if provided.",
            "func": explain_gnn_file_content,
            "arg_descriptions": {
                "file_path_str": "The absolute or relative path to the GNN file.",
                "aspect_to_explain": "(Optional) A specific part or concept within the GNN to focus the explanation on."
            }
        },
        {
            "name": "llm.generate_professional_summary",
            "description": "Reads a GNN file and optional experiment details, then uses an LLM to generate a professional summary suitable for reports or papers.",
            "func": generate_professional_summary_from_gnn,
            "arg_descriptions": {
                "file_path_str": "The absolute or relative path to the GNN file.",
                "experiment_details": "(Optional) Text describing the experiments conducted with the model, including setup, results, or observations.",
                "target_audience": "(Optional) The intended audience for the summary (e.g., 'fellow researchers', 'project managers'). Default: 'fellow researchers'."
            }
        }
    ]

    for tool_def in tool_definitions:
        try:
            properties = {}
            required_params = []
            
            sig = inspect.signature(tool_def["func"])
            for param_name, param in sig.parameters.items():
                desc = tool_def["arg_descriptions"].get(param_name, "")
                param_type_str = "string"
                if param.annotation == str:
                    param_type_str = "string"
                elif param.annotation == int:
                    param_type_str = "integer"
                elif param.annotation == float:
                    param_type_str = "number"
                elif param.annotation == bool:
                    param_type_str = "boolean"
                elif param.annotation == list:
                    param_type_str = "array"
                elif param.annotation == dict:
                    param_type_str = "object"
                elif param.annotation == Path:
                    param_type_str = "string"
                
                properties[param_name] = {"type": param_type_str, "description": desc}
                if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD and param.kind != inspect.Parameter.VAR_POSITIONAL:
                    # Consider a parameter required if it has no default and is not *args or **kwargs
                    # Further refinement: check if type hint is Optional or includes None
                    if not (str(param.annotation).startswith("Optional[") or ("None" in str(param.annotation))):
                         required_params.append(param_name)
            
            # Attempt to create and register the tool
            # MCPTool instance is created internally by mcp_instance_ref.register_tool
            mcp_instance_ref.register_tool(
                name=tool_def["name"],
                func=tool_def["func"],
                schema={
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                },
                description=tool_def["description"]
            )
            logger.info(f"Registered MCP tool: {tool_def['name']}")
        except Exception as e_reg:
            logger.error(f"Failed to register MCP tool {tool_def['name']}: {e_reg}", exc_info=True)
    
    logger.info("LLM module MCP tools registration process completed.")

def ensure_llm_tools_registered(mcp_instance_ref): # Added mcp_instance_ref parameter
    """
    Ensures that LLM tools are registered with the provided MCP instance.
    This function can be called from the main LLM processing script (e.g., 11_llm.py)
    to make sure tools are available before use, especially if MCP initialization
    is complex or happens in stages.
    """
    logger.info("Attempting to ensure LLM tools are registered with MCP.")
    
    # First, ensure the LLM module itself is initialized (loads API key, attempts to set MCPTool)
    # Pass the mcp_instance_ref so initialize_llm_module can also access it if needed (e.g. for sdk_status)
    if not initialize_llm_module(mcp_instance_ref):
        logger.warning("LLM module initialization failed. Tools may not register or function.")
        # Even if initialization has issues, try to proceed with registration if MCPTool was set somehow

    # Check API key status via llm_operations if possible
    if llm_operations and hasattr(llm_operations, 'is_api_key_loaded') and not llm_operations.is_api_key_loaded():
        logger.warning("LLM API key is not loaded. LLM tools will not function correctly even if registered.")
    elif not llm_operations:
        logger.warning("llm_operations module not available for API key check.")

    # Now, attempt to register tools
    # register_tools itself uses the global MCPTool set by initialize_llm_module
    register_tools(mcp_instance_ref)
    logger.info("ensure_llm_tools_registered call completed.")

# Example of how this module might be triggered by the main MCP system:
# if __name__ == '__main__':
#     # This is a simplified example. In a real scenario, mcp_instance would come from src.mcp.mcp
#     class MockMCPInstance:
#         def __init__(self):
#             self.tools = {}
#             self.sdk_status = False
#             self.sdk_status_message = "Not initialized"

#         def register_tool(self, tool_instance):
#             logger.info(f"MockMCP: Registering tool {tool_instance.name}")
#             self.tools[tool_instance.name] = tool_instance
        
#         def get_tool(self, tool_name):
#             return self.tools.get(tool_name)

#     logging.basicConfig(level=logging.INFO)
#     mock_mcp = MockMCPInstance()
    
#     # 1. Initialize the LLM module (which tries to load API key and set up MCPTool)
#     initialize_llm_module(mock_mcp) 
    
#     # 2. Register tools using the (now hopefully set) MCPTool
#     register_tools(mock_mcp)

#     # 3. (Optional) Verify by trying to get a tool
#     summarizer = mock_mcp.get_tool("llm.summarize_gnn_file")
#     if summarizer:
#         logger.info(f"Successfully retrieved tool: {summarizer.name}")
#     else:
#         logger.error("Failed to retrieve tool after registration attempt.")
