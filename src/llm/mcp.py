import logging
from pathlib import Path
from typing import Optional

# Attempt to import the main MCP instance and Tool class
# This structure assumes mcp.py is in src/ and this file is in src/llm/
try:
    from mcp import mcp_instance, MCPSDKTool, MCPSDKNotFoundError
    from llm import llm_operations # For direct calls if needed, or just for type hints
except ImportError:
    # Fallback for standalone execution or if src isn't directly in path correctly for this module
    try:
        # This might happen if running a script directly from within src/llm/
        # or if the project structure isn't added to PYTHONPATH as expected by main.py
        import sys
        # Add project root (parent of src) to sys.path to find top-level 'mcp' and 'llm'
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.mcp import mcp_instance, MCPSDKTool, MCPSDKNotFoundError 
        from src.llm import llm_operations
    except ImportError as e_fallback:
        logging.getLogger(__name__).critical(f"Failed to import MCP components or llm_operations in src/llm/mcp.py: {e_fallback}")
        # Define dummy/placeholder if import fails to prevent crashing if file is imported
        mcp_instance = None
        MCPSDKTool = type('MCPSDKTool', (object,), {})
        llm_operations = None
        MCPSDKNotFoundError = Exception

logger = logging.getLogger(__name__)

# Ensure API key is loaded when this module is loaded, 
# so tools are ready if MCP server starts.
# This will raise ValueError if key is not found, which is good to catch early.
if llm_operations:
    try:
        llm_operations.load_api_key()
    except ValueError as e:
        logger.error(f"MCP for LLM: OpenAI API Key not loaded: {e}. LLM tools will not function.")
        # mcp_instance might not be available if imports failed, so check
        if mcp_instance:
            mcp_instance.sdk_status = False # Indicate SDK (LLM part) is not ready
            mcp_instance.sdk_status_message = f"OpenAI API Key not loaded: {e}. LLM tools will not function."

# --- Tool Definitions ---

def summarize_gnn_file_content(file_path_str: str, user_prompt_suffix: Optional[str] = None) -> str:
    """
    Reads a GNN file, sends its content to an LLM, and returns a summary.
    An optional user prompt suffix can be added to guide the summary.
    """
    if not llm_operations:
        return "Error: LLM operations module not loaded."

    file_path = Path(file_path_str)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path_str}")
        return f"Error: File not found at {file_path_str}"

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
        return "Error: LLM operations module not loaded."

    file_path = Path(file_path_str)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path_str}")
        return f"Error: File not found at {file_path_str}"

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
        return "Error: LLM operations module not loaded."

    file_path = Path(file_path_str)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path_str}")
        return f"Error: File not found at {file_path_str}"

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
    if not mcp_instance_ref:
        logger.warning("MCP instance not available. LLM tools not registered.")
        return

    tools_to_register = [
        MCPSDKTool(
            name="llm.summarize_gnn_file",
            description="Reads a GNN specification file and uses an LLM to generate a concise summary of its content. Optionally, a user prompt suffix can refine the summary focus.",
            category=TOOL_CATEGORY,
            func=summarize_gnn_file_content,
            arg_descriptions={
                "file_path_str": "The absolute or relative path to the GNN file (.md, .gnn.md, .json).",
                "user_prompt_suffix": "(Optional) Additional instructions or focus points for the summary."
            }
        ),
        MCPSDKTool(
            name="llm.explain_gnn_file",
            description="Reads a GNN specification file and uses an LLM to generate an explanation of its content. Can focus on a specific aspect if provided.",
            category=TOOL_CATEGORY,
            func=explain_gnn_file_content,
            arg_descriptions={
                "file_path_str": "The absolute or relative path to the GNN file.",
                "aspect_to_explain": "(Optional) A specific part or concept within the GNN to focus the explanation on."
            }
        ),
        MCPSDKTool(
            name="llm.generate_professional_summary",
            description="Reads a GNN file and optional experiment details, then uses an LLM to generate a professional summary suitable for reports or papers.",
            category=TOOL_CATEGORY,
            func=generate_professional_summary_from_gnn,
            arg_descriptions={
                "file_path_str": "The absolute or relative path to the GNN file.",
                "experiment_details": "(Optional) Text describing the experiments conducted with the model, including setup, results, or observations.",
                "target_audience": "(Optional) The intended audience for the summary (e.g., 'fellow researchers', 'project managers'). Default: 'fellow researchers'."
            }
        )
    ]

    for tool in tools_to_register:
        try:
            mcp_instance_ref.register_tool(tool)
            logger.info(f"Successfully registered LLM tool: {tool.name}")
        except Exception as e:
            logger.error(f"Failed to register LLM tool {tool.name}: {e}", exc_info=True)

# Attempt to register tools if mcp_instance was successfully imported
if mcp_instance:
    register_tools(mcp_instance)
else:
    logger.warning("MCP instance not imported correctly; LLM tools cannot be registered at module load time.")
    logger.warning("LLM tools might be registered later if the main MCP system initializes this module.")

# To allow main.py or 7_mcp.py to explicitly call registration again after full MCP init:
def ensure_llm_tools_registered():
    if mcp_instance and llm_operations:
        # Check if API key is loaded, as it might have failed silently earlier
        # or if this is called before the initial load attempt in this file.
        try:
            llm_operations.load_api_key() # Reload or confirm
            if mcp_instance.sdk_status is False and "OpenAI API Key not loaded" in mcp_instance.sdk_status_message:
                 mcp_instance.sdk_status = True # Assume it's good now
                 mcp_instance.sdk_status_message = "OpenAI API Key re-check: OK"
        except ValueError as e:
            logger.error(f"Re-check API Key: OpenAI API Key not loaded: {e}. LLM tools cannot be registered/function.")
            if mcp_instance:
                mcp_instance.sdk_status = False
                mcp_instance.sdk_status_message = f"OpenAI API Key not loaded on re-check: {e}."
            return False # Cannot register if key is still not there
        
        logger.info("Attempting to ensure LLM tools are registered with MCP.")
        register_tools(mcp_instance)
        return True
    else:
        logger.warning("MCP instance or llm_operations not available for ensure_llm_tools_registered.")
        return False
