from pathlib import Path
import logging
from typing import Dict, Any

# The generator module is imported for its core function.
# It is assumed that the generator does not import this mcp.py file to avoid circular dependencies.
from .generator import generate_html_report

logger = logging.getLogger(__name__)

def generate_pipeline_summary_website_mcp(output_dir: str, website_output_filename: str, verbose: bool = False) -> Dict[str, Any]:
    """
    MCP tool to generate a single HTML website summarizing all contents of the GGN pipeline output directory.
    
    This function is a wrapper around the core website generation logic to make it compatible with the MCP.
    """
    # Get the logger for the generator module to control its verbosity
    generator_logger = logging.getLogger("src.website.generator")
    if verbose:
        generator_logger.setLevel(logging.DEBUG)
    else:
        generator_logger.setLevel(logging.INFO)

    try:
        output_dir_path = Path(output_dir).resolve()
        website_output_file_path = output_dir_path / website_output_filename

        if not output_dir_path.is_dir():
            error_msg = f"Output directory does not exist: {output_dir_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Ensure the parent directory for the website file exists
        website_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Call the core report generation function
        generate_html_report(output_dir_path, website_output_file_path)
        
        success_msg = f"HTML website generated successfully at {website_output_file_path}"
        logger.info(success_msg)
        return {"success": True, "message": success_msg, "website_path": str(website_output_file_path)}
    except Exception as e:
        error_msg = f"An unexpected error occurred during website generation: {str(e)}"
        logger.exception(error_msg)  # Log the full traceback for debugging
        return {"success": False, "error": error_msg, "error_type": type(e).__name__}

def register_tools(mcp_instance):
    """Registers the site generation tool with the MCP instance."""
    schema = {
        "type": "object",
        "properties": {
            "output_dir": {"type": "string", "description": "The main pipeline output directory to scan for results."},
            "site_output_filename": {"type": "string", "description": "The filename for the output HTML report (e.g., 'summary.html')."},
            "verbose": {"type": "boolean", "description": "Enable verbose logging for the generator."}
        },
        "required": ["output_dir", "site_output_filename"]
    }

    mcp_instance.register_tool(
        name="generate_pipeline_summary_site",
        func=generate_pipeline_summary_website_mcp,
        schema=schema,
        description="Generates a single HTML website summarizing all contents of the GNN pipeline output directory."
    )
    logger.info("Site generation MCP tool registered successfully.")
