from pathlib import Path
import logging

from src.mcp.mcp import mcp_instance, MCPSchema, MCPTool, MCPToolResponse
from src.site.generator import generate_html_report, logger as generator_logger # Use the logger from generator

logger = logging.getLogger(__name__) # Logger for this MCP module

class GenerateSiteSchema(MCPSchema):
    output_dir: Path
    site_output_file: Path
    verbose: bool = False

class GenerateSiteTool(MCPTool):
    name = "generate_pipeline_summary_site"
    description = "Generates a single HTML website summarizing all contents of the GNN pipeline output directory."
    schema = GenerateSiteSchema

    def handler(self, params: GenerateSiteSchema) -> MCPToolResponse:
        logger.info(f"MCP Tool '{self.name}' invoked with params: output_dir={params.output_dir}, site_output_file={params.site_output_file}")
        
        # Configure generator logger level based on verbose flag from MCP params
        if params.verbose:
            generator_logger.setLevel(logging.DEBUG)
        else:
            generator_logger.setLevel(logging.INFO)
        # Ensure console handler for generator_logger if not already present for MCP context
        if not any(isinstance(h, logging.StreamHandler) for h in generator_logger.handlers):
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            generator_logger.addHandler(ch)
            generator_logger.propagate = False # Avoid double logging if root has handler

        try:
            resolved_output_dir = params.output_dir.resolve()
            resolved_site_output_file = params.site_output_file.resolve()

            if not resolved_output_dir.is_dir():
                error_msg = f"Output directory does not exist: {resolved_output_dir}"
                logger.error(error_msg)
                return MCPToolResponse(status_code=400, message=error_msg, error=True)

            resolved_site_output_file.parent.mkdir(parents=True, exist_ok=True)
            
            generate_html_report(resolved_output_dir, resolved_site_output_file)
            
            success_msg = f"HTML site generated successfully at {resolved_site_output_file}"
            logger.info(success_msg)
            return MCPToolResponse(status_code=200, message=success_msg, data={"site_path": str(resolved_site_output_file)})
        except Exception as e:
            error_msg = f"Error during site generation: {str(e)}"
            logger.exception(error_msg) # Log full traceback
            return MCPToolResponse(status_code=500, message=error_msg, error=True)

# Register the tool with the central MCP instance
if mcp_instance:
    mcp_instance.register_tool(GenerateSiteTool())
    logger.info(f"Tool '{GenerateSiteTool.name}' registered with MCP.")
else:
    logger.warning("MCP instance not found. Cannot register GenerateSiteTool.")
