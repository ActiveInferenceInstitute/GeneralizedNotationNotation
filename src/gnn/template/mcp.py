"""
Template Step MCP Integration

This module provides Model Context Protocol integration for the template step.
It registers tools that can be used by MCP-enabled applications to interact with the template functionality.
"""

import logging
from pathlib import Path
from typing import Any, Dict

# Set up logging
logger = logging.getLogger(__name__)


def register_tools(registry: Any) -> Any:
    """
    Register all template tools with the MCP registry.

    Args:
        registry: The MCP tool registry
    """
    try:
        # Register process_file tool
        registry.register_tool(
            "template.process_file",
            process_file,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to process",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for processed files",
                        "default": "output/template",
                    },
                    "options": {
                        "type": "object",
                        "description": "Processing options",
                        "default": {},
                    },
                },
                "required": ["file_path"],
            },
            "Process a file using the template processor",
            module=__package__,
            category="template",
            examples=[
                {
                    "description": "Process a markdown file",
                    "code": 'template.process_file("input/example.md")',
                }
            ],
        )

        # Register process_directory tool
        registry.register_tool(
            "template.process_directory",
            process_directory,
            {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path to the directory to process",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to process files recursively",
                        "default": False,
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for processed files",
                        "default": "output/template",
                    },
                    "options": {
                        "type": "object",
                        "description": "Processing options",
                        "default": {},
                    },
                },
                "required": ["directory_path"],
            },
            "Process all files in a directory using the template processor",
            module=__package__,
            category="template",
            examples=[
                {
                    "description": "Process all files in a directory recursively",
                    "code": 'template.process_directory("input/gnn_files", recursive=True)',
                }
            ],
        )

        # Register get_template_info tool
        registry.register_tool(
            "template.get_info",
            get_template_info,
            {},
            "Get information about the template step",
            module=__package__,
            category="template",
            examples=[
                {
                    "description": "Get template step information",
                    "code": "template.get_info()",
                }
            ],
        )

        # Register template.pull tool
        registry.register_tool(
            "template.pull",
            pull_template_mcp,
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the maintained template to pull",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Directory to copy the template into, relative to the"
                            " server working directory"
                        ),
                        "default": "input/gnn_files",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "Report the copy plan without writing files"
                            " (copy-safe MCP default)"
                        ),
                        "default": True,
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Replace an existing destination with a different checksum"
                        ),
                        "default": False,
                    },
                },
                "required": ["name"],
            },
            "Pull a maintained template into an output directory",
            module=__package__,
            category="template",
            examples=[
                {
                    "description": "Dry-run pull of the gridworld template",
                    "code": 'template.pull("pomdp-gridworld-3x3")',
                }
            ],
        )

        # Register template.list tool
        registry.register_tool(
            "template.list",
            list_templates_mcp,
            {},
            "List maintained templates with checksums (CLI `gnn templates list` parity)",
            module=__package__,
            category="template",
            examples=[
                {
                    "description": "List maintained templates",
                    "code": "template.list()",
                }
            ],
        )

        # Register template.show tool
        registry.register_tool(
            "template.show",
            show_template_mcp,
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the maintained template to show",
                    },
                },
                "required": ["name"],
            },
            "Show one maintained template record with checksum metadata"
            " (CLI `gnn templates show` parity)",
            module=__package__,
            category="template",
            examples=[
                {
                    "description": "Show the gridworld template record",
                    "code": 'template.show("pomdp-gridworld-3x3")',
                }
            ],
        )

        logger.info("Successfully registered 6 template MCP tools")
        return True

    except Exception as e:
        logger.error(f"Failed to register template MCP tools: {e}")
        return False


def process_file(
    file_path: str,
    output_dir: str = "output/template",
    options: (Dict[str, Any]) | None = None,
) -> Dict[str, Any]:
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
        input_path = Path(file_path)
        output_path = Path(output_dir)

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Set default options if none provided
        if options is None:
            options = {}

        # Import the actual processing function from the template module
        from .processor import process_single_file

        # Process the file
        success = process_single_file(input_path, output_path, options)

        # Generate result
        result: dict[str, Any] = {
            "status": "success" if success else "error",
            "input_file": str(input_path),
            "output_directory": str(output_path),
            "processing_options": options,
        }

        # Add output file paths if successful
        if success:
            file_output_dir = output_path / input_path.stem
            output_file = (
                file_output_dir / f"{input_path.stem}_processed{input_path.suffix}"
            )
            report_file = file_output_dir / f"{input_path.stem}_report.json"

            result["output_file"] = str(output_file)
            result["report_file"] = str(report_file)

        return result

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        return {"status": "error", "error": str(e), "input_file": str(file_path)}


def process_directory(
    directory_path: str,
    recursive: bool = False,
    output_dir: str = "output/template",
    options: (Dict[str, Any]) | None = None,
) -> Dict[str, Any]:
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
        input_directory = Path(directory_path)
        output_path = Path(output_dir)

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Set default options if none provided
        if options is None:
            options = {}

        # Import the template module's main processing function
        from .processor import process_template_standardized

        # Set up a basic logger for this operation
        operation_logger = logging.getLogger("template.mcp.process_directory")

        # Process the directory
        success = process_template_standardized(
            target_dir=input_directory,
            output_dir=output_path,
            logger=operation_logger,
            recursive=recursive,
            verbose=options.get("verbose", False),
            **options,
        )

        # Generate result
        result: dict[str, Any] = {
            "status": "success" if success else "error",
            "input_directory": str(input_directory),
            "output_directory": str(output_path),
            "recursive": recursive,
            "processing_options": options,
        }

        # Add summary file path if it exists
        summary_file = output_path / "template_processing_summary.json"
        if summary_file.exists():
            import json

            with open(summary_file, "r") as f:
                summary = json.load(f)

            result["summary"] = summary

        return result

    except Exception as e:
        logger.error(f"Failed to process directory {directory_path}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "input_directory": str(directory_path),
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
            "Performance tracking",
        ],
        "input_formats": ["any"],
        "output_formats": ["processed files", "JSON reports"],
        "dependencies": [],
    }


def pull_template_mcp(
    name: str,
    output_dir: str = "input/gnn_files",
    dry_run: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Pull a maintained template into ``output_dir``.

    Thin MCP wrapper around :func:`gnn.cli.templates.pull_template`.

    Args:
        name: Name of the maintained template to pull.
        output_dir: Directory to copy the template into, relative to the
            server working directory (CLI parity default
            ``input/gnn_files``).
        dry_run: When True (the copy-safe MCP default) report the copy plan
            without writing any files.
        overwrite: When True, replace an existing destination with a
            different checksum instead of failing.

    Returns:
        Dictionary with ``success`` set to True plus the pull result keys
        (template, source, destination, sha256, dry_run, overwritten,
        copied, message), or ``success`` set to False with an ``error``
        message on failure.
    """

    def _build() -> Dict[str, Any]:
        from gnn.cli.templates import pull_template as _pull_template

        result = _pull_template(
            name, Path(output_dir), dry_run=dry_run, overwrite=overwrite
        )
        return {"success": True, **result}

    try:
        return _build()
    except (KeyError, FileExistsError, FileNotFoundError, OSError) as e:
        logger.error(f"Failed to pull template {name}: {e}")
        return {"success": False, "error": str(e)}


def list_templates_mcp() -> Dict[str, Any]:
    """
    List maintained templates with checksums.

    Thin MCP wrapper around :func:`gnn.cli.templates.list_templates`.

    Returns:
        Dictionary with ``success`` set to True plus the template records
        (``templates``) and their ``total`` count, or ``success`` set to
        False with an ``error`` message on failure.
    """

    def _build() -> Dict[str, Any]:
        from gnn.cli.templates import list_templates as _list_templates

        records = _list_templates()
        return {"success": True, "templates": records, "total": len(records)}

    try:
        return _build()
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return {"success": False, "error": str(e)}


def show_template_mcp(name: str) -> Dict[str, Any]:
    """
    Show one maintained template record with checksum metadata.

    Thin MCP wrapper around :func:`gnn.cli.templates.show_template`.

    Args:
        name: Name of the maintained template to show. The CLI parity
            error message lists the available templates when unknown.

    Returns:
        Dictionary with ``success`` set to True plus the ``template``
        record, or ``success`` set to False with an ``error`` message
        on failure.
    """

    def _build() -> Dict[str, Any]:
        from gnn.cli.templates import show_template as _show_template

        return {"success": True, "template": _show_template(name)}

    try:
        return _build()
    except KeyError as e:
        logger.error(f"Failed to show template {name}: {e}")
        return {"success": False, "error": str(e)}
