from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# The generator module is imported for its core function.
# It is assumed that the generator does not import this mcp.py file to avoid circular dependencies.
from .generator import generate_html_report, make_section_id

logger = logging.getLogger(__name__)

def generate_pipeline_summary_website_mcp(
    output_dir: str, 
    website_output_filename: str = "index.html", 
    verbose: bool = False,
    include_metadata: bool = True,
    search_enabled: bool = True,
    validate_output: bool = True,
    max_file_size_mb: int = 50
) -> Dict[str, Any]:
    """
    MCP tool to generate a single HTML website summarizing all contents of the GNN pipeline output directory.
    
    Args:
        output_dir: Path to the pipeline output directory
        website_output_filename: Name of the output HTML file (default: index.html)
        verbose: Enable verbose logging
        include_metadata: Include generation metadata in the website
        search_enabled: Enable search functionality in the website
        validate_output: Validate the generated website after creation
        max_file_size_mb: Maximum file size in MB to embed (default: 50MB)
        
    Returns:
        Dictionary with success status and metadata
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

        # Enhanced input validation
        if not output_dir_path.exists():
            error_msg = f"Output directory does not exist: {output_dir_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "error_type": "FileNotFoundError"}

        if not output_dir_path.is_dir():
            error_msg = f"Output path is not a directory: {output_dir_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "error_type": "NotADirectoryError"}

        # Check if directory has any content
        try:
            items = list(output_dir_path.iterdir())
            if not items:
                error_msg = f"Output directory is empty: {output_dir_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "error_type": "EmptyDirectoryError"}
        except PermissionError as e:
            error_msg = f"Permission denied accessing directory: {output_dir_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "error_type": "PermissionError"}

        # Validate filename
        if not website_output_filename.endswith('.html'):
            logger.warning(f"Website output filename should end with .html: {website_output_filename}")

        # Ensure the parent directory for the website file exists
        try:
            website_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            error_msg = f"Permission denied creating output directory: {website_output_file_path.parent}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "error_type": "PermissionError"}
        
        # Call the core report generation function
        generate_html_report(output_dir_path, website_output_file_path)
        
        # Validate output if requested
        validation_result = None
        if validate_output:
            validation_result = validate_website_output_mcp(str(website_output_file_path))
        
        # Generate comprehensive metadata
        metadata = {
            "generation_time": datetime.now().isoformat(),
            "output_directory": str(output_dir_path),
            "website_file": str(website_output_file_path),
            "file_size": website_output_file_path.stat().st_size if website_output_file_path.exists() else 0,
            "file_size_mb": round(website_output_file_path.stat().st_size / (1024 * 1024), 2) if website_output_file_path.exists() else 0,
            "features": {
                "search_enabled": search_enabled,
                "metadata_included": include_metadata,
                "responsive_design": True,
                "collapsible_sections": True,
                "file_highlighting": True,
                "enhanced_navigation": True,
                "max_file_size_mb": max_file_size_mb
            },
            "validation": validation_result,
            "directory_stats": {
                "total_items": len(items),
                "files": len([item for item in items if item.is_file()]),
                "directories": len([item for item in items if item.is_dir()])
            }
        }
        
        success_msg = f"HTML website generated successfully at {website_output_file_path}"
        logger.info(success_msg)
        
        return {
            "success": True, 
            "message": success_msg, 
            "website_path": str(website_output_file_path),
            "metadata": metadata
        }
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during website generation: {str(e)}"
        logger.exception(error_msg)  # Log the full traceback for debugging
        return {"success": False, "error": error_msg, "error_type": type(e).__name__}

def analyze_pipeline_outputs_mcp(output_dir: str) -> Dict[str, Any]:
    """
    MCP tool to analyze pipeline outputs and provide detailed statistics.
    
    Args:
        output_dir: Path to the pipeline output directory
        
    Returns:
        Dictionary with analysis results
    """
    try:
        output_dir_path = Path(output_dir).resolve()
        
        if not output_dir_path.exists():
            return {"success": False, "error": f"Directory does not exist: {output_dir}", "error_type": "FileNotFoundError"}
        
        if not output_dir_path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {output_dir}", "error_type": "NotADirectoryError"}
        
        analysis = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "step_directories": {},
            "largest_files": [],
            "recent_files": [],
            "pipeline_steps": {},
            "estimated_size_mb": 0,
            "accessibility": {
                "readable_files": 0,
                "unreadable_files": 0,
                "permission_errors": 0
            }
        }
        
        # Analyze directory structure
        for item in output_dir_path.rglob("*"):
            try:
                if item.is_file():
                    analysis["total_files"] += 1
                    suffix = item.suffix.lower()
                    analysis["file_types"][suffix] = analysis["file_types"].get(suffix, 0) + 1
                    
                    # Track largest files
                    try:
                        file_size = item.stat().st_size
                        analysis["estimated_size_mb"] += file_size / (1024 * 1024)
                        analysis["largest_files"].append({
                            "path": str(item.relative_to(output_dir_path)),
                            "size": file_size,
                            "size_mb": round(file_size / (1024 * 1024), 2)
                        })
                        analysis["accessibility"]["readable_files"] += 1
                    except (OSError, PermissionError):
                        analysis["accessibility"]["unreadable_files"] += 1
                        continue
                    
                    # Track recent files
                    try:
                        mtime = item.stat().st_mtime
                        analysis["recent_files"].append({
                            "path": str(item.relative_to(output_dir_path)),
                            "modified": datetime.fromtimestamp(mtime).isoformat(),
                            "mtime": mtime
                        })
                    except (OSError, PermissionError):
                        continue
                        
                elif item.is_dir():
                    analysis["total_directories"] += 1
                    
                    # Check for step directories
                    dir_name = item.name
                    if any(step in dir_name for step in ["step", "processing", "output"]):
                        try:
                            file_count = len(list(item.rglob("*")))
                            analysis["step_directories"][dir_name] = {
                                "path": str(item.relative_to(output_dir_path)),
                                "file_count": file_count,
                                "has_content": file_count > 0
                            }
                        except (OSError, PermissionError):
                            analysis["accessibility"]["permission_errors"] += 1
                            continue
                            
            except (OSError, PermissionError) as e:
                analysis["accessibility"]["permission_errors"] += 1
                continue
        
        # Sort and limit results
        analysis["largest_files"].sort(key=lambda x: x["size"], reverse=True)
        analysis["largest_files"] = analysis["largest_files"][:10]
        
        analysis["recent_files"].sort(key=lambda x: x["mtime"], reverse=True)
        analysis["recent_files"] = analysis["recent_files"][:10]
        
        # Analyze pipeline steps
        step_patterns = {
            "gnn_processing_step": "GNN File Processing",
            "test_reports": "Test Execution",
            "type_check": "Type Checking",
            "gnn_exports": "Export Generation",
            "visualization": "Visualization",
            "mcp_processing_step": "MCP Processing",
            "ontology_processing": "Ontology Processing",
            "gnn_rendered_simulators": "Simulator Rendering",
            "execution_results": "Execution Results",
            "llm_processing_step": "LLM Processing",
            "sapf_generation": "SAPF Generation",
            "website": "Website Generation"
        }
        
        for step_dir, step_name in step_patterns.items():
            step_path = output_dir_path / step_dir
            if step_path.exists():
                try:
                    file_count = len(list(step_path.rglob("*")))
                    analysis["pipeline_steps"][step_name] = {
                        "directory": step_dir,
                        "file_count": file_count,
                        "has_content": file_count > 0,
                        "status": "completed" if file_count > 0 else "empty"
                    }
                except (OSError, PermissionError):
                    analysis["pipeline_steps"][step_name] = {
                        "directory": step_dir,
                        "file_count": 0,
                        "has_content": False,
                        "status": "access_denied"
                    }
        
        return {
            "success": True,
            "analysis": analysis,
            "summary": {
                "total_items": analysis["total_files"] + analysis["total_directories"],
                "total_size_mb": round(analysis["estimated_size_mb"], 2),
                "pipeline_steps_completed": len([step for step in analysis["pipeline_steps"].values() if step["status"] == "completed"]),
                "accessibility_score": round((analysis["accessibility"]["readable_files"] / max(analysis["total_files"], 1)) * 100, 2)
            }
        }
        
    except Exception as e:
        return {"success": False, "error": f"Analysis failed: {str(e)}", "error_type": type(e).__name__}

def validate_website_output_mcp(website_path: str) -> Dict[str, Any]:
    """
    MCP tool to validate generated website for quality and accessibility.
    
    Args:
        website_path: Path to the generated website HTML file
        
    Returns:
        Dictionary with validation results
    """
    try:
        website_file_path = Path(website_path).resolve()
        
        if not website_file_path.exists():
            return {"success": False, "error": f"Website file does not exist: {website_path}", "error_type": "FileNotFoundError"}
        
        if not website_file_path.is_file():
            return {"success": False, "error": f"Path is not a file: {website_path}", "error_type": "NotAFileError"}
        
        validation_result = {
            "file_exists": True,
            "file_size": website_file_path.stat().st_size,
            "file_size_mb": round(website_file_path.stat().st_size / (1024 * 1024), 2),
            "file_readable": True,
            "html_validation": {
                "has_doctype": False,
                "has_html_tag": False,
                "has_head_tag": False,
                "has_body_tag": False,
                "has_title": False,
                "has_meta_charset": False,
                "has_viewport_meta": False
            },
            "content_validation": {
                "has_css": False,
                "has_javascript": False,
                "has_search_functionality": False,
                "has_navigation": False,
                "has_metadata": False,
                "has_table_of_contents": False
            },
            "accessibility": {
                "has_lang_attribute": False,
                "has_alt_text_for_images": False,
                "has_semantic_structure": False,
                "has_skip_links": False
            },
            "performance": {
                "file_size_ok": True,
                "inline_css": True,
                "inline_js": True
            },
            "errors": [],
            "warnings": []
        }
        
        # Read and analyze the HTML content
        try:
            with open(website_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            validation_result["file_readable"] = False
            validation_result["errors"].append("File contains invalid UTF-8 characters")
            return {"success": True, "validation": validation_result}
        except Exception as e:
            validation_result["file_readable"] = False
            validation_result["errors"].append(f"Error reading file: {str(e)}")
            return {"success": True, "validation": validation_result}
        
        # Basic HTML structure validation
        validation_result["html_validation"]["has_doctype"] = "<!DOCTYPE html>" in html_content
        validation_result["html_validation"]["has_html_tag"] = "<html" in html_content
        validation_result["html_validation"]["has_head_tag"] = "<head" in html_content
        validation_result["html_validation"]["has_body_tag"] = "<body" in html_content
        validation_result["html_validation"]["has_title"] = "<title>" in html_content
        validation_result["html_validation"]["has_meta_charset"] = 'charset="UTF-8"' in html_content
        validation_result["html_validation"]["has_viewport_meta"] = 'viewport' in html_content
        
        # Content validation
        validation_result["content_validation"]["has_css"] = "<style>" in html_content
        validation_result["content_validation"]["has_javascript"] = "<script>" in html_content
        validation_result["content_validation"]["has_search_functionality"] = "searchBox" in html_content
        validation_result["content_validation"]["has_navigation"] = "navbar" in html_content
        validation_result["content_validation"]["has_metadata"] = "metadata" in html_content
        validation_result["content_validation"]["has_table_of_contents"] = "toc" in html_content
        
        # Accessibility validation
        validation_result["accessibility"]["has_lang_attribute"] = 'lang="en"' in html_content
        validation_result["accessibility"]["has_alt_text_for_images"] = "alt=" in html_content
        validation_result["accessibility"]["has_semantic_structure"] = any(tag in html_content for tag in ["<header>", "<nav>", "<main>", "<section>", "<footer>"])
        
        # Performance validation
        if validation_result["file_size_mb"] > 10:
            validation_result["performance"]["file_size_ok"] = False
            validation_result["warnings"].append("File size is large (>10MB), may affect loading performance")
        
        # Calculate overall score
        html_score = sum(validation_result["html_validation"].values()) / len(validation_result["html_validation"])
        content_score = sum(validation_result["content_validation"].values()) / len(validation_result["content_validation"])
        accessibility_score = sum(validation_result["accessibility"].values()) / len(validation_result["accessibility"])
        
        validation_result["overall_score"] = round((html_score + content_score + accessibility_score) / 3 * 100, 2)
        validation_result["status"] = "excellent" if validation_result["overall_score"] >= 90 else "good" if validation_result["overall_score"] >= 70 else "needs_improvement"
        
        return {"success": True, "validation": validation_result}
        
    except Exception as e:
        return {"success": False, "error": f"Validation failed: {str(e)}", "error_type": type(e).__name__}

def get_website_statistics_mcp(website_path: str) -> Dict[str, Any]:
    """
    MCP tool to get detailed statistics about the generated website.
    
    Args:
        website_path: Path to the generated website HTML file
        
    Returns:
        Dictionary with website statistics
    """
    try:
        website_file = Path(website_path).resolve()
        
        if not website_file.exists():
            return {"success": False, "error": f"Website file does not exist: {website_path}"}
        
        with open(website_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse content for statistics
        stats = {
            "file_info": {
                "size_bytes": website_file.stat().st_size,
                "size_mb": website_file.stat().st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(website_file.stat().st_mtime).isoformat(),
                "line_count": len(content.split('\n'))
            },
            "content_breakdown": {
                "total_characters": len(content),
                "html_tags": len([tag for tag in content.split('<') if '>' in tag]),
                "sections": content.count('class="section"'),
                "images": content.count('<img'),
                "links": content.count('<a href'),
                "tables": content.count('<table'),
                "code_blocks": content.count('<pre'),
                "collapsible_sections": content.count('class="collapsible"')
            },
            "features_detected": {
                "search_functionality": 'id="searchBox"' in content,
                "navigation_menu": 'id="navbar"' in content,
                "table_of_contents": 'id="toc-container"' in content,
                "responsive_design": 'media (max-width' in content,
                "javascript_interactivity": '<script>' in content,
                "css_styling": '<style>' in content
            },
            "embedded_content": {
                "base64_images": content.count('data:image'),
                "json_data": content.count('"success":') + content.count('"error":'),
                "markdown_content": content.count('class="markdown-content"'),
                "text_files": content.count('class="text-file-content"'),
                "html_embeds": content.count('class="html-file-content"')
            }
        }
        
        return {
            "success": True,
            "statistics": stats,
            "website_path": str(website_file)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def register_tools(mcp_instance):
    """
    Register all website generation MCP tools with the MCP instance.
    
    Args:
        mcp_instance: The MCP instance to register tools with
    """
    try:
        # Register the main website generation tool
        mcp_instance.register_tool(
            name="generate_pipeline_summary_website",
            description="Generate a comprehensive HTML website summarizing GNN pipeline outputs with enhanced features",
            parameters={
                "output_dir": {
                    "type": "string",
                    "description": "Path to the pipeline output directory",
                    "required": True
                },
                "website_output_filename": {
                    "type": "string", 
                    "description": "Name of the output HTML file (default: index.html)",
                    "default": "index.html"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging",
                    "default": False
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include generation metadata in the website",
                    "default": True
                },
                "search_enabled": {
                    "type": "boolean",
                    "description": "Enable search functionality in the website",
                    "default": True
                },
                "validate_output": {
                    "type": "boolean",
                    "description": "Validate the generated website after creation",
                    "default": True
                },
                "max_file_size_mb": {
                    "type": "integer",
                    "description": "Maximum file size in MB to embed (default: 50MB)",
                    "default": 50
                }
            },
            function=generate_pipeline_summary_website_mcp
        )
        
        # Register the pipeline analysis tool
        mcp_instance.register_tool(
            name="analyze_pipeline_outputs",
            description="Analyze pipeline outputs and provide detailed statistics and accessibility information",
            parameters={
                "output_dir": {
                    "type": "string",
                    "description": "Path to the pipeline output directory",
                    "required": True
                }
            },
            function=analyze_pipeline_outputs_mcp
        )
        
        # Register the website validation tool
        mcp_instance.register_tool(
            name="validate_website_output",
            description="Validate generated website for quality, accessibility, and performance",
            parameters={
                "website_path": {
                    "type": "string",
                    "description": "Path to the generated website HTML file",
                    "required": True
                }
            },
            function=validate_website_output_mcp
        )
        
        # Register the website statistics tool
        mcp_instance.register_tool(
            name="get_website_statistics",
            description="Get detailed statistics about the generated website including file size and content analysis",
            parameters={
                "website_path": {
                    "type": "string",
                    "description": "Path to the generated website HTML file",
                    "required": True
                }
            },
            function=get_website_statistics_mcp
        )
        
        logger.info("✅ All website generation MCP tools registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to register MCP tools: {e}")
        raise
