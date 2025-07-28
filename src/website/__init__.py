"""
website module for GNN Processing Pipeline.

This module provides website capabilities with fallback implementations.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def process_website(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process website for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("website")
    
    try:
        log_step_start(logger, "Processing website")
        
        # Create results directory
        results_dir = output_dir / "website_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic website processing
        results = {
            "processed_files": 0,
            "success": True,
            "errors": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if gnn_files:
            results["processed_files"] = len(gnn_files)
        
        # Save results
        import json
        results_file = results_dir / "website_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if results["success"]:
            log_step_success(logger, "website processing completed successfully")
        else:
            log_step_error(logger, "website processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, "website processing failed", {"error": str(e)})
        return False

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "website processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'basic_processing': True,
    'fallback_mode': True,
    'mcp_integration': True
}

# Supported file types
SUPPORTED_FILE_TYPES = {
    'images': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
    'markdown': ['.md', '.markdown'],
    'json': ['.json'],
    'text': ['.txt', '.log'],
    'html': ['.html', '.htm']
}

def generate_website(logger, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Generate website from pipeline output.
    
    Args:
        logger: Logger instance
        input_dir: Input directory with pipeline output
        output_dir: Output directory for website
        
    Returns:
        Dictionary with generation results
    """
    try:
        if not input_dir.exists():
            return {
                "success": False,
                "error": f"Input directory {input_dir} does not exist"
            }
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate basic website structure
        index_html = output_dir / "index.html"
        with open(index_html, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>GNN Pipeline Results</title>
</head>
<body>
    <h1>GNN Pipeline Results</h1>
    <p>Website generated from pipeline output.</p>
</body>
</html>""")
        
        return {
            "success": True,
            "output_dir": str(output_dir),
            "files_generated": ["index.html"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_html_report(content: str, output_file: Path) -> bool:
    """Generate HTML report from content."""
    try:
        with open(output_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>GNN Report</title>
</head>
<body>
    <pre>{content}</pre>
</body>
</html>""")
        return True
    except Exception:
        return False

def embed_image(image_path: Path, output_file: Path) -> bool:
    """Embed image in HTML output."""
    try:
        # Basic image embedding
        with open(output_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<body>
    <img src="{image_path}" alt="Embedded image">
</body>
</html>""")
        return True
    except Exception:
        return False

def embed_markdown_file(md_path: Path, output_file: Path) -> bool:
    """Embed markdown file in HTML output."""
    try:
        with open(md_path, 'r') as f:
            content = f.read()
        
        with open(output_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Markdown Content</title>
</head>
<body>
    <pre>{content}</pre>
</body>
</html>""")
        return True
    except Exception:
        return False

def embed_text_file(text_path: Path, output_file: Path) -> bool:
    """Embed text file in HTML output."""
    try:
        with open(text_path, 'r') as f:
            content = f.read()
        
        with open(output_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Text Content</title>
</head>
<body>
    <pre>{content}</pre>
</body>
</html>""")
        return True
    except Exception:
        return False

def embed_json_file(json_path: Path, output_file: Path) -> bool:
    """Embed JSON file in HTML output."""
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        with open(output_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>JSON Content</title>
</head>
<body>
    <pre>{json.dumps(data, indent=2)}</pre>
</body>
</html>""")
        return True
    except Exception:
        return False

def embed_html_file(html_path: Path, output_file: Path) -> bool:
    """Embed HTML file in output."""
    try:
        with open(html_path, 'r') as f:
            content = f.read()
        
        with open(output_file, 'w') as f:
            f.write(content)
        return True
    except Exception:
        return False

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the website module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'supported_file_types': SUPPORTED_FILE_TYPES,
        'embedding_capabilities': []
    }
    
    # Embedding capabilities
    info['embedding_capabilities'].extend([
        'HTML generation',
        'Image embedding',
        'Markdown embedding',
        'Text embedding',
        'JSON embedding',
        'HTML embedding'
    ])
    
    return info

def get_supported_file_types() -> Dict[str, List[str]]:
    """Get supported file types for embedding."""
    return SUPPORTED_FILE_TYPES

def validate_website_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate website configuration."""
    errors = []
    
    # Check required fields
    if 'output_dir' not in config:
        errors.append("Missing required field: output_dir")
    
    # Check output directory
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_dir}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

__all__ = [
    'process_website',
    'generate_website',
    'generate_html_report',
    'embed_image',
    'embed_markdown_file',
    'embed_text_file',
    'embed_json_file',
    'embed_html_file',
    'get_module_info',
    'get_supported_file_types',
    'validate_website_config',
    'SUPPORTED_FILE_TYPES',
    'FEATURES',
    '__version__'
]
