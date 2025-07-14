#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 14: Site Generation

This script generates a static site from pipeline outputs:
- Consolidates reports, visualizations, and analysis results
- Creates an HTML documentation site
- Provides navigation and cross-references

Usage:
    python 14_site.py [options]
    (Typically called by main.py)
"""

# --- Imports and Setup ---
import sys
from pathlib import Path
import datetime
import json
import shutil
from typing import Dict, Any, List, Optional
import logging

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    PipelineLogger,
    UTILS_AVAILABLE
)

logger = setup_step_logging("12_site", verbose=False)

# --- Advanced Generator Integration ---
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import importlib.util
    site_generator_path = current_script_path.parent / "site" / "generator.py"
    spec = importlib.util.spec_from_file_location("site_generator", site_generator_path)
    site_generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site_generator_module)
    generate_html_report = site_generator_module.generate_html_report
    SITE_GENERATOR_AVAILABLE = True
except Exception as e:
    log_step_warning(logger, f"Site generator module not available: {e}")
    generate_html_report = None
    SITE_GENERATOR_AVAILABLE = False

DEFAULT_SITE_OUTPUT_DIR = "site_step"

# --- Artifact Collection ---
def collect_pipeline_artifacts(output_dir: Path) -> Dict[str, Any]:
    """
    Collects artifacts from all pipeline steps for site generation.
    Returns:
        Dictionary containing paths to artifacts organized by step
    """
    artifacts = {
        "pipeline_summary": output_dir / "pipeline_execution_summary.json",
        "gnn_discovery": output_dir / "gnn_processing_step" / "1_gnn_discovery_report.md",
        "test_reports": list((output_dir / "test_reports").glob("*.xml")) if (output_dir / "test_reports").exists() else [],
        "type_check": list((output_dir / "type_check").glob("*.md")) if (output_dir / "type_check").exists() else [],
        "gnn_exports": list((output_dir / "gnn_exports").rglob("*")) if (output_dir / "gnn_exports").exists() else [],
        "visualizations": [img for d in (output_dir / "visualization").iterdir() if d.is_dir() for img in d.glob("*.png")] if (output_dir / "visualization").exists() else [],
        "mcp_reports": list((output_dir / "mcp_processing_step").glob("*.md")) if (output_dir / "mcp_processing_step").exists() else [],
        "ontology_reports": list((output_dir / "ontology_processing").glob("*.md")) if (output_dir / "ontology_processing").exists() else [],
        "rendered_simulators": list((output_dir / "gnn_rendered_simulators").rglob("*")) if (output_dir / "gnn_rendered_simulators").exists() else [],
        "llm_outputs": list((output_dir / "llm_processing_step").rglob("*")) if (output_dir / "llm_processing_step").exists() else [],
        "logs": list((output_dir / "logs").glob("*.log")) if (output_dir / "logs").exists() else [],
        "other": []
    }
    # Find other top-level files/dirs not handled above
    handled = set([
        "pipeline_execution_summary.json", "gnn_processing_step", "test_reports", "type_check", "gnn_exports", "visualization", "mcp_processing_step", "ontology_processing", "gnn_rendered_simulators", "llm_processing_step", "logs", "site_step"
    ])
    for item in output_dir.iterdir():
        if item.name not in handled:
            artifacts["other"].append(item)
    logger.info(f"Collected artifacts: {sum(len(v) if isinstance(v, list) else 1 for v in artifacts.values())} total files/dirs")
    return artifacts

# --- Fallback HTML Index Generation ---
def render_markdown(md_path: Path) -> str:
    if not md_path.exists():
        return f"<p>Markdown file not found: {md_path.as_posix()}</p>"
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if MARKDOWN_AVAILABLE:
            return markdown.markdown(content, extensions=['fenced_code', 'tables', 'sane_lists'])
        else:
            return f"<pre>{content}</pre>"
    except Exception as e:
        return f"<p>Error rendering markdown: {e}</p>"

def preview_text_file(txt_path: Path, max_lines: int = 40) -> str:
    if not txt_path.exists():
        return f"<p>Text file not found: {txt_path.as_posix()}</p>"
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        preview = ''.join(lines[:max_lines])
        if len(lines) > max_lines:
            preview += f"\n... (truncated, {len(lines)} total lines)"
        return f"<pre>{preview}</pre>"
    except Exception as e:
        return f"<p>Error previewing text file: {e}</p>"

def preview_json_file(json_path: Path, max_chars: int = 2000) -> str:
    if not json_path.exists():
        return f"<p>JSON file not found: {json_path.as_posix()}</p>"
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pretty = json.dumps(data, indent=2)
        if len(pretty) > max_chars:
            pretty = pretty[:max_chars] + "\n... (truncated)"
        return f"<pre>{pretty}</pre>"
    except Exception as e:
        return f"<p>Error previewing JSON file: {e}</p>"

def embed_image(img_path: Path, max_height: int = 200) -> str:
    if not img_path.exists():
        return f"<p>Image not found: {img_path.as_posix()}</p>"
    rel_path = img_path.name
    return f'<img src="{rel_path}" alt="{img_path.name}" style="max-height:{max_height}px; margin:5px;" loading="lazy"><br><small>{img_path.name}</small>'

def create_fallback_index(artifacts: Dict[str, Any], site_output_dir: Path, output_dir: Path) -> bool:
    """
    Creates a robust HTML index with navigation, links, and previews for all artifacts.
    """
    try:
        site_output_dir.mkdir(parents=True, exist_ok=True)
        index_file = site_output_dir / "index.html"
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>GNN Pipeline Results</title>",
            "<style>body{font-family:sans-serif;margin:40px;}nav{margin-bottom:30px;}nav ul{list-style:none;padding:0;}nav ul li{display:inline;margin-right:15px;}section{margin-bottom:40px;}h1{color:#333;}h2{color:#666;}pre{background:#f4f4f4;padding:10px;border-radius:4px;overflow-x:auto;}img{max-height:200px;margin:5px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px;}</style>",
            "</head>",
            "<body>",
            f"<h1>GNN Pipeline Results</h1>",
            f"<p><em>Generated: {now}</em></p>",
            "<nav><ul>"
        ]
        # TOC
        toc_sections = [k for k in artifacts.keys() if artifacts[k]]
        for section in toc_sections:
            html.append(f"<li><a href='#{section}'>{section.replace('_',' ').title()}</a></li>")
        html.append("</ul></nav>")
        # Summary
        html.append("<section id='summary'><h2>Summary</h2>")
        html.append(f"<p>Total artifact categories: {len([k for k in artifacts if artifacts[k]])}</p>")
        if artifacts.get('pipeline_summary') and Path(artifacts['pipeline_summary']).exists():
            html.append("<h3>Pipeline Execution Summary</h3>")
            html.append(preview_json_file(artifacts['pipeline_summary']))
        html.append("</section>")
        # Sections
        for section, files in artifacts.items():
            if not files:
                continue
            html.append(f"<section id='{section}'><h2>{section.replace('_',' ').title()}</h2>")
            if isinstance(files, list):
                for fpath in files:
                    if isinstance(fpath, Path):
                        rel = fpath.relative_to(output_dir) if fpath.is_absolute() else fpath
                        if fpath.suffix in ['.png', '.jpg', '.jpeg', '.gif']:
                            html.append(embed_image(fpath))
                        elif fpath.suffix == '.md':
                            html.append(render_markdown(fpath))
                        elif fpath.suffix in ['.txt', '.log']:
                            html.append(preview_text_file(fpath))
                        elif fpath.suffix == '.json':
                            html.append(preview_json_file(fpath))
                        else:
                            html.append(f"<a href='{rel}' target='_blank'>{fpath.name}</a><br>")
                    else:
                        html.append(f"<div>{fpath}</div>")
            elif isinstance(files, Path):
                if files.suffix == '.json':
                    html.append(preview_json_file(files))
                elif files.suffix == '.md':
                    html.append(render_markdown(files))
                else:
                    html.append(f"<a href='{files.name}' target='_blank'>{files.name}</a>")
            html.append("</section>")
        html.append("</body></html>")
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
        logger.info(f"Created robust fallback HTML index at {index_file}")
        return True
    except Exception as e:
        log_step_error(logger, f"Error creating fallback HTML index: {e}")
        return False

# --- Main Orchestration ---
def main(args) -> int:
    """
    Main function to orchestrate site generation.
    """
    if args.verbose:
        PipelineLogger.set_verbosity(True)
    log_step_start(logger, "Starting site generation step")
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        log_step_error(logger, f"Output directory does not exist: {output_dir}")
        return 1
    site_output_dir = output_dir / DEFAULT_SITE_OUTPUT_DIR
    try:
        logger.info("Collecting pipeline artifacts...")
        artifacts = collect_pipeline_artifacts(output_dir)
        if not any(artifacts.values()):
            log_step_warning(logger, "No pipeline artifacts found to generate site from")
            return 0
        logger.info(f"Generating site in {site_output_dir}...")
        if SITE_GENERATOR_AVAILABLE and generate_html_report is not None:
            try:
                generate_html_report(output_dir, site_output_dir / "index.html")
                log_step_success(logger, f"Site generated successfully in {site_output_dir / 'index.html'}")
                return 0
            except Exception as e:
                log_step_error(logger, f"Site generator failed: {e}. Falling back to robust HTML index.")
        # Fallback
        success = create_fallback_index(artifacts, site_output_dir, output_dir)
        if success:
            log_step_success(logger, f"Fallback site generation completed successfully. Output: {site_output_dir / 'index.html'}")
            return 0
        else:
            log_step_error(logger, "Fallback site generation failed")
            return 1
    except Exception as e:
        log_step_error(logger, f"Unexpected error in site generation: {e}")
        return 1

if __name__ == '__main__':
    if UTILS_AVAILABLE:
        try:
            parsed_args = EnhancedArgumentParser.parse_step_arguments("12_site")
        except Exception as e:
            log_step_error(logger, f"Failed to parse arguments with enhanced parser: {e}")
            import argparse
            parser = argparse.ArgumentParser(description="Generate static site from GNN pipeline outputs")
            parser.add_argument("--output-dir", type=Path, required=True, help="Pipeline output directory containing artifacts to include in site")
            parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose logging")
            parsed_args = parser.parse_args()
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Generate static site from GNN pipeline outputs")
        parser.add_argument("--output-dir", type=Path, required=True, help="Pipeline output directory containing artifacts to include in site")
        parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose logging")
        parsed_args = parser.parse_args()
    if parsed_args.verbose:
        PipelineLogger.set_verbosity(True)
    exit_code = main(parsed_args)
    sys.exit(exit_code) 