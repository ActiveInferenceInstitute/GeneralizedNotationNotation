import argparse
import base64
import json
import logging
import os
from pathlib import Path
import shutil
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
import sys # Add sys import for exit codes in main_site_generator

import markdown # For rendering markdown content

logger = logging.getLogger(__name__)

# --- HTML Templates ---
HTML_START_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Output Summary</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        header {{ background-color: #333; color: #fff; padding: 1em 0; text-align: center; }}
        nav ul {{ list-style-type: none; padding: 0; text-align: center; background-color: #444; margin-bottom: 20px; }}
        nav ul li {{ display: inline; margin-right: 15px; }}
        nav ul li a {{ color: #fff; text-decoration: none; padding: 10px 15px; display: inline-block; }}
        nav ul li a:hover {{ background-color: #555; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        pre {{ background-color: #eee; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        img, iframe {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
        .file-link {{ display: inline-block; margin: 5px; padding: 8px 12px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
        .file-link:hover {{ background-color: #0056b3; }}
        .section {{ margin-bottom: 30px; }}
        .gallery img {{ margin: 5px; max-height: 200px; width: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
        .log-output {{ white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }}
        .collapsible {{ background-color: #f9f9f9; color: #444; cursor: pointer; padding: 12px; width: 100%; border: none; text-align: left; outline: none; font-size: 1.1em; margin-top: 10px; border-bottom: 1px solid #ddd; }}
        .active, .collapsible:hover {{ background-color: #efefef; }}
        .collapsible-content {{ padding: 0 18px; display: none; overflow: hidden; background-color: white; border: 1px solid #ddd; border-top: none; }}
        .toc {{ border: 1px solid #ddd; background-color: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc ul li a {{ text-decoration: none; color: #007bff; }}
        .toc ul li a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <header>
        <h1>Generalized Notation Notation (GNN) Pipeline Output Summary</h1>
    </header>
    <nav id="navbar">
        <ul>
            <!-- Nav links will be injected here -->
        </ul>
    </nav>
    <div class="container">
        <div id="toc-container" class="toc">
            <h2>Table of Contents</h2>
            <ul id="toc-list">
                <!-- TOC items will be injected here -->
            </ul>
        </div>
"""

HTML_END_TEMPLATE = """
    </div>
    <script>
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }

        // Smooth scroll for TOC links
        document.querySelectorAll('#toc-list a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
        
        // Navbar generation from h2 tags
        const sections = document.querySelectorAll('.container > h2');
        const navUl = document.querySelector('#navbar ul');
        const tocUl = document.getElementById('toc-list');

        sections.forEach(section => {
            const title = section.textContent;
            const id = section.id;
            if (id) {
                // Navbar link
                const navLi = document.createElement('li');
                const navA = document.createElement('a');
                navA.textContent = title;
                navA.href = `#${id}`;
                navLi.appendChild(navA);
                navUl.appendChild(navLi);

                // TOC link
                const tocLi = document.createElement('li');
                const tocA = document.createElement('a');
                tocA.textContent = title;
                tocA.href = `#${id}`;
                tocLi.appendChild(tocA);
                tocUl.appendChild(tocLi);
            }
        });

    </script>
</body>
</html>
"""

def make_section_id(title: str) -> str:
    """Generates a URL-friendly ID from a title."""
    return title.lower().replace(" ", "-").replace("/", "-").replace(":", "").replace("(", "").replace(")", "")

def add_collapsible_section(f: IO[str], title: str, content_html: str, is_open: bool = False):
    """Adds a collapsible section to the HTML file."""
    section_id = make_section_id(title) # Although collapsible doesn't use ID for navigation, it's good practice
    display_style = "block" if is_open else "none"
    f.write(f'<button type="button" class="collapsible">{title}</button>\n')
    f.write(f'<div class="collapsible-content" style="display: {display_style};">\n')
    f.write(content_html)
    f.write("</div>\n")

def embed_image(file_path: Path, alt_text: Optional[str] = None) -> str:
    """Embeds an image into HTML using base64 encoding."""
    if not file_path.exists():
        return f"<p>Image not found: {file_path.as_posix()}</p>"
    try:
        with open(file_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        alt = alt_text if alt_text else file_path.name
        return f'<img src="data:image/{file_path.suffix.lstrip(".")};base64,{encoded_string}" alt="{alt}" loading="lazy"><br><small>{file_path.name}</small>'
    except Exception as e:
        logger.error(f"Error embedding image {file_path.as_posix()}: {e}")
        return f"<p>Error embedding image {file_path.name}: {e}</p>"

def embed_markdown_file(file_path: Path) -> str:
    """Reads a markdown file and converts it to HTML."""
    if not file_path.exists():
        return f"<p>Markdown file not found: {file_path.as_posix()}</p>"
    try:
        with open(file_path, 'r', encoding='utf-8') as md_file:
            content = md_file.read()
        return markdown.markdown(content, extensions=['fenced_code', 'tables', 'sane_lists'])
    except Exception as e:
        logger.error(f"Error reading or converting markdown file {file_path.as_posix()}: {e}")
        return f"<p>Error processing markdown file {file_path.name}: {e}</p>"

def embed_text_file(file_path: Path, max_lines: Optional[int] = 100) -> str:
    """Reads a text file and embeds its content in a <pre> tag."""
    if not file_path.exists():
        return f"<p>Text file not found: {file_path.as_posix()}</p>"
    try:
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            lines = txt_file.readlines()
        content = "".join(lines[:max_lines])
        if len(lines) > max_lines:
            content += f"\n... (file truncated, total lines: {len(lines)})"
        return f"<pre class='log-output'><code>{content}</code></pre><small>{file_path.name}</small>"
    except Exception as e:
        logger.error(f"Error reading text file {file_path.as_posix()}: {e}")
        return f"<p>Error processing text file {file_path.name}: {e}</p>"

def embed_json_file(file_path: Path) -> str:
    """Reads a JSON file and pretty-prints it in a <pre> tag."""
    if not file_path.exists():
        return f"<p>JSON file not found: {file_path.as_posix()}</p>"
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return f"<pre class='log-output'><code>{json.dumps(data, indent=2)}</code></pre><small>{file_path.name}</small>"
    except Exception as e:
        logger.error(f"Error reading or parsing JSON file {file_path.as_posix()}: {e}")
        return f"<p>Error processing JSON file {file_path.name}: {e}</p>"
        
def embed_html_file(file_path: Path) -> str:
    """Embeds an HTML file content directly or within an iframe if too complex."""
    if not file_path.exists():
        return f"<p>HTML file not found: {file_path.as_posix()}</p>"
    try:
        # For simplicity, using iframe. Direct embedding might break styles.
        # A more sophisticated approach would be to copy assets and adjust paths,
        # or to parse and sanitize the HTML.
        # Copy the HTML file and its associated directory (if one exists, e.g., for Plotly)
        # to a subdirectory in the site output to ensure relative paths work.
        
        # Simplified: provide a link to open it, or try iframe for basic ones.
        # This assumes the output HTML is in the same directory as the main report.
        # For robust iframe, files need to be served or be in a predictable relative path.
        # For now, let's just link to it, assuming it's a sibling or in a known relative path.
        # Or, if we copy it to an 'assets' folder within the site output:
        # Create a unique name for the asset
        
        # For now, let's assume direct embedding of simple HTML reports (e.g. from type checker)
        # If a file is complex (e.g. has its own JS, complex CSS), an iframe pointing to a copied version is better.
        # For this initial version, we'll link it.
        
        # Let's try to read and embed directly if it's simple.
        # For complex HTML like plotly, it's better to copy to an assets dir and iframe.
        # This will be improved if the user wants interactive plots directly embedded.
        with open(file_path, 'r', encoding='utf-8') as html_f:
            content = html_f.read()
        # Basic check for complexity (very naive)
        if "<script" in content.lower() or "<link rel=\"stylesheet\"" in content.lower():
            # For complex HTML, provide a link. Or, copy to an assets folder and iframe.
            # For this iteration, providing a link to the file (relative to the main output folder)
            # This requires the user to navigate the output folder structure.
            # A better way: copy to a "site_assets" folder and link relatively.
            return (f'<p><a href="{file_path.name}" target="_blank" class="file-link">View HTML Report: {file_path.name}</a> (Opens in new tab)</p>' +
                    f'<p><em>Embedding complex HTML directly can be problematic. This report is linked.</em></p>' +
                    f'<iframe src="{file_path.name}" width="100%" height="500px" style="border:1px solid #ccc;"></iframe>' +
                    f'<small>Attempting to iframe: {file_path.name}. If it does not load correctly, please use the link above.</small>')
        else: # Simple HTML
            return content

    except Exception as e:
        logger.error(f"Error processing HTML file {file_path.as_posix()}: {e}")
        return f"<p>Error processing HTML file {file_path.name}: {e}</p>"

def process_directory_generic(f: IO[str], dir_path: Path, base_output_dir: Path, title_prefix: str = ""):
    """Generic handler for directories: lists images, MDs, TXTs, JSONs, HTMLs."""
    if not dir_path.is_dir():
        return

    content_html = ""
    images = []
    md_files = []
    txt_files = []
    json_files = []
    html_files = []
    other_files = []

    for item in sorted(dir_path.iterdir()):
        if item.is_file():
            if item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                images.append(item)
            elif item.suffix.lower() == '.md':
                md_files.append(item)
            elif item.suffix.lower() in ['.txt', '.log']:
                txt_files.append(item)
            elif item.suffix.lower() == '.json':
                json_files.append(item)
            elif item.suffix.lower() in ['.html', '.htm']:
                html_files.append(item)
            else:
                other_files.append(item)
    
    if images:
        content_html += "<h3>Images</h3><div class='gallery'>"
        for img_path in images:
            content_html += embed_image(img_path)
        content_html += "</div>"

    if md_files:
        content_html += "<h3>Markdown Reports</h3>"
        for md_path in md_files:
            content_html += f"<h4>{md_path.name}</h4>"
            content_html += embed_markdown_file(md_path)
            
    if html_files:
        content_html += "<h3>HTML Reports/Outputs</h3>"
        for html_path in html_files:
            relative_html_path = html_path.relative_to(base_output_dir).as_posix()
            content_html += f"<h4>{html_path.name}</h4>"
            content_html += f'<p><a href="{relative_html_path}" target="_blank" class="file-link">View standalone: {html_path.name}</a></p>'
            # Sandbox iframe for security, allow-scripts and allow-same-origin for functionality if needed by the embedded HTML.
            content_html += f'<iframe src="{relative_html_path}" width="100%" height="600px" style="border:1px solid #ccc;" sandbox="allow-scripts allow-same-origin allow-popups allow-forms"></iframe>'

    if json_files:
        content_html += "<h3>JSON Files</h3>"
        for json_path in json_files:
            content_html += f"<h4>{json_path.name}</h4>"
            content_html += embed_json_file(json_path)

    if txt_files:
        content_html += "<h3>Text/Log Files</h3>"
        for txt_path in txt_files:
            content_html += f"<h4>{txt_path.name}</h4>"
            content_html += embed_text_file(txt_path)
            
    if other_files:
        content_html += "<h3>Other Files</h3><ul>"
        for other_path in other_files:
            relative_other_path = other_path.relative_to(base_output_dir).as_posix()
            content_html += f'<li><a href="{relative_other_path}" class="file-link" target="_blank">{other_path.name}</a></li>'
        content_html += "</ul>"

    if content_html:
        section_title = f"{title_prefix}{dir_path.name}"
        f.write(f"<div class='section' id='{make_section_id(section_title)}'>\n")
        f.write(f"<h2>{section_title}</h2>\n")
        f.write(content_html)
        f.write("</div>\n")

# --- Section specific helpers ---

def _add_pipeline_summary_section(f: IO[str], output_dir: Path):
    summary_json_path = output_dir / "pipeline_execution_summary.json"
    if summary_json_path.exists():
        f.write(f"<div class='section' id='{make_section_id('Pipeline Execution Summary')}'>\n")
        f.write(f"<h2>Pipeline Execution Summary</h2>\n")
        f.write(embed_json_file(summary_json_path))
        f.write("</div>\n")
    else:
        logger.warning(f"Pipeline summary JSON not found: {summary_json_path.as_posix()}")

def _add_gnn_discovery_section(f: IO[str], output_dir: Path):
    gnn_discovery_dir = output_dir / "gnn_processing_step"
    gnn_discovery_report = gnn_discovery_dir / "1_gnn_discovery_report.md"
    if gnn_discovery_report.exists():
        f.write(f"<div class='section' id='{make_section_id('GNN Discovery')}'>\n")
        f.write(f"<h2>GNN Discovery (Step 1)</h2>\n")
        f.write(embed_markdown_file(gnn_discovery_report))
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Discovery report not found: {gnn_discovery_report.as_posix()}")

def _add_test_reports_section(f: IO[str], output_dir: Path):
    test_reports_dir = output_dir / "test_reports"
    pytest_report_xml = test_reports_dir / "pytest_report.xml"
    if pytest_report_xml.exists():
        f.write(f"<div class='section' id='{make_section_id('Test Reports')}'>\n")
        f.write(f"<h2>Test Reports (Step 3)</h2>\n")
        content_html = embed_text_file(pytest_report_xml, max_lines=200)
        add_collapsible_section(f, "Pytest Report (pytest_report.xml - partial)", content_html)
        f.write("</div>\n")
    else:
        logger.warning(f"Pytest XML report not found: {pytest_report_xml.as_posix()}")

def _add_type_checker_section(f: IO[str], output_dir: Path):
    type_check_dir = output_dir / "type_check"
    if type_check_dir.is_dir():
        f.write(f"<div class='section' id='{make_section_id('GNN Type Checker')}'>\n")
        f.write(f"<h2>GNN Type Checker (Step 4)</h2>\n")
        
        type_check_report_md = type_check_dir / "type_check_report.md"
        if type_check_report_md.exists():
            f.write("<h3>Type Check Report</h3>")
            f.write(embed_markdown_file(type_check_report_md))

        resource_data_json = type_check_dir / "resources" / "type_check_data.json"
        if resource_data_json.exists():
                add_collapsible_section(f, "Type Check Data (JSON)", embed_json_file(resource_data_json))
        
        html_vis_dir = type_check_dir / "resources" / "html_vis"
        if html_vis_dir.is_dir():
            process_directory_generic(f, html_vis_dir, output_dir, title_prefix="Type Checker HTML Visualizations: ")

        resource_estimates_dir = type_check_dir / "resource_estimates"
        if resource_estimates_dir.is_dir():
            process_directory_generic(f, resource_estimates_dir, output_dir, title_prefix="Resource Estimates: ")
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Type Checker directory not found: {type_check_dir.as_posix()}")

def _add_gnn_exports_section(f: IO[str], output_dir: Path):
    gnn_exports_dir = output_dir / "gnn_exports"
    if gnn_exports_dir.is_dir():
        f.write(f"<div class='section' id='{make_section_id('GNN Exports')}'>\n")
        f.write(f"<h2>GNN Exports (Step 5)</h2>\n")
        
        export_step_report = gnn_exports_dir / "5_export_step_report.md"
        if export_step_report.exists():
            f.write("<h3>Export Step Report</h3>")
            f.write(embed_markdown_file(export_step_report))

        for model_export_dir in sorted(gnn_exports_dir.iterdir()):
            if model_export_dir.is_dir():
                process_directory_generic(f, model_export_dir, output_dir, title_prefix=f"Exports for {model_export_dir.name}: ")
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Exports directory not found: {gnn_exports_dir.as_posix()}")

    gnn_proc_summary_md = output_dir / "gnn_processing_summary.md"
    if gnn_proc_summary_md.exists():
        f.write(f"<div class='section' id='{make_section_id('GNN Processing Summary')}'>\n")
        f.write(f"<h2>GNN Processing Summary (Overall File List)</h2>\n")
        f.write(embed_markdown_file(gnn_proc_summary_md))
        f.write("</div>\n")

def _add_visualizations_section(f: IO[str], output_dir: Path):
    viz_dir = output_dir / "visualization"
    if viz_dir.is_dir():
        f.write(f"<div class='section' id='{make_section_id('GNN Visualizations')}'>\n")
        f.write(f"<h2>GNN Visualizations (Step 6)</h2>\n")
        for model_viz_dir in sorted(viz_dir.iterdir()):
            if model_viz_dir.is_dir():
                process_directory_generic(f, model_viz_dir, output_dir, title_prefix=f"Visualizations for {model_viz_dir.name}: ")
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Visualizations directory not found: {viz_dir.as_posix()}")

def _add_mcp_report_section(f: IO[str], output_dir: Path):
    mcp_report_dir = output_dir / "mcp_processing_step"
    mcp_report_md = mcp_report_dir / "7_mcp_integration_report.md"
    if mcp_report_md.exists():
        f.write(f"<div class='section' id='{make_section_id('MCP Integration Report')}'>\n")
        f.write(f"<h2>MCP Integration Report (Step 7)</h2>\n")
        f.write(embed_markdown_file(mcp_report_md))
        f.write("</div>\n")
    else:
        logger.warning(f"MCP Integration report not found: {mcp_report_md.as_posix()}")

def _add_ontology_processing_section(f: IO[str], output_dir: Path):
    ontology_dir = output_dir / "ontology_processing"
    ontology_report_md = ontology_dir / "ontology_processing_report.md"
    if ontology_report_md.exists():
        f.write(f"<div class='section' id='{make_section_id('Ontology Processing')}'>\n")
        f.write(f"<h2>Ontology Processing (Step 8)</h2>\n")
        f.write(embed_markdown_file(ontology_report_md))
        f.write("</div>\n")
    else:
        logger.warning(f"Ontology Processing report not found: {ontology_report_md.as_posix()}")

def _add_rendered_simulators_section(f: IO[str], output_dir: Path):
    rendered_sim_dir = output_dir / "gnn_rendered_simulators"
    if rendered_sim_dir.is_dir():
        f.write(f"<div class='section' id='{make_section_id('Rendered Simulators')}'>\n")
        f.write(f"<h2>Rendered Simulators (Step 9)</h2>\n")
        for framework_dir in sorted(rendered_sim_dir.iterdir()): 
            if framework_dir.is_dir():
                process_directory_generic(f, framework_dir, output_dir, title_prefix=f"Simulators for {framework_dir.name}: ")
        f.write("</div>\n")
    else:
        logger.warning(f"Rendered Simulators directory not found: {rendered_sim_dir.as_posix()}")

def _add_execution_logs_section(f: IO[str], output_dir: Path):
    exec_logs_main_dir = output_dir / "pymdp_execute_logs"
    if exec_logs_main_dir.is_dir():
        f.write(f"<div class='section' id='{make_section_id('Simulator Execution Logs')}'>\n")
        f.write(f"<h2>Simulator Execution Logs (Step 10)</h2>\n")
        for model_exec_dir in sorted(exec_logs_main_dir.iterdir()):
            if model_exec_dir.is_dir():
                process_directory_generic(f, model_exec_dir, output_dir, title_prefix=f"Execution Logs for {model_exec_dir.name}: ")
        f.write("</div>\n")
    else:
        logger.warning(f"PyMDP Execute Logs directory not found: {exec_logs_main_dir.as_posix()}")

def _add_llm_outputs_section(f: IO[str], output_dir: Path):
    llm_dir = output_dir / "llm_processing_step"
    if llm_dir.is_dir():
        f.write(f"<div class='section' id='{make_section_id('LLM Processing Outputs')}'>\n")
        f.write(f"<h2>LLM Processing Outputs (Step 11)</h2>\n")
        for model_llm_dir in sorted(llm_dir.iterdir()):
            if model_llm_dir.is_dir():
                process_directory_generic(f, model_llm_dir, output_dir, title_prefix=f"LLM Outputs for {model_llm_dir.name}: ")
        f.write("</div>\n")
    else:
        logger.warning(f"LLM Processing directory not found: {llm_dir.as_posix()}")

def _add_pipeline_log_section(f: IO[str], output_dir: Path):
    pipeline_log_dir = output_dir / "logs"
    pipeline_log_file = pipeline_log_dir / "pipeline.log"
    if pipeline_log_file.exists():
        f.write(f"<div class='section' id='{make_section_id('Pipeline Log')}'>\n")
        f.write(f"<h2>Pipeline Log</h2>\n")
        content_html = embed_text_file(pipeline_log_file, max_lines=500) 
        add_collapsible_section(f, "pipeline.log (partial)", content_html, is_open=False)
        f.write("</div>\n")
    else:
        logger.warning(f"Pipeline log file not found: {pipeline_log_file.as_posix()}")

def _add_other_outputs_section(f: IO[str], output_dir: Path, site_output_file: Path):
    f.write(f"<div class='section' id='{make_section_id('Other Output Files')}'>\n")
    f.write("<h2>Other Output Files/Directories</h2>\n")
    other_content_html = ""
    handled_items = {
        "pipeline_execution_summary.json", "gnn_processing_step", "test_reports",
                    "type_check", "gnn_exports", "gnn_processing_summary.md",
        "visualization", "mcp_processing_step", "ontology_processing",
        "gnn_rendered_simulators", "pymdp_execute_logs", "llm_processing_step", "logs",
        site_output_file.name 
    }
    found_other = False
    items_list_html = "<ul>"
    for item in sorted(output_dir.iterdir()):
        if item.name not in handled_items:
            found_other = True
            relative_path = item.relative_to(output_dir).as_posix()
            if item.is_file():
                items_list_html += f'<li><a href="{relative_path}" class="file-link" target="_blank">{item.name}</a></li>'
            elif item.is_dir():
                items_list_html += f"<li><strong>{item.name}/</strong> (Directory - <a href='{relative_path}' class='file-link' target='_blank'>Browse</a>)</li>" 
    items_list_html += "</ul>"
    
    if found_other:
        other_content_html = items_list_html
    else:
        other_content_html = "<p>No other top-level files or directories found or all were processed above.</p>"
    
    f.write(other_content_html)
    f.write("</div>\n")

# --- Main Report Generation Function ---

def generate_html_report(output_dir: Path, site_output_file: Path):
    """
    Generates a single HTML file summarizing the contents of the output_dir.
    """
    logger.info(f"Starting HTML report generation for directory: {output_dir.as_posix()}")
    logger.info(f"Output HTML will be saved to: {site_output_file.as_posix()}")

    with open(site_output_file, 'w', encoding='utf-8') as f:
        f.write(HTML_START_TEMPLATE)

        _add_pipeline_summary_section(f, output_dir)
        _add_gnn_discovery_section(f, output_dir)
        _add_test_reports_section(f, output_dir)
        _add_type_checker_section(f, output_dir)
        _add_gnn_exports_section(f, output_dir) # Also handles gnn_processing_summary.md internally
        _add_visualizations_section(f, output_dir)
        _add_mcp_report_section(f, output_dir)
        _add_ontology_processing_section(f, output_dir)
        _add_rendered_simulators_section(f, output_dir)
        _add_execution_logs_section(f, output_dir)
        _add_llm_outputs_section(f, output_dir)
        _add_pipeline_log_section(f, output_dir)
        _add_other_outputs_section(f, output_dir, site_output_file)

        f.write(HTML_END_TEMPLATE)
    logger.info(f"✅ HTML report generated successfully: {site_output_file.as_posix()}")


def main_site_generator():
    """
    Main function to run the site generator.
    Parses arguments for output directory and site HTML file path.
    """
    parser = argparse.ArgumentParser(description="Generate an HTML summary site from GNN pipeline outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The main output directory of the GNN pipeline (e.g., ../output or ./output)."
    )
    parser.add_argument(
        "--site-output-file",
        type=Path,
        required=True,
        help="The path where the final HTML site file should be saved (e.g., output/gnn_pipeline_summary.html)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.INFO)

    resolved_output_dir = args.output_dir.resolve()
    resolved_site_output_file = args.site_output_file.resolve()

    if not resolved_output_dir.is_dir():
        logger.error(f"Output directory does not exist: {resolved_output_dir.as_posix()}")
        sys.exit(1) # Use sys.exit(1) for error exit code

    resolved_site_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    generate_html_report(resolved_output_dir, resolved_site_output_file)
    sys.exit(0) # Success exit code

if __name__ == "__main__":
    print("src.site.generator called directly. Use 12_site.py for pipeline integration or provide CLI args for direct test.")
    # To test from project root (GeneralizedNotationNotation/):
    # Ensure you have a populated 'output' directory from a previous pipeline run.
    # And install Markdown: pip install Markdown
    # python src/site/generator.py --output-dir output --site-output-file output/test_site.html --verbose
    
    # Basic test execution when called directly
    if Path.cwd().name == "GeneralizedNotationNotation": 
        test_output_dir_arg = Path.cwd() / "output"
        test_site_file_arg = test_output_dir_arg / "test_generated_site_by_generator_main.html"
        
        print(f"Attempting direct test generation with output_dir='{test_output_dir_arg}' and site_output_file='{test_site_file_arg}'")
        
        # Simulate command line arguments for testing
        # Note: This requires the script to be run from the project root for these relative paths to be standard.
        # For more robust direct testing, use absolute paths or ensure correct CWD.
        sys.argv.extend([
            "--output-dir", str(test_output_dir_arg),
            "--site-output-file", str(test_site_file_arg),
            "--verbose"
        ])
        # Check if output dir exists before running test
        if not test_output_dir_arg.is_dir():
            logger.warning(f"Test output directory '{test_output_dir_arg}' does not exist. Skipping direct test run of main_site_generator().")
            print(f"Test output directory '{test_output_dir_arg}' does not exist. Skipping direct test run of main_site_generator().")
        else:
            main_site_generator()
    else:
        print("To test generator.py directly with its main_site_generator(), run from the project root 'GeneralizedNotationNotation/' and provide args, or ensure paths are absolute.") 

def generate_site(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False):
    """Generate static HTML site from pipeline artifacts."""
    log_step_start(logger, "Generating static HTML site from pipeline artifacts")
    
    # Use centralized output directory configuration
    site_output_dir = get_output_dir_for_script("12_site.py", output_dir)
    site_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create site generator instance
        site_generator = SiteGenerator(output_dir=str(site_output_dir))
        
        # Initialize results dictionary
        results = {'success': False, 'files_processed': 0}
        
        # Use performance tracking for site generation
        with performance_tracker.track_operation("generate_static_site"):
            # Find pipeline artifacts
            if recursive:
                artifact_files = list(target_dir.rglob("*"))
            else:
                artifact_files = list(target_dir.glob("*"))
            
            log_step_success(logger, f"Found {len(artifact_files)} pipeline artifacts to process")
            
            # Generate site
            success = site_generator.generate_site(
                pipeline_dir=str(target_dir),
                recursive=recursive
            )
            
            if success:
                results['success'] = True
                results['files_processed'] = len(artifact_files)
                log_step_success(logger, "Static site generation completed successfully")
            else:
                log_step_warning(logger, "Static site generation completed with issues")
        
        return results.get('success', False)
        
    except Exception as e:
        log_step_error(logger, f"Site generation failed: {e}")
        return False 