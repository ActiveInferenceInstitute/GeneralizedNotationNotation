import argparse
import base64
import json
import logging
import os
from pathlib import Path
import shutil
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
import sys
from datetime import datetime

import markdown

# Import logging utilities
try:
    from utils import (
        log_step_start,
        log_step_success,
        log_step_error
    )
    UTILS_AVAILABLE = True
except ImportError:
    # Fallback logging functions if utils not available
    def log_step_start(logger, message):
        logger.info(f"🚀 {message}")
    
    def log_step_success(logger, message):
        logger.info(f"✅ {message}")
    
    def log_step_error(logger, message):
        logger.error(f"❌ {message}")
    
    UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- HTML Templates ---
HTML_START_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Output Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 2em 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        nav {{
            background-color: #343a40;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        nav ul {{
            list-style-type: none;
            padding: 0;
            margin: 0;
            text-align: center;
        }}
        nav ul li {{
            display: inline-block;
            margin: 0;
        }}
        nav ul li a {{
            color: #fff;
            text-decoration: none;
            padding: 15px 20px;
            display: block;
            transition: background-color 0.3s;
        }}
        nav ul li a:hover {{
            background-color: #495057;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .main-content {{
            background-color: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 0.5em;
        }}
        h2 {{
            font-size: 1.8em;
            border-bottom: 3px solid #e9ecef;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h3 {{
            font-size: 1.4em;
            color: #495057;
            margin-top: 30px;
        }}
        h4 {{
            font-size: 1.2em;
            color: #6c757d;
            margin-top: 25px;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            border-left: 4px solid #007bff;
        }}
        code {{
            background-color: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        img, iframe {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 15px 0;
        }}
        .file-link {{
            display: inline-block;
            margin: 5px;
            padding: 10px 15px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: background-color 0.3s;
        }}
        .file-link:hover {{
            background-color: #0056b3;
            transform: translateY(-1px);
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            border-radius: 8px;
            background-color: #fff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .gallery img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .log-output {{
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }}
        .collapsible {{
            background-color: #f8f9fa;
            color: #495057;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1.1em;
            margin-top: 15px;
            border-radius: 6px;
            transition: background-color 0.3s;
        }}
        .collapsible:hover, .collapsible.active {{
            background-color: #e9ecef;
        }}
        .collapsible-content {{
            padding: 0 20px;
            display: none;
            overflow: hidden;
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-top: none;
            border-radius: 0 0 6px 6px;
        }}
        .toc {{
            border: 1px solid #dee2e6;
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc ul li {{
            margin: 8px 0;
        }}
        .toc ul li a {{
            text-decoration: none;
            color: #007bff;
            transition: color 0.3s;
        }}
        .toc ul li a:hover {{
            color: #0056b3;
            text-decoration: underline;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .status-success {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .status-error {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
        }}
        .metadata-item {{
            margin: 5px 0;
        }}
        .metadata-label {{
            font-weight: 600;
            color: #495057;
        }}
        .search-box {{
            width: 100%;
            padding: 10px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 16px;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
        }}
        .hidden {{
            display: none;
        }}
        .file-stats {{
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        .file-type-icon {{
            margin-right: 8px;
        }}
        .json-file-content {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }}
        .markdown-content {{
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 20px;
            margin: 10px 0;
        }}
        .text-file-content {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }}
        .html-file-content {{
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }}
        .loading {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }}
        .error-message {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            border: 1px solid #f5c6cb;
        }}
        .success-message {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            border: 1px solid #c3e6cb;
        }}
        .diagnostics-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .diagnostic-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        .diagnostic-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .diagnostic-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            transition: width 0.3s ease;
        }}
        .table-responsive {{
            overflow-x: auto;
            margin: 15px 0;
        }}
        .enhanced-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .enhanced-table th {{
            background: linear-gradient(135deg, #495057 0%, #343a40 100%);
            color: #fff;
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
        }}
        .enhanced-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .enhanced-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .component-description {{
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 3px;
        }}
        .path-code {{
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            .main-content {{
                padding: 20px;
            }}
            nav ul li {{
                display: block;
            }}
            .gallery {{
                grid-template-columns: 1fr;
            }}
            .diagnostics-summary {{
                grid-template-columns: 1fr;
            }}
            .enhanced-table {{
                font-size: 0.9em;
            }}
            .enhanced-table th,
            .enhanced-table td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Generalized Notation Notation (GNN) Pipeline Output Summary</h1>
        <p>Comprehensive analysis and visualization of GNN model processing results</p>
    </header>
    <nav id="navbar">
        <ul>
            <!-- Nav links will be injected here -->
        </ul>
    </nav>
    <div class="container">
        <div class="main-content">
            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Generated:</span> {generation_time}
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Pipeline Output Directory:</span> {output_dir}
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Total Files Processed:</span> <span id="file-count">Calculating...</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Website Version:</span> 2.1.0
                </div>
            </div>
            
            <input type="text" id="searchBox" class="search-box" placeholder="Search sections and content...">
            
            <div id="toc-container" class="toc">
                <h2>Table of Contents</h2>
                <ul id="toc-list">
                    <!-- TOC items will be injected here -->
                </ul>
            </div>
"""

HTML_END_TEMPLATE = """
        </div>
    </div>
    <script>
        // Enhanced search functionality with highlighting
        function setupSearch() {
            const searchBox = document.getElementById('searchBox');
            const sections = document.querySelectorAll('.section');
            
            searchBox.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                let visibleCount = 0;
                
                sections.forEach(section => {
                    const title = section.querySelector('h2')?.textContent.toLowerCase() || '';
                    const content = section.textContent.toLowerCase();
                    const isVisible = title.includes(searchTerm) || content.includes(searchTerm);
                    
                    section.style.display = isVisible ? 'block' : 'none';
                    if (isVisible) visibleCount++;
                    
                    // Highlight search terms
                    if (searchTerm && isVisible) {
                        highlightText(section, searchTerm);
                    } else {
                        removeHighlighting(section);
                    }
                });
                
                // Update file count
                document.getElementById('file-count').textContent = `${visibleCount} sections visible`;
            });
        }

        // Highlight search terms
        function highlightText(element, searchTerm) {
            const walker = document.createTreeWalker(
                element,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node);
            }
            
            textNodes.forEach(textNode => {
                const text = textNode.textContent;
                const regex = new RegExp(`(${searchTerm})`, 'gi');
                if (regex.test(text)) {
                    const highlightedText = text.replace(regex, '<mark>$1</mark>');
                    const span = document.createElement('span');
                    span.innerHTML = highlightedText;
                    textNode.parentNode.replaceChild(span, textNode);
                }
            });
        }

        // Remove highlighting
        function removeHighlighting(element) {
            const marks = element.querySelectorAll('mark');
            marks.forEach(mark => {
                const parent = mark.parentNode;
                parent.replaceChild(document.createTextNode(mark.textContent), mark);
                parent.normalize();
            });
        }

        // Enhanced collapsible sections
        function setupCollapsible() {
            const collapsibles = document.getElementsByClassName("collapsible");
            for (let i = 0; i < collapsibles.length; i++) {
                collapsibles[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    const content = this.nextElementSibling;
                    if (content.style.display === "block") {
                        content.style.display = "none";
                    } else {
                        content.style.display = "block";
                    }
                });
            }
        }

        // Smooth scroll for navigation
        function setupSmoothScroll() {
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
        }
        
        // Dynamic navigation and TOC generation
        function setupNavigation() {
            const sections = document.querySelectorAll('.section h2');
            const navUl = document.querySelector('#navbar ul');
            const tocUl = document.getElementById('toc-list');
            
            // Clear existing content
            navUl.innerHTML = '';
            tocUl.innerHTML = '';
            
            sections.forEach(section => {
                const sectionId = section.parentElement.id;
                const sectionTitle = section.textContent;
                
                // Create navigation link
                const navLi = document.createElement('li');
                const navLink = document.createElement('a');
                navLink.href = `#${sectionId}`;
                navLink.textContent = sectionTitle;
                navLi.appendChild(navLink);
                navUl.appendChild(navLi);
                
                // Create TOC link
                const tocLi = document.createElement('li');
                const tocLink = document.createElement('a');
                tocLink.href = `#${sectionId}`;
                tocLink.textContent = sectionTitle;
                tocLi.appendChild(tocLink);
                tocUl.appendChild(tocLi);
            });
        }

        // Initialize all functionality
        document.addEventListener('DOMContentLoaded', function() {
            setupSearch();
            setupCollapsible();
            setupSmoothScroll();
            setupNavigation();
            
            // Update file count on load
            const sections = document.querySelectorAll('.section');
            document.getElementById('file-count').textContent = `${sections.length} sections`;
        });
    </script>
</body>
</html>
"""

def make_section_id(title: str) -> str:
    """Generates a URL-friendly ID from a title."""
    import re
    # Remove special characters and replace spaces with hyphens
    id_str = re.sub(r'[^\w\s-]', '', title.lower())
    id_str = re.sub(r'[-\s]+', '-', id_str)
    return id_str.strip('-')

def add_collapsible_section(f: IO[str], title: str, content_html: str, is_open: bool = False):
    """Adds a collapsible section to the HTML file."""
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
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        return f'''<div style="text-align: center; margin: 15px 0;">
            <img src="data:image/{file_path.suffix.lstrip(".")};base64,{encoded_string}" 
                 alt="{alt}" loading="lazy" style="max-width: 100%; height: auto;">
            <br><small>{file_path.name} ({size_mb:.2f} MB)</small>
        </div>'''
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
        
        # Enhanced markdown processing with more extensions
        html_content = markdown.markdown(
            content, 
            extensions=[
                'fenced_code', 
                'tables', 
                'sane_lists', 
                'codehilite',
                'toc',
                'attr_list'
            ]
        )
        
        return f'<div class="markdown-content">{html_content}</div>'
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
        truncated_note = ""
        if len(lines) > max_lines:
            truncated_note = f"\n... (file truncated, showing {max_lines} of {len(lines)} lines)"
        
        file_size = file_path.stat().st_size
        size_kb = file_size / 1024
        
        return f'''<div class="text-file-content">
            <div class="metadata">
                <span class="metadata-label">File:</span> {file_path.name} ({size_kb:.1f} KB)
            </div>
            <pre class="log-output"><code>{content}{truncated_note}</code></pre>
        </div>'''
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
        
        file_size = file_path.stat().st_size
        size_kb = file_size / 1024
        
        return f'''<div class="json-file-content">
            <div class="metadata">
                <span class="metadata-label">File:</span> {file_path.name} ({size_kb:.1f} KB)
            </div>
            <pre class="log-output"><code>{json.dumps(data, indent=2)}</code></pre>
        </div>'''
    except Exception as e:
        logger.error(f"Error reading or parsing JSON file {file_path.as_posix()}: {e}")
        return f"<p>Error processing JSON file {file_path.name}: {e}</p>"
        
def embed_html_file(file_path: Path) -> str:
    """Embeds an HTML file content with improved handling."""
    if not file_path.exists():
        return f"<p>HTML file not found: {file_path.as_posix()}</p>"
    try:
        with open(file_path, 'r', encoding='utf-8') as html_f:
            content = html_f.read()
        
        file_size = file_path.stat().st_size
        size_kb = file_size / 1024
        
        # Check for complexity indicators
        is_complex = any(indicator in content.lower() for indicator in [
            "<script", "<link rel=\"stylesheet\"", "plotly", "d3", "chart"
        ])
        
        if is_complex:
            relative_path = file_path.name
            return f'''<div class="html-file-content">
                <div class="metadata">
                    <span class="metadata-label">File:</span> {file_path.name} ({size_kb:.1f} KB)
                    <span class="status-badge status-warning">Complex HTML</span>
                </div>
                <p><a href="{relative_path}" target="_blank" class="file-link">View standalone: {file_path.name}</a></p>
                <iframe src="{relative_path}" width="100%" height="600px" 
                        style="border:1px solid #dee2e6; border-radius: 6px;" 
                        sandbox="allow-scripts allow-same-origin allow-popups allow-forms"></iframe>
            </div>'''
        else:
            return f'''<div class="html-file-content">
                <div class="metadata">
                    <span class="metadata-label">File:</span> {file_path.name} ({size_kb:.1f} KB)
                </div>
                {content}
            </div>'''

    except Exception as e:
        logger.error(f"Error processing HTML file {file_path.as_posix()}: {e}")
        return f"<p>Error processing HTML file {file_path.name}: {e}</p>"

def process_directory_generic(f: IO[str], dir_path: Path, base_output_dir: Path, title_prefix: str = ""):
    """Generic handler for directories with improved organization."""
    if not dir_path.is_dir():
        return

    # Categorize files
    file_categories = {
        'images': [],
        'markdown': [],
        'html': [],
        'json': [],
        'text': [],
        'other': []
    }

    for item in sorted(dir_path.iterdir()):
        if item.is_file():
            suffix = item.suffix.lower()
            if suffix in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                file_categories['images'].append(item)
            elif suffix == '.md':
                file_categories['markdown'].append(item)
            elif suffix in ['.html', '.htm']:
                file_categories['html'].append(item)
            elif suffix == '.json':
                file_categories['json'].append(item)
            elif suffix in ['.txt', '.log', '.out']:
                file_categories['text'].append(item)
            else:
                file_categories['other'].append(item)
    
    content_html = ""
    
    # Process images
    if file_categories['images']:
        content_html += "<h3>Images</h3><div class='gallery'>"
        for img_path in file_categories['images']:
            content_html += embed_image(img_path)
        content_html += "</div>"

    # Process markdown files
    if file_categories['markdown']:
        content_html += "<h3>Markdown Reports</h3>"
        for md_path in file_categories['markdown']:
            content_html += f"<h4>{md_path.name}</h4>"
            content_html += embed_markdown_file(md_path)
            
    # Process HTML files
    if file_categories['html']:
        content_html += "<h3>HTML Reports</h3>"
        for html_path in file_categories['html']:
            content_html += f"<h4>{html_path.name}</h4>"
            content_html += embed_html_file(html_path)

    # Process JSON files
    if file_categories['json']:
        content_html += "<h3>JSON Files</h3>"
        for json_path in file_categories['json']:
            content_html += f"<h4>{json_path.name}</h4>"
            content_html += embed_json_file(json_path)

    # Process text files
    if file_categories['text']:
        content_html += "<h3>Text/Log Files</h3>"
        for txt_path in file_categories['text']:
            content_html += f"<h4>{txt_path.name}</h4>"
            content_html += embed_text_file(txt_path)
            
    # Process other files
    if file_categories['other']:
        content_html += "<h3>Other Files</h3><ul>"
        for other_path in file_categories['other']:
            relative_other_path = other_path.relative_to(base_output_dir).as_posix()
            content_html += f'<li><a href="{relative_other_path}" class="file-link" target="_blank">{other_path.name}</a></li>'
        content_html += "</ul>"

    if content_html:
        section_title = f"{title_prefix}{dir_path.name}"
        section_id = make_section_id(section_title)
        f.write(f"<div class='section' id='{section_id}'>\n")
        f.write(f"<h2>{section_title}</h2>\n")
        f.write(content_html)
        f.write("</div>\n")

# --- Directory Detection Utilities ---

def find_content_directory(base_dir: Path, expected_name: str) -> Optional[Path]:
    """
    Find a content directory, handling both flat and nested structures.
    
    Args:
        base_dir: Base directory to search in
        expected_name: Expected directory name
        
    Returns:
        Path to the content directory if found, None otherwise
    """
    # Try direct path first
    direct_path = base_dir / expected_name
    if direct_path.exists() and direct_path.is_dir():
        # Check if it has content or a nested directory
        nested_path = direct_path / expected_name
        if nested_path.exists() and nested_path.is_dir():
            # Prefer nested if it has more content
            nested_items = list(nested_path.iterdir()) if nested_path.exists() else []
            direct_items = list(direct_path.iterdir()) if direct_path.exists() else []
            if len(nested_items) > len([item for item in direct_items if item.is_file()]):
                return nested_path
        return direct_path
    
    # Try nested path
    nested_path = base_dir / expected_name / expected_name
    if nested_path.exists() and nested_path.is_dir():
        return nested_path
    
    # Try alternative naming patterns
    alternatives = [
        expected_name.replace("_", "-"),
        expected_name.replace("-", "_"),
        expected_name + "s",
        expected_name[:-1] if expected_name.endswith("s") else expected_name
    ]
    
    for alt_name in alternatives:
        alt_path = base_dir / alt_name
        if alt_path.exists() and alt_path.is_dir():
            return alt_path
    
    return None

def generate_directory_diagnostics(output_dir: Path) -> Dict[str, Any]:
    """
    Generate diagnostics about directory structure and content discovery.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary with diagnostics information
    """
    expected_dirs = {
        "pipeline_execution_summary.json": "Pipeline execution summary",
        "gnn_processing_step": "GNN file processing results",
        "test_reports": "Test execution reports",
        "type_check": "Type checking results",
        "gnn_exports": "GNN export artifacts",
        "visualization": "Generated visualizations",
        "mcp_processing_step": "MCP integration results",
        "ontology_processing": "Ontology processing results",
        "gnn_rendered_simulators": "Rendered simulator code",
        "execution_results": "Execution results",
        "llm_processing_step": "LLM processing outputs",
        "sapf_processing_step": "SAPF generation results",
        "logs": "Pipeline logs"
    }
    
    diagnostics = {
        "expected_count": len(expected_dirs),
        "found_count": 0,
        "missing_count": 0,
        "found_items": {},
        "missing_items": [],
        "directory_structure": {},
        "total_files": 0,
        "total_size_mb": 0
    }
    
    for expected_name, description in expected_dirs.items():
        if expected_name.endswith(".json"):
            # Handle files
            file_path = output_dir / expected_name
            if file_path.exists() and file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                diagnostics["found_items"][expected_name] = {
                    "type": "file",
                    "description": description,
                    "path": str(file_path.relative_to(output_dir)),
                    "size_mb": round(size_mb, 2),
                    "exists": True
                }
                diagnostics["found_count"] += 1
                diagnostics["total_files"] += 1
                diagnostics["total_size_mb"] += size_mb
            else:
                diagnostics["missing_items"].append({
                    "name": expected_name,
                    "type": "file",
                    "description": description
                })
                diagnostics["missing_count"] += 1
        else:
            # Handle directories
            found_dir = find_content_directory(output_dir, expected_name)
            if found_dir:
                try:
                    file_count = len(list(found_dir.rglob("*")))
                    dir_size_mb = sum(f.stat().st_size for f in found_dir.rglob("*") if f.is_file()) / (1024 * 1024)
                    diagnostics["found_items"][expected_name] = {
                        "type": "directory",
                        "description": description,
                        "path": str(found_dir.relative_to(output_dir)),
                        "file_count": file_count,
                        "size_mb": round(dir_size_mb, 2),
                        "exists": True
                    }
                    diagnostics["found_count"] += 1
                    diagnostics["total_files"] += file_count
                    diagnostics["total_size_mb"] += dir_size_mb
                except (OSError, PermissionError):
                    diagnostics["found_items"][expected_name] = {
                        "type": "directory",
                        "description": description,
                        "path": str(found_dir.relative_to(output_dir)),
                        "file_count": 0,
                        "size_mb": 0,
                        "exists": True,
                        "error": "Permission denied"
                    }
                    diagnostics["found_count"] += 1
            else:
                diagnostics["missing_items"].append({
                    "name": expected_name,
                    "type": "directory",
                    "description": description
                })
                diagnostics["missing_count"] += 1
    
    # Add information about other directories found
    other_items = []
    for item in output_dir.iterdir():
        if item.name not in expected_dirs:
            if item.is_dir():
                try:
                    file_count = len(list(item.rglob("*")))
                    other_items.append({
                        "name": item.name,
                        "type": "directory",
                        "file_count": file_count
                    })
                except (OSError, PermissionError):
                    other_items.append({
                        "name": item.name,
                        "type": "directory",
                        "file_count": 0,
                        "error": "Permission denied"
                    })
            elif item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                other_items.append({
                    "name": item.name,
                    "type": "file",
                    "size_mb": round(size_mb, 2)
                })
    
    diagnostics["other_items"] = other_items
    diagnostics["completion_percentage"] = round((diagnostics["found_count"] / diagnostics["expected_count"]) * 100, 1)
    
    return diagnostics

# --- Section specific helpers ---

def _add_pipeline_summary_section(f: IO[str], output_dir: Path):
    """Add pipeline execution summary section."""
    summary_json_path = output_dir / "pipeline_execution_summary.json"
    if summary_json_path.exists():
        f.write(f"<div class='section' id='{make_section_id('Pipeline Execution Summary')}'>\n")
        f.write(f"<h2>Pipeline Execution Summary</h2>\n")
        f.write(embed_json_file(summary_json_path))
        f.write("</div>\n")
    else:
        logger.warning(f"Pipeline summary JSON not found: {summary_json_path.as_posix()}")

def _add_diagnostics_section(f: IO[str], output_dir: Path):
    """Add pipeline diagnostics section."""
    diagnostics = generate_directory_diagnostics(output_dir)
    
    f.write(f"<div class='section' id='{make_section_id('Pipeline Diagnostics')}'>\n")
    f.write(f"<h2>Pipeline Diagnostics</h2>\n")
    
    # Enhanced summary metrics with cards
    f.write("<div class='diagnostics-summary'>")
    f.write(f"<div class='diagnostic-card'>")
    f.write(f"<div class='diagnostic-value'>{diagnostics['completion_percentage']}%</div>")
    f.write(f"<div class='diagnostic-label'>Pipeline Completion</div>")
    f.write(f"<div class='progress-bar'><div class='progress-fill' style='width: {diagnostics['completion_percentage']}%'></div></div>")
    f.write(f"</div>")
    
    f.write(f"<div class='diagnostic-card'>")
    f.write(f"<div class='diagnostic-value'>{diagnostics['found_count']}/{diagnostics['expected_count']}</div>")
    f.write(f"<div class='diagnostic-label'>Found Components</div>")
    f.write(f"</div>")
    
    f.write(f"<div class='diagnostic-card'>")
    f.write(f"<div class='diagnostic-value'>{diagnostics['total_files']}</div>")
    f.write(f"<div class='diagnostic-label'>Total Files</div>")
    f.write(f"</div>")
    
    f.write(f"<div class='diagnostic-card'>")
    f.write(f"<div class='diagnostic-value'>{diagnostics['total_size_mb']:.1f}</div>")
    f.write(f"<div class='diagnostic-label'>Total Size (MB)</div>")
    f.write(f"</div>")
    f.write("</div>")
    
    # Found items table with enhanced styling
    if diagnostics['found_items']:
        f.write("<h3>Found Pipeline Components</h3>")
        f.write("<div class='table-responsive'>")
        f.write("<table class='enhanced-table'>")
        f.write("<tr><th>Component</th><th>Type</th><th>Path</th><th>Size/Files</th><th>Status</th></tr>")
        
        for name, info in diagnostics['found_items'].items():
            status_class = "status-success" if info['exists'] else "status-error"
            if info['type'] == 'file':
                size_info = f"{info['size_mb']} MB"
            else:
                size_info = f"{info['file_count']} files ({info['size_mb']:.1f} MB)"
            
            error_note = f" ({info.get('error', '')})" if 'error' in info else ""
            
            f.write(f"<tr>")
            f.write(f"<td><strong>{name}</strong><div class='component-description'>{info['description']}</div></td>")
            f.write(f"<td>{info['type'].title()}</td>")
            f.write(f"<td><span class='path-code'>{info['path']}</span></td>")
            f.write(f"<td>{size_info}</td>")
            f.write(f"<td><span class='status-badge {status_class}'>Found{error_note}</span></td>")
            f.write(f"</tr>")
        
        f.write("</table>")
        f.write("</div>")
    
    # Missing items with enhanced styling
    if diagnostics['missing_items']:
        f.write("<h3>Missing Pipeline Components</h3>")
        f.write("<div class='table-responsive'>")
        f.write("<table class='enhanced-table'>")
        f.write("<tr><th>Component</th><th>Type</th><th>Description</th><th>Status</th></tr>")
        
        for item in diagnostics['missing_items']:
            f.write(f"<tr>")
            f.write(f"<td><strong>{item['name']}</strong></td>")
            f.write(f"<td>{item['type'].title()}</td>")
            f.write(f"<td>{item['description']}</td>")
            f.write(f"<td><span class='status-badge status-warning'>Missing</span></td>")
            f.write(f"</tr>")
        
        f.write("</table>")
        f.write("</div>")
    
    # Other items
    if diagnostics['other_items']:
        f.write("<h3>Additional Files/Directories</h3>")
        f.write("<ul>")
        for item in diagnostics['other_items']:
            if item['type'] == 'directory':
                file_info = f" ({item['file_count']} files)" if 'file_count' in item else ""
                error_info = f" - {item['error']}" if 'error' in item else ""
                f.write(f"<li><strong>{item['name']}/</strong>{file_info}{error_info}</li>")
            else:
                f.write(f"<li>{item['name']} ({item['size_mb']} MB)</li>")
        f.write("</ul>")
    
    f.write("</div>\n")

def _add_gnn_discovery_section(f: IO[str], output_dir: Path):
    """Add GNN discovery section."""
    gnn_discovery_dir = find_content_directory(output_dir, "gnn_processing_step")
    if gnn_discovery_dir:
        f.write(f"<div class='section' id='{make_section_id('GNN Discovery')}'>\n")
        f.write(f"<h2>GNN Discovery (Step 1)</h2>\n")
        
        # Look for discovery report
        gnn_discovery_report = gnn_discovery_dir / "1_gnn_discovery_report.md"
        if gnn_discovery_report.exists():
            f.write(embed_markdown_file(gnn_discovery_report))
        else:
            # Process the entire directory if specific report not found
            process_directory_generic(f, gnn_discovery_dir, output_dir, title_prefix="")
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Discovery directory not found in expected locations")

def _add_test_reports_section(f: IO[str], output_dir: Path):
    """Add test reports section."""
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
    """Add type checker section."""
    type_check_dir = find_content_directory(output_dir, "type_check")
    if type_check_dir:
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
        
        # If no specific content found, process the entire directory
        if not any([type_check_report_md.exists(), resource_data_json.exists(), 
                   html_vis_dir.is_dir(), resource_estimates_dir.is_dir()]):
            process_directory_generic(f, type_check_dir, output_dir, title_prefix="")
        
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Type Checker directory not found in expected locations")

def _add_gnn_exports_section(f: IO[str], output_dir: Path):
    """Add GNN exports section."""
    gnn_exports_dir = find_content_directory(output_dir, "gnn_exports")
    if gnn_exports_dir:
        f.write(f"<div class='section' id='{make_section_id('GNN Exports')}'>\n")
        f.write(f"<h2>GNN Exports (Step 5)</h2>\n")
        
        export_step_report = gnn_exports_dir / "5_export_step_report.md"
        if export_step_report.exists():
            f.write("<h3>Export Step Report</h3>")
            f.write(embed_markdown_file(export_step_report))

        # Process subdirectories
        subdirs_processed = False
        for model_export_dir in sorted(gnn_exports_dir.iterdir()):
            if model_export_dir.is_dir():
                process_directory_generic(f, model_export_dir, output_dir, title_prefix=f"Exports for {model_export_dir.name}: ")
                subdirs_processed = True
        
        # If no subdirectories, process the main directory
        if not subdirs_processed:
            process_directory_generic(f, gnn_exports_dir, output_dir, title_prefix="")
        
        f.write("</div>\n")
    else:
        logger.warning(f"GNN Exports directory not found in expected locations")

    gnn_proc_summary_md = output_dir / "gnn_processing_summary.md"
    if gnn_proc_summary_md.exists():
        f.write(f"<div class='section' id='{make_section_id('GNN Processing Summary')}'>\n")
        f.write(f"<h2>GNN Processing Summary (Overall File List)</h2>\n")
        f.write(embed_markdown_file(gnn_proc_summary_md))
        f.write("</div>\n")

def _add_visualizations_section(f: IO[str], output_dir: Path):
    """Add visualizations section."""
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
    """Add MCP report section."""
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
    """Add ontology processing section."""
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
    """Add rendered simulators section."""
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
    """Add execution logs section."""
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
    """Add LLM outputs section."""
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
    """Add pipeline log section."""
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

def _add_other_outputs_section(f: IO[str], output_dir: Path, website_output_file: Path):
    """Add other outputs section."""
    f.write(f"<div class='section' id='{make_section_id('Other Output Files')}'>\n")
    f.write("<h2>Other Output Files/Directories</h2>\n")
    
    handled_items = {
        "pipeline_execution_summary.json", "gnn_processing_step", "test_reports",
        "type_check", "gnn_exports", "gnn_processing_summary.md",
        "visualization", "mcp_processing_step", "ontology_processing",
        "gnn_rendered_simulators", "pymdp_execute_logs", "llm_processing_step", "logs",
        website_output_file.name 
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
        f.write(items_list_html)
    else:
        f.write("<p>No other top-level files or directories found or all were processed above.</p>")
    
    f.write("</div>\n")

# --- Main Report Generation Function ---

def generate_html_report(output_dir: Path, website_output_file: Path):
    """
    Generates a single HTML file summarizing the contents of the output_dir as a website.
    """
    logger.info(f"Starting HTML report generation for directory: {output_dir.as_posix()}")
    logger.info(f"Output HTML will be saved to: {website_output_file.as_posix()}")

    # Prepare template variables
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir_str = str(output_dir.resolve())

    with open(website_output_file, 'w', encoding='utf-8') as f:
        # Write start template with variables
        start_template = HTML_START_TEMPLATE.format(
            generation_time=generation_time,
            output_dir=output_dir_str
        )
        f.write(start_template)

        # Add all sections
        _add_pipeline_summary_section(f, output_dir)
        _add_diagnostics_section(f, output_dir)
        _add_gnn_discovery_section(f, output_dir)
        _add_test_reports_section(f, output_dir)
        _add_type_checker_section(f, output_dir)
        _add_gnn_exports_section(f, output_dir)
        _add_visualizations_section(f, output_dir)
        _add_mcp_report_section(f, output_dir)
        _add_ontology_processing_section(f, output_dir)
        _add_rendered_simulators_section(f, output_dir)
        _add_execution_logs_section(f, output_dir)
        _add_llm_outputs_section(f, output_dir)
        _add_pipeline_log_section(f, output_dir)
        _add_other_outputs_section(f, output_dir, website_output_file)

        f.write(HTML_END_TEMPLATE)
    
    logger.info(f"✅ HTML report generated successfully: {website_output_file.as_posix()}")

def main_website_generator():
    """
    Main function to run the website generator.
    Parses arguments for output directory and website HTML file path.
    """
    parser = argparse.ArgumentParser(description="Generate an HTML summary website from GNN pipeline outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The main output directory of the GNN pipeline (e.g., ../output or ./output)."
    )
    parser.add_argument(
        "--website-output-file",
        type=Path,
        required=True,
        help="The path where the final HTML website file should be saved (e.g., output/gnn_pipeline_summary.html)."
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
    resolved_website_output_file = args.website_output_file.resolve()

    if not resolved_output_dir.is_dir():
        logger.error(f"Output directory does not exist: {resolved_output_dir.as_posix()}")
        sys.exit(1)

    resolved_website_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    generate_html_report(resolved_output_dir, resolved_website_output_file)
    sys.exit(0)

if __name__ == "__main__":
    print("src.website.generator called directly. Use 13_website.py for pipeline integration or provide CLI args for direct test.")
    
    # Basic test execution when called directly
    if Path.cwd().name == "GeneralizedNotationNotation": 
        test_output_dir_arg = Path.cwd() / "output"
        test_website_file_arg = test_output_dir_arg / "test_generated_website_by_generator_main.html"
        
        print(f"Attempting direct test generation with output_dir='{test_output_dir_arg}' and website_output_file='{test_website_file_arg}'")
        
        # Simulate command line arguments for testing
        sys.argv.extend([
            "--output-dir", str(test_output_dir_arg),
            "--website-output-file", str(test_website_file_arg),
            "--verbose"
        ])
        
        # Check if output dir exists before running test
        if not test_output_dir_arg.is_dir():
            logger.warning(f"Test output directory '{test_output_dir_arg}' does not exist. Skipping direct test run of main_website_generator().")
            print(f"Test output directory '{test_output_dir_arg}' does not exist. Skipping direct test run of main_website_generator().")
        else:
            main_website_generator()
    else:
        print("To test generator.py directly with its main_website_generator(), run from the project root 'GeneralizedNotationNotation/' and provide args, or ensure paths are absolute.")

def generate_website(logger: logging.Logger, output_dir: Path, website_output_dir: Path):
    """Generate static HTML website from pipeline artifacts."""
    log_step_start(logger, "Generating static HTML website from pipeline artifacts")
    
    try:
        # Ensure the website output directory exists
        website_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the website HTML file
        website_html_file = website_output_dir / "index.html"
        
        # Log the generation process
        logger.info(f"Starting website generation from {output_dir} to {website_output_dir}")
        
        # Generate the HTML report
        generate_html_report(output_dir, website_html_file)
        
        log_step_success(logger, f"Static website generation completed successfully: {website_html_file}")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Website generation failed: {e}")
        return False 

def _add_semantic_structure_improvements():
    """Additional semantic HTML improvements that can be added to future versions."""
    return """
    <!-- Enhanced semantic structure improvements for future versions:
    - Add proper ARIA labels for navigation and search
    - Include skip navigation links for accessibility
    - Add structured data markup for better SEO
    - Implement keyboard navigation support
    - Add dark mode toggle functionality
    - Include print stylesheet optimization
    -->
    """

# Enhanced HTML structure with better accessibility
ENHANCED_HTML_START_ADDITIONS = """
    <!-- Enhanced accessibility and semantic structure -->
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <div id="top" aria-label="Top of page"></div>
"""

# Add improved accessibility attributes to the main template
def get_enhanced_html_start_template():
    """Get the HTML start template with enhanced accessibility features."""
    base_template = HTML_START_TEMPLATE
    # Add lang attribute and improved accessibility
    enhanced_template = base_template.replace(
        '<html lang="en">',
        '<html lang="en" dir="ltr">'
    ).replace(
        '<body>',
        '<body>' + ENHANCED_HTML_START_ADDITIONS
    ).replace(
        '<div class="container">',
        '<main id="main-content" class="container" role="main" aria-label="Pipeline output content">'
    ).replace(
        '<div class="main-content">',
        '<div class="main-content" role="document">'
    ).replace(
        'id="searchBox"',
        'id="searchBox" role="searchbox" aria-label="Search pipeline content"'
    ).replace(
        'id="navbar"',
        'id="navbar" role="navigation" aria-label="Pipeline sections navigation"'
    )
    
    # Add skip link CSS
    skip_link_css = """
        .skip-link {
            position: absolute;
            top: -40px;
            left: 6px;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
            border-radius: 3px;
            z-index: 10000;
        }
        .skip-link:focus {
            top: 6px;
        }
    """
    
    return enhanced_template.replace('</style>', skip_link_css + '\n    </style>') 