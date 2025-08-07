#!/usr/bin/env python3
"""
Website generator module for GNN pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import json
import shutil
from datetime import datetime

class WebsiteGenerator:
    """Generates static HTML websites from pipeline artifacts."""
    
    def __init__(self):
        """Initialize the website generator."""
        self.template_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"
    
    def generate_website(self, website_data: dict) -> dict:
        """Generate a complete website from the provided data."""
        try:
            result = {
                "success": True,
                "pages_created": 0,
                "errors": [],
                "warnings": []
            }
            
            # Extract data
            output_dir = Path(website_data.get("output_dir", "output/website"))
            input_dir = Path(website_data.get("input_dir", "output"))
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create pages
            pages_result = self.create_pages(output_dir, website_data)
            result["pages_created"] = pages_result.get("pages_created", 0)
            result["errors"].extend(pages_result.get("errors", []))
            result["warnings"].extend(pages_result.get("warnings", []))
            
            # Copy static assets
            if self.static_dir.exists():
                static_output = output_dir / "static"
                shutil.copytree(self.static_dir, static_output, dirs_exist_ok=True)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "pages_created": 0,
                "errors": [str(e)],
                "warnings": []
            }
    
    def create_pages(self, output_dir: Path, data: dict) -> dict:
        """Create individual website pages."""
        result = {
            "pages_created": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create index page
            index_content = self._generate_index_page(data)
            index_file = output_dir / "index.html"
            with open(index_file, 'w') as f:
                f.write(index_content)
            result["pages_created"] += 1
            
            # Create analysis page
            if data.get("analysis_results"):
                analysis_content = self._generate_analysis_page(data)
                analysis_file = output_dir / "analysis.html"
                with open(analysis_file, 'w') as f:
                    f.write(analysis_content)
                result["pages_created"] += 1
            
            # Create visualization page
            if data.get("visualization_results"):
                viz_content = self._generate_visualization_page(data)
                viz_file = output_dir / "visualization.html"
                with open(viz_file, 'w') as f:
                    f.write(viz_content)
                result["pages_created"] += 1
            
        except Exception as e:
            result["errors"].append(f"Error creating pages: {e}")
        
        return result
    
    def _generate_index_page(self, data: dict) -> str:
        """Generate the main index page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .link {{ color: #0066cc; text-decoration: none; }}
        .link:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GNN Pipeline Results</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Pipeline Overview</h2>
        <p>This website contains the results from the GNN processing pipeline.</p>
        
        <h3>Available Pages</h3>
        <ul>
            <li><a href="analysis.html" class="link">Analysis Results</a></li>
            <li><a href="visualization.html" class="link">Visualizations</a></li>
            <li><a href="render.html" class="link">Generated Code</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Pipeline Statistics</h2>
        <ul>
            <li>Files Processed: {data.get('processed_files', 0)}</li>
            <li>Analysis Results: {len(data.get('analysis_results', []))}</li>
            <li>Visualizations: {len(data.get('visualization_results', []))}</li>
        </ul>
    </div>
</body>
</html>"""
    
    def _generate_analysis_page(self, data: dict) -> str:
        """Generate the analysis results page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - GNN Pipeline</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .result {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analysis Results</h1>
        <p><a href="index.html">← Back to Index</a></p>
    </div>
    
    <div class="section">
        <h2>Statistical Analysis</h2>
        {self._format_analysis_results(data.get('analysis_results', []))}
    </div>
    
    <div class="section">
        <h2>Complexity Metrics</h2>
        {self._format_complexity_results(data.get('complexity_metrics', []))}
    </div>
</body>
</html>"""
    
    def _generate_visualization_page(self, data: dict) -> str:
        """Generate the visualization page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizations - GNN Pipeline</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .viz {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Visualizations</h1>
        <p><a href="index.html">← Back to Index</a></p>
    </div>
    
    <div class="section">
        <h2>Generated Visualizations</h2>
        {self._format_visualization_results(data.get('visualization_results', []))}
    </div>
</body>
</html>"""
    
    def _format_analysis_results(self, results: List[Dict]) -> str:
        """Format analysis results for HTML."""
        if not results:
            return "<p>No analysis results available.</p>"
        
        html = ""
        for result in results:
            html += f"""
            <div class="result">
                <h3>{result.get('file_name', 'Unknown')}</h3>
                <p><strong>Variables:</strong> {len(result.get('variables', []))}</p>
                <p><strong>Connections:</strong> {len(result.get('connections', []))}</p>
                <p><strong>Complexity:</strong> {result.get('complexity_metrics', {}).get('total_elements', 0)}</p>
            </div>"""
        
        return html
    
    def _format_complexity_results(self, results: List[Dict]) -> str:
        """Format complexity results for HTML."""
        if not results:
            return "<p>No complexity metrics available.</p>"
        
        html = ""
        for result in results:
            html += f"""
            <div class="result">
                <h3>{result.get('file_name', 'Unknown')}</h3>
                <p><strong>Cyclomatic Complexity:</strong> {result.get('cyclomatic_complexity', 0):.2f}</p>
                <p><strong>Cognitive Complexity:</strong> {result.get('cognitive_complexity', 0):.2f}</p>
                <p><strong>Maintainability Index:</strong> {result.get('maintainability_index', 0):.2f}</p>
            </div>"""
        
        return html
    
    def _format_visualization_results(self, results: List[Dict]) -> str:
        """Format visualization results for HTML."""
        if not results:
            return "<p>No visualizations available.</p>"
        
        html = ""
        for result in results:
            html += f"""
            <div class="viz">
                <h3>{result.get('title', 'Visualization')}</h3>
                <p>{result.get('description', 'No description available')}</p>
                <img src="{result.get('file_path', '')}" alt="{result.get('title', 'Visualization')}" style="max-width: 100%;">
            </div>"""
        
        return html

def generate_website(logger, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Generate a website from pipeline artifacts."""
    try:
        generator = WebsiteGenerator()
        
        # Collect website data
        website_data = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "analysis_results": [],
            "visualization_results": [],
            "render_results": []
        }
        
        # Find analysis results
        analysis_dir = input_dir / "analysis"
        if analysis_dir.exists():
            for json_file in analysis_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    website_data["analysis_results"].append(data)
                except Exception as e:
                    logger.warning(f"Could not read analysis file {json_file}: {e}")
        
        # Find visualization results
        viz_dir = input_dir / "visualization"
        if viz_dir.exists():
            for html_file in viz_dir.glob("*.html"):
                website_data["visualization_results"].append({
                    "title": html_file.stem,
                    "file_path": str(html_file.relative_to(output_dir)),
                    "description": f"Visualization from {html_file.stem}"
                })
        
        # Generate website
        result = generator.generate_website(website_data)
        
        return {
            "success": result["success"],
            "pages_created": result["pages_created"],
            "errors": result["errors"],
            "warnings": result["warnings"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "pages_created": 0,
            "errors": [str(e)],
            "warnings": []
        } 