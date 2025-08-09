#!/usr/bin/env python3
"""
Advanced visualization module for GNN pipeline (real implementations).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

# Use non-interactive backend for server/CI environments
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np

# Reuse existing parsing and visualization utilities
try:
    from visualization.processor import (
        parse_gnn_content,
        generate_matrix_visualizations,
        generate_network_visualizations,
        generate_combined_analysis,
    )
    VIS_PROCESSOR_AVAILABLE = True
except Exception:
    VIS_PROCESSOR_AVAILABLE = False


class AdvancedVisualizer:
    """
    Real advanced visualizer that composes multiple visualization backends
    to generate a comprehensive set of artifacts per GNN file.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("advanced_visualization")

    def generate_visualizations(
        self,
        content: str,
        model_name: str,
        output_dir: Path,
        viz_type: str = "all",
        interactive: bool = True,
        export_formats: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate advanced visualizations from raw GNN content.

        Returns a list of generated file paths (strings).
        """
        export_formats = export_formats or ["html", "json"]

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        generated: List[str] = []

        if not VIS_PROCESSOR_AVAILABLE:
            return generated

        # Parse content using visualization.processor's parser
        parsed_data = parse_gnn_content(content)

        # Matrix visualizations
        try:
            matrix_files = generate_matrix_visualizations(parsed_data, model_output_dir, model_name)
            generated.extend(matrix_files)
        except Exception as e:
            self.logger.warning(f"Matrix visualizations failed for {model_name}: {e}")

        # Network visualizations
        try:
            network_files = generate_network_visualizations(parsed_data, model_output_dir, model_name)
            generated.extend(network_files)
        except Exception as e:
            self.logger.warning(f"Network visualizations failed for {model_name}: {e}")

        # Combined analysis
        try:
            combined_files = generate_combined_analysis(parsed_data, model_output_dir, model_name)
            generated.extend(combined_files)
        except Exception as e:
            self.logger.warning(f"Combined analysis failed for {model_name}: {e}")

        # Optional HTML summary page that links artifacts (real, non-interactive)
        if "html" in export_formats:
            try:
                html_path = self._generate_summary_html(model_name, model_output_dir, generated)
                if html_path:
                    generated.append(str(html_path))
            except Exception as e:
                self.logger.warning(f"Summary HTML generation failed for {model_name}: {e}")

        # Optional JSON manifest of generated files
        if "json" in export_formats:
            try:
                manifest = {
                    "model": model_name,
                    "generated": generated,
                    "timestamp": datetime.now().isoformat(),
                }
                manifest_path = model_output_dir / f"{model_name}_advanced_viz_manifest.json"
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                generated.append(str(manifest_path))
            except Exception as e:
                self.logger.warning(f"Manifest JSON write failed for {model_name}: {e}")

        return generated

    def _generate_summary_html(self, model_name: str, model_output_dir: Path, files: List[str]) -> Optional[Path]:
        """Generate a simple HTML page linking to produced artifacts."""
        try:
            rel_files = [Path(f) for f in files]
            # Build HTML content with embedded previews for PNGs
            items = []
            for f in rel_files:
                name = f.name
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg'}:
                    items.append(f"<div class='item'><h4>{name}</h4><img src='{name}' style='max-width:100%'></div>")
                else:
                    items.append(f"<div class='item'><a href='{name}' target='_blank'>{name}</a></div>")
            html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{model_name} Advanced Visualizations</title>
<style>body{{font-family:Arial,sans-serif;padding:20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}.item{{background:#f8f9fa;padding:10px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}h2{{margin:0 0 10px 0}}</style>
</head><body>
<h2>Advanced Visualizations: {model_name}</h2>
<div class='grid'>
{''.join(items)}
</div>
</body></html>
"""
            out = model_output_dir / f"{model_name}_advanced_summary.html"
            with open(out, 'w', encoding='utf-8') as f:
                f.write(html)
            return out
        except Exception:
            return None

def create_visualization_from_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a visualization from data."""
    try:
        viz_type = data.get("type", "default")
        
        if viz_type == "network":
            return create_network_visualization(data)
        elif viz_type == "timeline":
            return create_timeline_visualization(data)
        elif viz_type == "heatmap":
            return create_heatmap_visualization(data)
        else:
            return create_default_visualization(data)
            
    except Exception as e:
        return None

def create_dashboard_section(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a dashboard section from data."""
    try:
        section = {
            "title": data.get("title", "Section"),
            "type": data.get("type", "text"),
            "content": data.get("content", ""),
            "metrics": data.get("metrics", {})
        }
        
        return section
        
    except Exception as e:
        return None

def create_network_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a network visualization."""
    try:
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # Create network visualization data
        viz_data = {
            "type": "network",
            "nodes": nodes,
            "edges": edges,
            "layout": "force_directed",
            "options": {
                "node_size": 10,
                "edge_width": 1,
                "node_color": "blue",
                "edge_color": "gray"
            }
        }
        
        return viz_data
        
    except Exception as e:
        return {"error": str(e)}

def create_timeline_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a timeline visualization."""
    try:
        events = data.get("events", [])
        
        # Create timeline visualization data
        viz_data = {
            "type": "timeline",
            "events": events,
            "options": {
                "height": 400,
                "width": 800,
                "show_labels": True
            }
        }
        
        return viz_data
        
    except Exception as e:
        return {"error": str(e)}

def create_heatmap_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a heatmap visualization."""
    try:
        matrix = data.get("matrix", [])
        
        # Create heatmap visualization data
        viz_data = {
            "type": "heatmap",
            "matrix": matrix,
            "options": {
                "colormap": "viridis",
                "show_values": True,
                "aspect_ratio": "auto"
            }
        }
        
        return viz_data
        
    except Exception as e:
        return {"error": str(e)}

def create_default_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a default visualization."""
    try:
        # Create a simple chart visualization
        viz_data = {
            "type": "chart",
            "data": data,
            "options": {
                "chart_type": "line",
                "title": "GNN Analysis",
                "x_label": "Time",
                "y_label": "Value"
            }
        }
        
        return viz_data
        
    except Exception as e:
        return {"error": str(e)} 