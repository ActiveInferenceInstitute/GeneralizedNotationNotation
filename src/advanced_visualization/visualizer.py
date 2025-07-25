"""
Main visualizer class for advanced GNN visualizations.

This module provides the main AdvancedVisualizer class that coordinates
data extraction and visualization generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .data_extractor import VisualizationDataExtractor
from .html_generator import HTMLVisualizationGenerator


class AdvancedVisualizer:
    """
    Main class for generating advanced visualizations of GNN models.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the advanced visualizer.
        
        Args:
            strict_validation: Whether to use strict validation during parsing
        """
        self.data_extractor = VisualizationDataExtractor(strict_validation=strict_validation)
        self.html_generator = HTMLVisualizationGenerator()
    
    def generate_visualizations(self, content: str, model_name: str, output_dir: Path, 
                               viz_type: str = "all", interactive: bool = True, 
                               export_formats: List[str] = None) -> List[str]:
        """
        Generate advanced visualizations for a GNN model.
        
        Args:
            content: GNN file content as string
            model_name: Name of the model
            output_dir: Output directory for generated files
            viz_type: Type of visualization to generate
            interactive: Whether to generate interactive visualizations
            export_formats: List of export formats
            
        Returns:
            List of generated file paths
        """
        if export_formats is None:
            export_formats = ["html", "json"]
        
        generated_files = []
        
        try:
            # Create model-specific output directory
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract visualization data
            extracted_data = self.data_extractor.extract_from_content(content)
            
            # Add statistics to the extracted data
            if extracted_data.get("success", False):
                statistics = self.data_extractor.get_model_statistics(extracted_data)
                extracted_data["statistics"] = statistics
            
            # Generate JSON data file
            if "json" in export_formats:
                json_file = model_output_dir / f"{model_name}_viz_data.json"
                with open(json_file, 'w') as f:
                    json.dump(extracted_data, f, indent=2, default=str)
                generated_files.append(str(json_file))
            
            # Generate HTML visualization
            if "html" in export_formats:
                html_content = self.html_generator.generate_advanced_visualization(
                    extracted_data, model_name
                )
                html_file = model_output_dir / f"{model_name}_advanced_viz.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                generated_files.append(str(html_file))
            
            # Generate additional format-specific visualizations
            if viz_type in ["all", "3d"] and "html" in export_formats:
                # Generate 3D visualization if requested
                html_3d_content = self._generate_3d_visualization(extracted_data, model_name)
                html_3d_file = model_output_dir / f"{model_name}_3d_viz.html"
                with open(html_3d_file, 'w', encoding='utf-8') as f:
                    f.write(html_3d_content)
                generated_files.append(str(html_3d_file))
            
            if viz_type in ["all", "interactive"] and interactive and "html" in export_formats:
                # Generate interactive visualization if requested
                html_interactive_content = self._generate_interactive_visualization(
                    extracted_data, model_name
                )
                html_interactive_file = model_output_dir / f"{model_name}_interactive_viz.html"
                with open(html_interactive_file, 'w', encoding='utf-8') as f:
                    f.write(html_interactive_content)
                generated_files.append(str(html_interactive_file))
            
        except Exception as e:
            # Generate error visualization
            error_html = self.html_generator._generate_error_page(model_name, [str(e)])
            error_file = output_dir / f"{model_name}_error_viz.html"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(error_html)
            generated_files.append(str(error_file))
        
        return generated_files
    
    def _generate_3d_visualization(self, extracted_data: Dict[str, Any], model_name: str) -> str:
        """
        Generate a 3D visualization of the GNN model.
        
        Args:
            extracted_data: Extracted visualization data
            model_name: Name of the model
            
        Returns:
            HTML content for 3D visualization
        """
        if not extracted_data.get("success", False):
            return self.html_generator._generate_error_page(model_name, ["3D visualization not available"])
        
        blocks = extracted_data.get("blocks", [])
        connections = extracted_data.get("connections", [])
        
        # Simple 3D visualization using CSS transforms
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D GNN Visualization - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
        }}
        
        .header {{
            color: white;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .scene {{
            perspective: 1000px;
            height: 600px;
            position: relative;
            margin: 40px 0;
        }}
        
        .scene-container {{
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            animation: rotate 20s infinite linear;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotateY(0deg) rotateX(20deg); }}
            to {{ transform: rotateY(360deg) rotateX(20deg); }}
        }}
        
        .node {{
            position: absolute;
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.8em;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .node:hover {{
            transform: scale(1.2);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        }}
        
        .connection-line {{
            position: absolute;
            height: 2px;
            background: linear-gradient(90deg, #a29bfe 0%, #6c5ce7 100%);
            transform-origin: left center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
        
        .controls {{
            margin: 20px 0;
            color: white;
        }}
        
        .control-btn {{
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 25px;
            cursor: pointer;
            transition: background 0.3s ease;
        }}
        
        .control-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
        
        .info-panel {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            margin-top: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .info-panel h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            color: white;
            border-radius: 10px;
        }}
        
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 3D GNN Visualization</h1>
            <h2>Model: {model_name}</h2>
        </div>
        
        <div class="controls">
            <button class="control-btn" onclick="toggleRotation()">Pause/Rotate</button>
            <button class="control-btn" onclick="resetView()">Reset View</button>
        </div>
        
        <div class="scene">
            <div class="scene-container" id="scene">
"""
        
        # Position nodes in 3D space
        import math
        radius = 200
        for i, block in enumerate(blocks):
            angle = (2 * math.pi * i) / len(blocks) if blocks else 0
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = (i % 3 - 1) * 50  # Vary height slightly
            
            html += f"""
                <div class="node" style="transform: translate3d({x}px, {y}px, {z}px);" 
                     title="{block.get('name', 'Unknown')} - {block.get('type', 'Unknown')}">
                    {block.get('name', 'Unknown')[:8]}
                </div>
"""
        
        # Add connection lines (simplified - just show some connections)
        for i, conn in enumerate(connections[:min(10, len(connections))]):  # Limit to first 10 connections
            if conn.get('from') and conn.get('to'):
                html += f"""
                <div class="connection-line" style="
                    width: {radius * 0.8}px;
                    transform: translate3d({radius * 0.1}px, 0, 0) rotateY({i * 36}deg);
                "></div>
"""
        
        html += """
            </div>
        </div>
        
        <div class="info-panel">
            <h3>📊 Model Statistics</h3>
            <div class="stats-grid">
"""
        
        # Add statistics
        stats = [
            ("Variables", len(blocks)),
            ("Connections", len(connections)),
            ("Parameters", extracted_data.get("total_parameters", 0)),
            ("Equations", extracted_data.get("total_equations", 0))
        ]
        
        for label, value in stats:
            html += f"""
                <div class="stat-item">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
"""
        
        html += """
            </div>
        </div>
    </div>
    
    <script>
        let isRotating = true;
        const scene = document.getElementById('scene');
        
        function toggleRotation() {{
            isRotating = !isRotating;
            if (isRotating) {{
                scene.style.animationPlayState = 'running';
            }} else {{
                scene.style.animationPlayState = 'paused';
            }}
        }}
        
        function resetView() {{
            scene.style.transform = 'rotateY(0deg) rotateX(20deg)';
        }}
        
        // Add click handlers for nodes
        document.querySelectorAll('.node').forEach(node => {{
            node.addEventListener('click', function() {{
                alert('Node: ' + this.title);
            }});
        }});
    </script>
</body>
</html>
"""
        
        return html
    
    def _generate_interactive_visualization(self, extracted_data: Dict[str, Any], model_name: str) -> str:
        """
        Generate an interactive visualization of the GNN model.
        
        Args:
            extracted_data: Extracted visualization data
            model_name: Name of the model
            
        Returns:
            HTML content for interactive visualization
        """
        if not extracted_data.get("success", False):
            return self.html_generator._generate_error_page(model_name, ["Interactive visualization not available"])
        
        blocks = extracted_data.get("blocks", [])
        connections = extracted_data.get("connections", [])
        
        # Interactive visualization with JavaScript
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive GNN Visualization - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .visualization-area {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .control-btn {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            font-weight: bold;
        }}
        
        .control-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }}
        
        .control-btn.active {{
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        }}
        
        .graph-container {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            height: 500px;
            position: relative;
            overflow: hidden;
            background: #f8f9fa;
        }}
        
        .node {{
            position: absolute;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.7em;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .node:hover {{
            transform: scale(1.1);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}
        
        .node.selected {{
            transform: scale(1.2);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        }}
        
        .connection {{
            position: absolute;
            height: 3px;
            background: linear-gradient(90deg, #a29bfe 0%, #6c5ce7 100%);
            transform-origin: left center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
        
        .info-panel {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .info-panel h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .node-info {{
            display: none;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }}
        
        .node-info.show {{
            display: block;
        }}
        
        .filter-panel {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .filter-group {{
            display: inline-block;
            margin: 0 15px;
        }}
        
        .filter-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .filter-group select {{
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ddd;
            background: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Interactive GNN Visualization</h1>
            <h2>Model: {model_name}</h2>
        </div>
        
        <div class="visualization-area">
            <div class="controls">
                <button class="control-btn" onclick="toggleLayout()">Toggle Layout</button>
                <button class="control-btn" onclick="resetView()">Reset View</button>
                <button class="control-btn" onclick="toggleConnections()">Toggle Connections</button>
                <button class="control-btn" onclick="highlightSelected()">Highlight Selected</button>
            </div>
            
            <div class="filter-panel">
                <div class="filter-group">
                    <label>Variable Type:</label>
                    <select id="typeFilter" onchange="filterNodes()">
                        <option value="">All Types</option>
                        <option value="hidden_state">Hidden State</option>
                        <option value="observation">Observation</option>
                        <option value="action">Action</option>
                        <option value="policy">Policy</option>
                        <option value="likelihood_matrix">Likelihood Matrix</option>
                        <option value="transition_matrix">Transition Matrix</option>
                        <option value="preference_vector">Preference Vector</option>
                        <option value="prior_vector">Prior Vector</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Data Type:</label>
                    <select id="dataTypeFilter" onchange="filterNodes()">
                        <option value="">All Data Types</option>
                        <option value="categorical">Categorical</option>
                        <option value="continuous">Continuous</option>
                        <option value="binary">Binary</option>
                        <option value="integer">Integer</option>
                        <option value="float">Float</option>
                    </select>
                </div>
            </div>
            
            <div class="graph-container" id="graphContainer">
"""
        
        # Add nodes and connections with JavaScript positioning
        html += f"""
            </div>
        </div>
        
        <div class="info-panel">
            <h3>📋 Node Information</h3>
            <div class="node-info" id="nodeInfo">
                <p><strong>Click on a node to see details</strong></p>
            </div>
            
            <h3 style="margin-top: 30px;">📊 Model Overview</h3>
            <p><strong>Total Variables:</strong> {len(blocks)}</p>
            <p><strong>Total Connections:</strong> {len(connections)}</p>
            <p><strong>Total Parameters:</strong> {extracted_data.get("total_parameters", 0)}</p>
            <p><strong>Total Equations:</strong> {extracted_data.get("total_equations", 0)}</p>
        </div>
    </div>
    
    <script>
        // Model data
        const modelData = {{
            nodes: {json.dumps(blocks)},
            connections: {json.dumps(connections)},
            modelName: "{model_name}"
        }};
        
        let currentLayout = 'circular';
        let showConnections = true;
        let selectedNodes = [];
        
        // Initialize visualization
        function initializeVisualization() {{
            createNodes();
            createConnections();
        }}
        
        function createNodes() {{
            const container = document.getElementById('graphContainer');
            const nodes = modelData.nodes;
            
            nodes.forEach((node, index) => {{
                const nodeElement = document.createElement('div');
                nodeElement.className = 'node';
                nodeElement.id = `node-${{index}}`;
                nodeElement.setAttribute('data-type', node.type);
                nodeElement.setAttribute('data-data-type', node.data_type);
                nodeElement.innerHTML = node.name.substring(0, 8);
                nodeElement.title = `${{node.name}} - ${{node.type}}`;
                
                // Position nodes in circular layout
                const angle = (2 * Math.PI * index) / nodes.length;
                const radius = 150;
                const x = 250 + radius * Math.cos(angle);
                const y = 250 + radius * Math.sin(angle);
                
                nodeElement.style.left = x + 'px';
                nodeElement.style.top = y + 'px';
                nodeElement.style.backgroundColor = getNodeColor(node.type);
                
                nodeElement.addEventListener('click', () => showNodeInfo(node, index));
                
                container.appendChild(nodeElement);
            }});
        }}
        
        function createConnections() {{
            const container = document.getElementById('graphContainer');
            const connections = modelData.connections;
            
            connections.forEach((conn, index) => {{
                if (conn.from && conn.to && conn.from.length > 0 && conn.to.length > 0) {{
                    const connectionElement = document.createElement('div');
                    connectionElement.className = 'connection';
                    connectionElement.id = `connection-${{index}}`;
                    
                    // Simple connection visualization
                    connectionElement.style.width = '100px';
                    connectionElement.style.left = '200px';
                    connectionElement.style.top = '200px';
                    connectionElement.style.transform = `rotate(${{index * 30}}deg)`;
                    
                    container.appendChild(connectionElement);
                }}
            }});
        }}
        
        function getNodeColor(type) {{
            const colors = {{
                'hidden_state': 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
                'observation': 'linear-gradient(135deg, #00b894 0%, #00a085 100%)',
                'action': 'linear-gradient(135deg, #fd79a8 0%, #e84393 100%)',
                'policy': 'linear-gradient(135deg, #fdcb6e 0%, #e17055 100%)',
                'likelihood_matrix': 'linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%)',
                'transition_matrix': 'linear-gradient(135deg, #fd79a8 0%, #e84393 100%)',
                'preference_vector': 'linear-gradient(135deg, #00b894 0%, #00a085 100%)',
                'prior_vector': 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)'
            }};
            return colors[type] || 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)';
        }}
        
        function showNodeInfo(node, index) {{
            const nodeInfo = document.getElementById('nodeInfo');
            nodeInfo.innerHTML = `
                <h4>${{node.name}}</h4>
                <p><strong>Type:</strong> ${{node.type}}</p>
                <p><strong>Data Type:</strong> ${{node.data_type}}</p>
                <p><strong>Dimensions:</strong> ${{JSON.stringify(node.dimensions)}}</p>
                ${{node.description ? `<p><strong>Description:</strong> ${{node.description}}</p>` : ''}}
            `;
            nodeInfo.classList.add('show');
            
            // Highlight selected node
            document.querySelectorAll('.node').forEach(n => n.classList.remove('selected'));
            document.getElementById(`node-${{index}}`).classList.add('selected');
        }}
        
        function toggleLayout() {{
            currentLayout = currentLayout === 'circular' ? 'grid' : 'circular';
            applyLayout();
        }}
        
        function applyLayout() {{
            const nodes = document.querySelectorAll('.node');
            const container = document.getElementById('graphContainer');
            
            if (currentLayout === 'grid') {{
                const cols = Math.ceil(Math.sqrt(nodes.length));
                nodes.forEach((node, index) => {{
                    const row = Math.floor(index / cols);
                    const col = index % cols;
                    const x = 50 + col * 120;
                    const y = 50 + row * 120;
                    node.style.left = x + 'px';
                    node.style.top = y + 'px';
                }});
            }} else {{
                nodes.forEach((node, index) => {{
                    const angle = (2 * Math.PI * index) / nodes.length;
                    const radius = 150;
                    const x = 250 + radius * Math.cos(angle);
                    const y = 250 + radius * Math.sin(angle);
                    node.style.left = x + 'px';
                    node.style.top = y + 'px';
                }});
            }}
        }}
        
        function resetView() {{
            currentLayout = 'circular';
            applyLayout();
            document.querySelectorAll('.node').forEach(n => n.classList.remove('selected'));
            document.getElementById('nodeInfo').classList.remove('show');
        }}
        
        function toggleConnections() {{
            showConnections = !showConnections;
            document.querySelectorAll('.connection').forEach(conn => {{
                conn.style.display = showConnections ? 'block' : 'none';
            }});
        }}
        
        function highlightSelected() {{
            const selected = document.querySelector('.node.selected');
            if (selected) {{
                selected.style.transform = 'scale(1.5)';
                setTimeout(() => {{
                    selected.style.transform = 'scale(1.2)';
                }}, 1000);
            }}
        }}
        
        function filterNodes() {{
            const typeFilter = document.getElementById('typeFilter').value;
            const dataTypeFilter = document.getElementById('dataTypeFilter').value;
            
            document.querySelectorAll('.node').forEach((node, index) => {{
                const nodeData = modelData.nodes[index];
                const typeMatch = !typeFilter || nodeData.type === typeFilter;
                const dataTypeMatch = !dataTypeFilter || nodeData.data_type === dataTypeFilter;
                
                if (typeMatch && dataTypeMatch) {{
                    node.style.display = 'block';
                    node.style.opacity = '1';
                }} else {{
                    node.style.display = 'none';
                    node.style.opacity = '0.3';
                }}
            }});
        }}
        
        // Initialize when page loads
        window.addEventListener('load', initializeVisualization);
    </script>
</body>
</html>
"""
        
        return html


def process_advanced_visualization(target_dir, output_dir, **kwargs):
    """
    Process advanced visualizations for GNN files in the target directory.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory to save visualization results
        **kwargs: Additional arguments including viz_type, interactive, export_formats
        
    Returns:
        Dictionary with visualization results
    """
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    
    viz_type = kwargs.get('viz_type', 'all')
    interactive = kwargs.get('interactive', True)
    export_formats = kwargs.get('export_formats', ['html', 'json'])
    
    visualizer = AdvancedVisualizer(strict_validation=False)
    
    results = {
        "processed_files": 0,
        "successful_visualizations": 0,
        "failed_visualizations": 0,
        "generated_files": [],
        "errors": []
    }
    
    # Find all GNN files
    gnn_extensions = ['.md', '.gnn', '.json', '.yaml', '.yml']
    gnn_files = []
    
    for ext in gnn_extensions:
        gnn_files.extend(target_dir.glob(f"**/*{ext}"))
    
    for gnn_file in gnn_files:
        try:
            with open(gnn_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            model_name = gnn_file.stem
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            generated_files = visualizer.generate_visualizations(
                content=content,
                model_name=model_name,
                output_dir=model_output_dir,
                viz_type=viz_type,
                interactive=interactive,
                export_formats=export_formats
            )
            
            results["processed_files"] += 1
            results["successful_visualizations"] += 1
            results["generated_files"].extend([str(f) for f in generated_files])
            
        except Exception as e:
            results["processed_files"] += 1
            results["failed_visualizations"] += 1
            results["errors"].append(f"Failed to visualize {gnn_file}: {e}")
    
    # Save summary
    summary_file = output_dir / "advanced_visualization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results 