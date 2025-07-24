"""
HTML visualization generator for GNN models.

This module provides functionality to generate rich HTML visualizations
from extracted GNN model data.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class HTMLVisualizationGenerator:
    """
    Generates rich HTML visualizations from GNN model data.
    """
    
    def __init__(self):
        """Initialize the HTML generator."""
        pass
    
    def generate_advanced_visualization(self, extracted_data: Dict[str, Any], model_name: str) -> str:
        """
        Generate a comprehensive HTML visualization for a GNN model.
        
        Args:
            extracted_data: Data extracted from GNN file
            model_name: Name of the model
            
        Returns:
            HTML content as string
        """
        if not extracted_data.get("success", False):
            return self._generate_error_page(model_name, extracted_data.get("errors", ["Unknown error"]))
        
        blocks = extracted_data.get("blocks", [])
        connections = extracted_data.get("connections", [])
        parameters = extracted_data.get("parameters", [])
        equations = extracted_data.get("equations", [])
        model_info = extracted_data.get("model_info", {})
        statistics = extracted_data.get("statistics", {})
        
        # Generate the HTML content
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced GNN Visualization - {model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .header h2 {{
            color: #7f8c8d;
            font-size: 1.3em;
            font-weight: 300;
        }}
        
        .model-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .info-card {{
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .info-card h4 {{
            color: #34495e;
            margin-bottom: 8px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .info-card p {{
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .section {{
            background: rgba(255, 255, 255, 0.95);
            margin: 30px 0;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .section h3 {{
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .blocks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .block-card {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .block-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
        }}
        
        .block-name {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }}
        
        .block-details {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .block-type {{
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 8px;
            border-radius: 15px;
            font-size: 0.8em;
            display: inline-block;
            margin: 5px 5px 5px 0;
        }}
        
        .connections-list {{
            list-style: none;
        }}
        
        .connection-item {{
            background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
            color: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .connection-header {{
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .connection-details {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .parameters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }}
        
        .parameter-card {{
            background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .parameter-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .parameter-value {{
            font-family: 'Courier New', monospace;
            background: rgba(255, 255, 255, 0.2);
            padding: 5px;
            border-radius: 5px;
            font-size: 0.9em;
            word-break: break-all;
        }}
        
        .equations-list {{
            list-style: none;
        }}
        
        .equation-item {{
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            margin: 15px 0;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .equation-label {{
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .equation-content {{
            font-family: 'Courier New', monospace;
            background: rgba(255, 255, 255, 0.2);
            padding: 10px;
            border-radius: 8px;
            font-size: 0.9em;
            white-space: pre-wrap;
        }}
        
        .statistics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            color: #7f8c8d;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .blocks-grid {{
                grid-template-columns: 1fr;
            }}
            
            .parameters-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Advanced GNN Visualization</h1>
            <h2>Model: {model_name}</h2>
            
            <div class="model-info">
                <div class="info-card">
                    <h4>Model Name</h4>
                    <p>{model_info.get('name', 'Unknown')}</p>
                </div>
                <div class="info-card">
                    <h4>Version</h4>
                    <p>{model_info.get('version', 'Unknown')}</p>
                </div>
                <div class="info-card">
                    <h4>Format</h4>
                    <p>{model_info.get('source_format', 'Unknown')}</p>
                </div>
                <div class="info-card">
                    <h4>Created</h4>
                    <p>{model_info.get('created_at', 'Unknown')[:10] if model_info.get('created_at') else 'Unknown'}</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>📊 Model Statistics</h3>
            <div class="statistics-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(blocks)}</div>
                    <div class="stat-label">Variables</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(connections)}</div>
                    <div class="stat-label">Connections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(parameters)}</div>
                    <div class="stat-label">Parameters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(equations)}</div>
                    <div class="stat-label">Equations</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>🧩 Model Variables</h3>
            <div class="blocks-grid">
"""
        
        # Add variable blocks
        for block in blocks:
            html += f"""
                <div class="block-card">
                    <div class="block-name">{block.get('name', 'Unknown')}</div>
                    <div class="block-details">
                        <span class="block-type">{block.get('type', 'Unknown')}</span>
                        <span class="block-type">{block.get('data_type', 'Unknown')}</span>
                        <br>
                        <strong>Dimensions:</strong> {block.get('dimensions', [])}
                        {f"<br><strong>Description:</strong> {block.get('description', '')}" if block.get('description') else ''}
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h3>🔗 Model Connections</h3>
            <ul class="connections-list">
"""
        
        # Add connections
        for i, conn in enumerate(connections, 1):
            html += f"""
                <li class="connection-item">
                    <div class="connection-header">Connection {i}</div>
                    <div class="connection-details">
                        <strong>From:</strong> {', '.join(conn.get('from', []))}<br>
                        <strong>To:</strong> {', '.join(conn.get('to', []))}<br>
                        <strong>Type:</strong> {conn.get('type', 'Unknown')}
                        {f"<br><strong>Description:</strong> {conn.get('description', '')}" if conn.get('description') else ''}
                    </div>
                </li>
"""
        
        html += """
            </ul>
        </div>
"""
        
        # Add parameters section if there are parameters
        if parameters:
            html += """
        <div class="section">
            <h3>⚙️ Model Parameters</h3>
            <div class="parameters-grid">
"""
            
            for param in parameters:
                html += f"""
                <div class="parameter-card">
                    <div class="parameter-name">{param.get('name', 'Unknown')}</div>
                    <div class="parameter-value">{json.dumps(param.get('value', 'Unknown'), indent=2)}</div>
                    {f"<div style='margin-top: 5px; font-size: 0.8em; opacity: 0.8;'>{param.get('description', '')}</div>" if param.get('description') else ''}
                </div>
"""
            
            html += """
            </div>
        </div>
"""
        
        # Add equations section if there are equations
        if equations:
            html += """
        <div class="section">
            <h3>📐 Model Equations</h3>
            <ul class="equations-list">
"""
            
            for eq in equations:
                html += f"""
                <li class="equation-item">
                    <div class="equation-label">{eq.get('label', 'Equation')}</div>
                    <div class="equation-content">{eq.get('content', '')}</div>
                    {f"<div style='margin-top: 10px; font-size: 0.9em; opacity: 0.8;'>{eq.get('description', '')}</div>" if eq.get('description') else ''}
                </li>
"""
            
            html += """
            </ul>
        </div>
"""
        
        # Add footer
        html += f"""
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | GNN Advanced Visualization System</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_error_page(self, model_name: str, errors: List[str]) -> str:
        """
        Generate an error page when visualization data extraction fails.
        
        Args:
            model_name: Name of the model
            errors: List of error messages
            
        Returns:
            HTML content for error page
        """
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Advanced GNN Visualization - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 20px;
        }}
        
        .error-container {{
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            text-align: center;
            max-width: 600px;
        }}
        
        .error-icon {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
        
        .error-title {{
            color: #c0392b;
            font-size: 2em;
            margin-bottom: 20px;
        }}
        
        .error-message {{
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        
        .error-details {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: left;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-icon">⚠️</div>
        <h1 class="error-title">Visualization Error</h1>
        <p class="error-message">Failed to generate advanced visualization for model: <strong>{model_name}</strong></p>
        <div class="error-details">
            <strong>Errors:</strong><br>
            {chr(10).join(f"• {error}" for error in errors)}
        </div>
    </div>
</body>
</html>
""" 