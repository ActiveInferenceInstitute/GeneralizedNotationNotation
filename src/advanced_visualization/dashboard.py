"""
Dashboard generator for comprehensive GNN model dashboards.

This module provides functionality to generate comprehensive dashboards
that combine multiple visualization types and interactive features.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .data_extractor import VisualizationDataExtractor
from .html_generator import HTMLVisualizationGenerator


class DashboardGenerator:
    """
    Generates comprehensive dashboards for GNN models.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the dashboard generator.
        
        Args:
            strict_validation: Whether to use strict validation during parsing
        """
        self.data_extractor = VisualizationDataExtractor(strict_validation=strict_validation)
        self.html_generator = HTMLVisualizationGenerator()
    
    def generate_dashboard(self, content: str, model_name: str, output_dir: Path) -> Optional[Path]:
        """
        Generate a comprehensive dashboard for a GNN model.
        
        Args:
            content: GNN file content as string
            model_name: Name of the model
            output_dir: Output directory for generated files
            
        Returns:
            Path to the generated dashboard file, or None if generation failed
        """
        try:
            # Create model-specific output directory
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract visualization data
            extracted_data = self.data_extractor.extract_from_content(content)
            
            if not extracted_data.get("success", False):
                return None
            
            # Add statistics to the extracted data
            statistics = self.data_extractor.get_model_statistics(extracted_data)
            extracted_data["statistics"] = statistics
            
            # Generate dashboard HTML
            dashboard_html = self._generate_dashboard_html(extracted_data, model_name)
            
            # Save dashboard file
            dashboard_file = model_output_dir / f"{model_name}_dashboard.html"
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            return dashboard_file
            
        except Exception as e:
            print(f"Failed to generate dashboard for {model_name}: {e}")
            return None
    
    def _generate_dashboard_html(self, extracted_data: Dict[str, Any], model_name: str) -> str:
        """
        Generate comprehensive dashboard HTML.
        
        Args:
            extracted_data: Extracted visualization data
            model_name: Name of the model
            
        Returns:
            HTML content for the dashboard
        """
        blocks = extracted_data.get("blocks", [])
        connections = extracted_data.get("connections", [])
        parameters = extracted_data.get("parameters", [])
        equations = extracted_data.get("equations", [])
        model_info = extracted_data.get("model_info", {})
        statistics = extracted_data.get("statistics", {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Dashboard - {model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .dashboard {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header h2 {{
            color: #7f8c8d;
            font-size: 1.3em;
            font-weight: 300;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .main-content {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .card h3 {{
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 20px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .model-variables {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .variable-item {{
            background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
            color: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .variable-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .variable-details {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .connection-item {{
            background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
            color: white;
            margin: 8px 0;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9em;
        }}
        
        .parameter-item {{
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            margin: 8px 0;
            padding: 12px;
            border-radius: 8px;
        }}
        
        .parameter-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .parameter-value {{
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            background: rgba(255, 255, 255, 0.2);
            padding: 5px;
            border-radius: 5px;
            word-break: break-all;
        }}
        
        .equation-item {{
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            color: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
        }}
        
        .equation-label {{
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .equation-content {{
            font-family: 'Courier New', monospace;
            background: rgba(255, 255, 255, 0.2);
            padding: 8px;
            border-radius: 5px;
            font-size: 0.9em;
            white-space: pre-wrap;
        }}
        
        .model-info {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        
        .info-item {{
            margin: 8px 0;
        }}
        
        .info-label {{
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }}
        
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .tab.active {{
            border-bottom-color: #3498db;
            background: rgba(52, 152, 219, 0.1);
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            color: #7f8c8d;
        }}
        
        @media (max-width: 1200px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media (max-width: 768px) {{
            .dashboard {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>📊 GNN Model Dashboard</h1>
            <h2>{model_name}</h2>
        </div>
        
        <div class="dashboard-grid">
            <div class="main-content">
                <div class="card">
                    <h3>📈 Model Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{len(blocks)}</div>
                            <div class="stat-label">Variables</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{len(connections)}</div>
                            <div class="stat-label">Connections</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{len(parameters)}</div>
                            <div class="stat-label">Parameters</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{len(equations)}</div>
                            <div class="stat-label">Equations</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🧩 Model Components</h3>
                    <div class="tabs">
                        <div class="tab active" onclick="showTab('variables')">Variables</div>
                        <div class="tab" onclick="showTab('connections')">Connections</div>
                        <div class="tab" onclick="showTab('parameters')">Parameters</div>
                        <div class="tab" onclick="showTab('equations')">Equations</div>
                    </div>
                    
                    <div id="variables" class="tab-content active">
                        <div class="model-variables">
"""
        
        # Add variables
        for block in blocks:
            html += f"""
                            <div class="variable-item">
                                <div class="variable-name">{block.get('name', 'Unknown')}</div>
                                <div class="variable-details">
                                    <strong>Type:</strong> {block.get('type', 'Unknown')}<br>
                                    <strong>Data Type:</strong> {block.get('data_type', 'Unknown')}<br>
                                    <strong>Dimensions:</strong> {block.get('dimensions', [])}
                                    {f"<br><strong>Description:</strong> {block.get('description', '')}" if block.get('description') else ''}
                                </div>
                            </div>
"""
        
        html += """
                        </div>
                    </div>
                    
                    <div id="connections" class="tab-content">
"""
        
        # Add connections
        for conn in connections:
            html += f"""
                        <div class="connection-item">
                            <strong>From:</strong> {', '.join(conn.get('from', []))} → 
                            <strong>To:</strong> {', '.join(conn.get('to', []))}<br>
                            <strong>Type:</strong> {conn.get('type', 'Unknown')}
                            {f"<br><strong>Description:</strong> {conn.get('description', '')}" if conn.get('description') else ''}
                        </div>
"""
        
        html += """
                    </div>
                    
                    <div id="parameters" class="tab-content">
"""
        
        # Add parameters
        for param in parameters:
            html += f"""
                        <div class="parameter-item">
                            <div class="parameter-name">{param.get('name', 'Unknown')}</div>
                            <div class="parameter-value">{json.dumps(param.get('value', 'Unknown'), indent=2)}</div>
                            {f"<div style='margin-top: 5px; font-size: 0.8em; opacity: 0.8;'>{param.get('description', '')}</div>" if param.get('description') else ''}
                        </div>
"""
        
        html += """
                    </div>
                    
                    <div id="equations" class="tab-content">
"""
        
        # Add equations
        for eq in equations:
            html += f"""
                        <div class="equation-item">
                            <div class="equation-label">{eq.get('label', 'Equation')}</div>
                            <div class="equation-content">{eq.get('content', '')}</div>
                            {f"<div style='margin-top: 8px; font-size: 0.9em; opacity: 0.8;'>{eq.get('description', '')}</div>" if eq.get('description') else ''}
                        </div>
"""
        
        html += """
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <h3>ℹ️ Model Information</h3>
                    <div class="model-info">
"""
        
        # Add model information
        info_items = [
            ("Name", model_info.get('name', 'Unknown')),
            ("Version", model_info.get('version', 'Unknown')),
            ("Format", model_info.get('source_format', 'Unknown')),
            ("Created", model_info.get('created_at', 'Unknown')[:10] if model_info.get('created_at') else 'Unknown'),
            ("Modified", model_info.get('modified_at', 'Unknown')[:10] if model_info.get('modified_at') else 'Unknown')
        ]
        
        for label, value in info_items:
            html += f"""
                        <div class="info-item">
                            <span class="info-label">{label}:</span>
                            <span>{value}</span>
                        </div>
"""
        
        html += """
                    </div>
                </div>
                
                <div class="card">
                    <h3>📊 Type Distribution</h3>
                    <div class="stats-grid">
"""
        
        # Add type distribution statistics
        type_stats = statistics.get('variable_types', {})
        for var_type, count in type_stats.items():
            html += f"""
                        <div class="stat-item">
                            <div class="stat-value">{count}</div>
                            <div class="stat-label">{var_type.replace('_', ' ').title()}</div>
                        </div>
"""
        
        html += """
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | GNN Dashboard System</p>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
        
        // Add some interactivity
        document.querySelectorAll('.variable-item').forEach(item => {{
            item.addEventListener('click', function() {{
                this.style.transform = 'scale(1.05)';
                setTimeout(() => {{
                    this.style.transform = 'scale(1)';
                }}, 200);
            }});
        }});
    </script>
</body>
</html>
"""
        
        return html 