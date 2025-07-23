#!/usr/bin/env python3
"""
Step 7: Multi-format Export Generation

This step generates exports in multiple formats (JSON, XML, GraphML, GEXF, Pickle).
"""

import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def export_to_json(model_data: Dict, output_path: Path) -> bool:
    """Export model to JSON format."""
    try:
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"JSON export failed: {e}")
        return False

def export_to_xml(model_data: Dict, output_path: Path) -> bool:
    """Export model to XML format."""
    try:
        # Create root element
        root = ET.Element("gnn_model")
        root.set("name", model_data.get("model_name", "Unknown"))
        root.set("version", model_data.get("model_version", "1.0"))
        root.set("exported", datetime.now().isoformat())
        
        # Add variables
        variables_elem = ET.SubElement(root, "variables")
        for var in model_data.get("variables", []):
            var_elem = ET.SubElement(variables_elem, "variable")
            var_elem.set("name", var.get("name", ""))
            var_elem.set("type", var.get("type", ""))
            var_elem.set("data_type", var.get("data_type", ""))
            var_elem.set("dimensions", str(var.get("dimensions", [])))
            if var.get("description"):
                desc_elem = ET.SubElement(var_elem, "description")
                desc_elem.text = var["description"]
        
        # Add connections
        connections_elem = ET.SubElement(root, "connections")
        for conn in model_data.get("connections", []):
            conn_elem = ET.SubElement(connections_elem, "connection")
            conn_elem.set("type", conn.get("type", ""))
            source_elem = ET.SubElement(conn_elem, "source")
            source_elem.text = str(conn.get("source", []))
            target_elem = ET.SubElement(conn_elem, "target")
            target_elem.text = str(conn.get("target", []))
            if conn.get("description"):
                desc_elem = ET.SubElement(conn_elem, "description")
                desc_elem.text = conn["description"]
        
        # Add parameters
        parameters_elem = ET.SubElement(root, "parameters")
        for param in model_data.get("parameters", []):
            param_elem = ET.SubElement(parameters_elem, "parameter")
            param_elem.set("name", param.get("name", ""))
            param_elem.set("value_type", param.get("value_type", ""))
            if param.get("description"):
                desc_elem = ET.SubElement(param_elem, "description")
                desc_elem.text = param["description"]
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"XML export failed: {e}")
        return False

def export_to_graphml(model_data: Dict, output_path: Path) -> bool:
    """Export model to GraphML format."""
    try:
        # Create GraphML structure
        graphml = ET.Element("graphml")
        graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        
        # Add key definitions
        key_node_type = ET.SubElement(graphml, "key")
        key_node_type.set("id", "type")
        key_node_type.set("for", "node")
        key_node_type.set("attr.name", "type")
        key_node_type.set("attr.type", "string")
        
        key_node_dim = ET.SubElement(graphml, "key")
        key_node_dim.set("id", "dimensions")
        key_node_dim.set("for", "node")
        key_node_dim.set("attr.name", "dimensions")
        key_node_dim.set("attr.type", "string")
        
        key_edge_type = ET.SubElement(graphml, "key")
        key_edge_type.set("id", "connection_type")
        key_edge_type.set("for", "edge")
        key_edge_type.set("attr.name", "connection_type")
        key_edge_type.set("attr.type", "string")
        
        # Create graph
        graph = ET.SubElement(graphml, "graph")
        graph.set("id", model_data.get("model_name", "gnn_model"))
        graph.set("edgedefault", "directed")
        
        # Add nodes (variables)
        var_names = set()
        for var in model_data.get("variables", []):
            var_name = var.get("name", "")
            var_names.add(var_name)
            
            node = ET.SubElement(graph, "node")
            node.set("id", var_name)
            
            data_type = ET.SubElement(node, "data")
            data_type.set("key", "type")
            data_type.text = var.get("type", "")
            
            data_dim = ET.SubElement(node, "data")
            data_dim.set("key", "dimensions")
            data_dim.text = str(var.get("dimensions", []))
        
        # Add edges (connections)
        edge_id = 0
        for conn in model_data.get("connections", []):
            sources = conn.get("source", [])
            targets = conn.get("target", [])
            
            for source in sources:
                for target in targets:
                    if source in var_names and target in var_names:
                        edge = ET.SubElement(graph, "edge")
                        edge.set("id", f"e{edge_id}")
                        edge.set("source", source)
                        edge.set("target", target)
                        
                        data_type = ET.SubElement(edge, "data")
                        data_type.set("key", "connection_type")
                        data_type.text = conn.get("type", "")
                        
                        edge_id += 1
        
        # Write GraphML
        tree = ET.ElementTree(graphml)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"GraphML export failed: {e}")
        return False

def export_to_gexf(model_data: Dict, output_path: Path) -> bool:
    """Export model to GEXF format."""
    try:
        # Create GEXF structure
        gexf = ET.Element("gexf")
        gexf.set("xmlns", "http://www.gexf.net/1.3")
        gexf.set("version", "1.3")
        
        # Add meta information
        meta = ET.SubElement(gexf, "meta")
        meta.set("lastmodifieddate", datetime.now().strftime("%Y-%m-%d"))
        creator = ET.SubElement(meta, "creator")
        creator.text = "GNN Pipeline"
        description = ET.SubElement(meta, "description")
        description.text = f"GNN Model: {model_data.get('model_name', 'Unknown')}"
        
        # Create graph
        graph = ET.SubElement(gexf, "graph")
        graph.set("mode", "static")
        graph.set("defaultedgetype", "directed")
        
        # Add attributes
        attributes = ET.SubElement(graph, "attributes")
        attributes.set("class", "node")
        
        attr_type = ET.SubElement(attributes, "attribute")
        attr_type.set("id", "type")
        attr_type.set("title", "Variable Type")
        attr_type.set("type", "string")
        
        attr_dim = ET.SubElement(attributes, "attribute")
        attr_dim.set("id", "dimensions")
        attr_dim.set("title", "Dimensions")
        attr_dim.set("type", "string")
        
        # Add nodes
        nodes = ET.SubElement(graph, "nodes")
        var_names = set()
        for var in model_data.get("variables", []):
            var_name = var.get("name", "")
            var_names.add(var_name)
            
            node = ET.SubElement(nodes, "node")
            node.set("id", var_name)
            node.set("label", var_name)
            
            attvalues = ET.SubElement(node, "attvalues")
            
            attvalue_type = ET.SubElement(attvalues, "attvalue")
            attvalue_type.set("for", "type")
            attvalue_type.set("value", var.get("type", ""))
            
            attvalue_dim = ET.SubElement(attvalues, "attvalue")
            attvalue_dim.set("for", "dimensions")
            attvalue_dim.set("value", str(var.get("dimensions", [])))
        
        # Add edges
        edges = ET.SubElement(graph, "edges")
        edge_id = 0
        for conn in model_data.get("connections", []):
            sources = conn.get("source", [])
            targets = conn.get("target", [])
            
            for source in sources:
                for target in targets:
                    if source in var_names and target in var_names:
                        edge = ET.SubElement(edges, "edge")
                        edge.set("id", f"e{edge_id}")
                        edge.set("source", source)
                        edge.set("target", target)
                        edge.set("label", f"{source}->{target}")
                        
                        edge_id += 1
        
        # Write GEXF
        tree = ET.ElementTree(gexf)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"GEXF export failed: {e}")
        return False

def export_to_pickle(model_data: Dict, output_path: Path) -> bool:
    """Export model to Pickle format."""
    try:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        return True
    except Exception as e:
        print(f"Pickle export failed: {e}")
        return False

def main():
    """Main export generation function."""
    args = EnhancedArgumentParser.parse_step_arguments("7_export.py")
    
    # Setup logging
    logger = setup_step_logging("export", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("7_export.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Generating multi-format exports")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", config.base_output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return 1
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Export results
        export_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "files_exported": [],
            "summary": {
                "total_files": 0,
                "successful_exports": 0,
                "failed_exports": 0,
                "formats_generated": {
                    "json": 0,
                    "xml": 0,
                    "graphml": 0,
                    "gexf": 0,
                    "pickle": 0
                }
            }
        }
        
        # Export formats
        export_formats = [
            ("json", export_to_json),
            ("xml", export_to_xml),
            ("graphml", export_to_graphml),
            ("gexf", export_to_gexf),
            ("pickle", export_to_pickle)
        ]
        
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Exporting: {file_name}")
            
            # Create file-specific output directory
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            file_export_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "exports": {},
                "success": True
            }
            
            # Export to each format
            for format_name, export_func in export_formats:
                try:
                    output_path = file_output_dir / f"{file_name.replace('.md', '')}.{format_name}"
                    success = export_func(file_result, output_path)
                    
                    file_export_result["exports"][format_name] = {
                        "output_path": str(output_path),
                        "success": success,
                        "file_size": output_path.stat().st_size if success else 0
                    }
                    
                    if success:
                        export_results["summary"]["formats_generated"][format_name] += 1
                        logger.info(f"  Exported {format_name}: {output_path.stat().st_size} bytes")
                    else:
                        file_export_result["success"] = False
                        logger.warning(f"  Failed to export {format_name}")
                        
                except Exception as e:
                    file_export_result["exports"][format_name] = {
                        "output_path": str(file_output_dir / f"{file_name.replace('.md', '')}.{format_name}"),
                        "success": False,
                        "error": str(e)
                    }
                    file_export_result["success"] = False
                    logger.error(f"  Export {format_name} failed: {e}")
            
            export_results["files_exported"].append(file_export_result)
            
            # Update summary
            export_results["summary"]["total_files"] += 1
            if file_export_result["success"]:
                export_results["summary"]["successful_exports"] += 1
            else:
                export_results["summary"]["failed_exports"] += 1
        
        # Save export results
        results_file = output_dir / "export_results.json"
        with open(results_file, 'w') as f:
            json.dump(export_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "export_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(export_results["summary"], f, indent=2)
        
        # Determine success
        success = export_results["summary"]["successful_exports"] > 0
        
        if success:
            total_formats = sum(export_results["summary"]["formats_generated"].values())
            log_step_success(logger, f"Exported {export_results['summary']['successful_exports']} files in {total_formats} format instances")
            return 0
        else:
            log_step_error(logger, "Export generation failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Export generation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
