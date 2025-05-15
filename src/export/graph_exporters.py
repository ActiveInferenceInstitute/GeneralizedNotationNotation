"""
Specialized GNN Exporters for Graph Formats (GEXF, GraphML, JSON Adjacency List)
"""
import logging
from typing import Dict, Any

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False
    logging.getLogger(__name__).warning("NetworkX library not found. Graph export functionalities will be disabled.")

logger = logging.getLogger(__name__)

def _build_networkx_graph(gnn_model: dict) -> 'nx.DiGraph | None':
    """Helper to build a NetworkX graph from the GNN model."""
    if not HAS_NETWORKX:
        logger.error("NetworkX library is not available. Cannot build graph.")
        return None

    graph = nx.DiGraph()
    model_name = gnn_model.get('name', gnn_model.get('metadata', {}).get('name', 'GNN_Model'))
    graph.graph['name'] = model_name

    # Add states as nodes
    for state_data in gnn_model.get('states', []):
        node_id = state_data.get('id')
        if node_id:
            attributes = {k: v for k, v in state_data.items() if k != 'id'}
            graph.add_node(node_id, **attributes)

    # Add observations as nodes (if distinct from states)
    for obs_data in gnn_model.get('observations', []):
        node_id = obs_data.get('id')
        if node_id and not graph.has_node(node_id): 
            attributes = {k: v for k, v in obs_data.items() if k != 'id'}
            graph.add_node(node_id, **attributes)

    # Add transitions as edges
    for trans_data in gnn_model.get('transitions', []):
        source = trans_data.get('source')
        target = trans_data.get('target')
        if source and target:
            attributes = trans_data.get('attributes', {})
            if not graph.has_node(source): graph.add_node(source, label=source)
            if not graph.has_node(target): graph.add_node(target, label=target)
            graph.add_edge(source, target, **attributes)
            
    return graph

def export_to_gexf(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a GEXF file."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX not available, GEXF export failed.")
    graph = _build_networkx_graph(gnn_model)
    if graph is None: return # Error logged in _build_networkx_graph
    try:
        nx.write_gexf(graph, output_file_path)
        logger.debug(f"Successfully exported GNN model to GEXF: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to GEXF {output_file_path}: {e}", exc_info=True)
        raise

def export_to_graphml(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a GraphML file."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX not available, GraphML export failed.")
    graph = _build_networkx_graph(gnn_model)
    if graph is None: return
    try:
        nx.write_graphml(graph, output_file_path)
        logger.debug(f"Successfully exported GNN model to GraphML: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to GraphML {output_file_path}: {e}", exc_info=True)
        raise

def export_to_json_adjacency_list(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a JSON adjacency list format."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX not available, JSON adjacency list export failed.")
    graph = _build_networkx_graph(gnn_model)
    if graph is None: return
    try:
        # Need to import json here as it's not a top-level import for this specific module
        import json 
        adj_data = nx.readwrite.json_graph.adjacency_data(graph)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(adj_data, f, indent=4, ensure_ascii=False)
        logger.debug(f"Successfully exported GNN model to JSON adjacency list: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to JSON adjacency list {output_file_path}: {e}", exc_info=True)
        raise 