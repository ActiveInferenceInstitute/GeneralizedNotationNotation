#!/usr/bin/env python3
"""
Integration Processor module for GNN Processing Pipeline.

This module provides integration processing capabilities.
"""

from pathlib import Path
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error
)

logger = logging.getLogger(__name__)

def process_integration(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process integration for GNN files.
    
    This module performs system-level consistency checks, builds a dependency graph
    of components, and detects circular dependencies or isolated components.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("integration")

    try:
        log_step_start(logger, "Processing integration")

        # Create results directory (with integration_results subdirectory)
        results_dir = output_dir / "integration_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "processed_files": 0,
            "success": True,
            "errors": [],
            "system_graph_stats": {},
            "issues": []
        }

        # Search for GNN files in multiple locations
        import re
        gnn_files = list(target_dir.glob("*.md"))

        # Also check parent/input directories for GNN source files
        project_root = target_dir
        for _ in range(5):  # Walk up to find project root
            input_dir = project_root / "input" / "gnn_files"
            if input_dir.exists():
                gnn_files.extend(list(input_dir.glob("*.md")))
                break
            parent = project_root.parent
            if parent == project_root:
                break
            project_root = parent

        # Deduplicate by filename
        seen_names = set()
        unique_gnn_files = []
        for f in gnn_files:
            if f.name not in seen_names:
                seen_names.add(f.name)
                unique_gnn_files.append(f)
        gnn_files = unique_gnn_files

        results["processed_files"] = len(gnn_files)

        # Build System Graph
        try:
            import networkx as nx
            G = nx.DiGraph()
            has_networkx = True
        except ImportError:
            has_networkx = False
            G = {}

        component_locations = {}  # component_name -> filename

        if verbose:
            logger.debug(f"Processing {len(gnn_files)} GNN files for integration analysis")

        for gnn_file in gnn_files:
            try:
                content = gnn_file.read_text()

                # 1. Extract variables from ## StateSpaceBlock section
                state_section = re.search(
                    r'##\s*StateSpaceBlock\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
                )
                if state_section:
                    section_text = state_section.group(1)
                    for line in section_text.strip().split('\n'):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        # Match variable declarations like: A[3,3,type=float]
                        var_match = re.match(r'(\w+)\s*[\[\(]', line)
                        if var_match:
                            comp = var_match.group(1)
                            component_locations[comp] = gnn_file.name
                            if has_networkx:
                                G.add_node(comp, file=gnn_file.name)
                            else:
                                G.setdefault(comp, [])

                # 2. Also extract from YAML-style definitions (previous support)
                matches = re.finditer(r'^\s*-\s*name:\s*(\w+)', content, re.MULTILINE)
                for match in matches:
                    comp = match.group(1)
                    component_locations[comp] = gnn_file.name
                    if has_networkx:
                        G.add_node(comp, file=gnn_file.name)
                    else:
                        G.setdefault(comp, [])

                # 3. Extract connections from ## Connections section
                conn_section = re.search(
                    r'##\s*Connections\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
                )
                if conn_section:
                    section_text = conn_section.group(1)
                    for line in section_text.strip().split('\n'):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        # GNN connection operators: > (directional), - (bidirectional), < (reverse)
                        conn_match = re.match(r'(\w+)\s*([>\-<])\s*(\w+)', line)
                        if conn_match:
                            src, op, tgt = conn_match.group(1), conn_match.group(2), conn_match.group(3)
                            # Ensure nodes exist
                            for node in (src, tgt):
                                if has_networkx:
                                    if not G.has_node(node):
                                        G.add_node(node, file=gnn_file.name)
                                else:
                                    G.setdefault(node, [])
                                component_locations.setdefault(node, gnn_file.name)

                            if op == '>':
                                if has_networkx:
                                    G.add_edge(src, tgt, type="directional")
                                else:
                                    G.setdefault(src, []).append(tgt)
                            elif op == '-':
                                if has_networkx:
                                    G.add_edge(src, tgt, type="bidirectional")
                                    G.add_edge(tgt, src, type="bidirectional")
                                else:
                                    G.setdefault(src, []).append(tgt)
                                    G.setdefault(tgt, []).append(src)
                            elif op == '<':
                                if has_networkx:
                                    G.add_edge(tgt, src, type="reverse")
                                else:
                                    G.setdefault(tgt, []).append(src)

                if verbose:
                    logger.debug(f"Parsed {gnn_file.name}: found {len(component_locations)} components")

            except Exception as e:
                logger.warning(f"Failed to parse {gnn_file.name}: {e}")
                continue

        # NOTE: Cross-file reference edges via content matching have been removed.
        # GNN models share a common mathematical vocabulary (s_prime, beta, alpha,
        # s_tau1, etc.), so substring/word-boundary matching always generates
        # false-positive edges. Real cross-file dependencies are detected via
        # explicit $ref: syntax (see the "Verify cross-references" section below).

        if has_networkx:
            try:
                # Count cycles for structural analysis. Intra-model cycles from
                # ## Connections are expected mathematical relationships (e.g.,
                # bidirectional s - A creates s→A→s loops), so we report them
                # as informational structure, not as dependency issues.
                import time as _t

                MAX_CYCLE_LENGTH = 6   # Short cycles only for structural metrics
                MAX_CYCLE_TIME = 5     # Quick scan — not a critical check

                cycle_count = 0
                try:
                    _deadline = _t.monotonic() + MAX_CYCLE_TIME
                    for _c in nx.simple_cycles(G, length_bound=MAX_CYCLE_LENGTH):
                        cycle_count += 1
                        if _t.monotonic() > _deadline or cycle_count >= 500:
                            break
                except TypeError:
                    # Older networkx without length_bound
                    try:
                        _deadline = _t.monotonic() + MAX_CYCLE_TIME
                        for _c in nx.simple_cycles(G):
                            cycle_count += 1
                            if _t.monotonic() > _deadline or cycle_count >= 500:
                                break
                    except Exception:
                        cycle_count = 0

                # Detect isolated nodes
                isolated = list(nx.isolates(G))
                if isolated:
                    results["issues"].append(f"Isolated components (no connections): {isolated}")

                results["system_graph_stats"] = {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "cycles": cycle_count,
                    "isolated_nodes": len(isolated),
                    "components": nx.number_weakly_connected_components(G)
                }

                logger.info(f"System graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                           f"{cycle_count} intra-model cycles (structural), {len(isolated)} isolated")
            except Exception as e:
                logger.warning(f"Failed to analyze graph: {e}")
        else:
            node_count = len(G)
            edge_count = sum(len(edges) for edges in G.values())
            results["system_graph_stats"] = {
                "nodes": node_count,
                "edges": edge_count
            }

        # Verify cross-references
        for gnn_file in gnn_files:
            try:
                content = gnn_file.read_text()
                refs = re.findall(r'\$ref:\s*(\w+)', content)
                for ref in refs:
                    if ref not in component_locations:
                        results["issues"].append(
                            f"Undefined reference '{ref}' in {gnn_file.name}"
                        )
                type_refs = re.findall(r'type:\s*(\w+)', content)
                for ref in type_refs:
                    if ref[0].isupper() and ref not in component_locations:
                        if ref not in ['String', 'Integer', 'Float', 'Boolean', 'Array', 'Object']:
                            results["issues"].append(
                                f"Possible undefined type '{ref}' in {gnn_file.name}"
                            )
            except Exception as e:
                logger.warning(f"Failed to verify references in {gnn_file.name}: {e}")

        # Save results
        import json
        results_file = results_dir / "integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate summary
        node_count = results['system_graph_stats'].get('nodes', 0)
        edge_count = results['system_graph_stats'].get('edges', 0)
        summary = "# System Integration Report\n\n"
        summary += f"Scanned {len(gnn_files)} files.\n\n"
        summary += f"- **Graph Nodes**: {node_count}\n"
        summary += f"- **Graph Edges**: {edge_count}\n"
        if results['system_graph_stats'].get('cycles', 0) > 0:
            summary += f"- **Cycles Detected**: {results['system_graph_stats']['cycles']}\n"
        if results['system_graph_stats'].get('isolated_nodes', 0) > 0:
            summary += f"- **Isolated Nodes**: {results['system_graph_stats']['isolated_nodes']}\n"
        if results["issues"]:
            summary += f"\n## Issues ({len(results['issues'])})\n\n"
            for issue in results["issues"]:
                summary += f"- {issue}\n"
        else:
            summary += "\nNo issues detected.\n"

        (results_dir / "integration_summary.md").write_text(summary)

        if results["success"]:
            log_step_success(logger, f"Integration processing completed: {node_count} nodes, {edge_count} edges")
        else:
            log_step_error(logger, "integration processing failed")

        return results["success"]

    except Exception as e:
        log_step_error(logger, "integration processing failed", {"error": str(e)})
        return False
