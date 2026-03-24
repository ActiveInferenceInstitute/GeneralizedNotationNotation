"""Step-8 visualization orchestration (JSON-first model loading)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from advanced_visualization._shared import normalize_connection_format
from utils.logging.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

from visualization.analysis.combined_analysis import (
    generate_combined_analysis,
    generate_combined_visualizations,
)
from visualization.core.parsed_model import (
    load_visualization_model,
    write_stale_json_note_if_needed,
)
from visualization.graph import (
    generate_network_visualizations,
    generate_variable_parameter_bipartite,
)
from visualization.matrix.compat import generate_matrix_visualizations, parse_matrix_data

logger = logging.getLogger(__name__)


def _filter_connections(
    connections: List[Dict[str, Any]], var_names: Set[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for conn in connections:
        if not isinstance(conn, dict):
            continue
        n = normalize_connection_format(conn)
        sources = n.get("source_variables") or []
        targets = n.get("target_variables") or []
        if any(s in var_names for s in sources) and any(t in var_names for t in targets):
            out.append(conn)
    return out


def process_visualization(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    logger_v = logging.getLogger("visualization")
    try:
        log_step_start(logger_v, "Processing visualizations")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            gnn_files = list(target_dir.glob("*.gnn"))
        if not gnn_files:
            log_step_warning(logger_v, "No GNN files found for visualization")
            return True

        all_visualizations: List[str] = []
        for gnn_file in gnn_files:
            try:
                all_visualizations.extend(
                    process_single_gnn_file(gnn_file, results_dir, verbose)
                )
            except Exception as e:
                logger_v.error("Error processing %s: %s", gnn_file, e)

        if len(gnn_files) > 1:
            try:
                all_visualizations.extend(
                    generate_combined_visualizations(gnn_files, results_dir, verbose)
                )
            except Exception as e:
                logger_v.error("Error generating combined visualizations: %s", e)

        results_summary = {
            "processed_files": len(gnn_files),
            "total_visualizations": len(all_visualizations),
            "visualization_files": all_visualizations,
            "success": len(all_visualizations) > 0,
        }

        summary_file = results_dir / "visualization_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results_summary, f, indent=2)

        if results_summary["success"]:
            log_step_success(
                logger_v, f"Generated {len(all_visualizations)} visualizations"
            )
        else:
            log_step_error(logger_v, "No visualizations generated")

        return results_summary["success"]

    except Exception as e:
        log_step_error(logger_v, f"Visualization processing failed: {e}")
        return False


def process_single_gnn_file(
    gnn_file: Path, results_dir: Path, verbose: bool = False
) -> List[str]:
    from visualization.matrix.visualizer import MatrixVisualizer

    with open(gnn_file, encoding="utf-8") as f:
        content = f.read()

    model_name = gnn_file.stem
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True)

    existing_pngs = sorted(str(p) for p in model_dir.glob("*.png"))
    if existing_pngs:
        source_mtime = gnn_file.stat().st_mtime
        cache_mtime = min(Path(png).stat().st_mtime for png in existing_pngs)
        if cache_mtime >= source_mtime:
            if verbose:
                print(f"Using cached visualizations for {model_name}")
            return existing_pngs
        for png_file in existing_pngs:
            try:
                Path(png_file).unlink()
            except OSError as e:
                logger.debug("Could not remove stale cache file %s: %s", png_file, e)

    parsed_data = load_visualization_model(gnn_file, content, results_dir, verbose)
    write_stale_json_note_if_needed(parsed_data, model_dir, model_name, gnn_file)

    should_sample = False
    if parsed_data.get("variables") and len(parsed_data["variables"]) > 100:
        should_sample = True
        if verbose:
            print(f"Large dataset detected for {model_name}, applying sampling")

    if should_sample:
        original_vars = len(parsed_data.get("variables", []))
        original_conns = len(parsed_data.get("connections", []))
        parsed_data["variables"] = parsed_data["variables"][:100]
        var_names = {var["name"] for var in parsed_data["variables"] if isinstance(var, dict)}
        parsed_data["connections"] = _filter_connections(
            parsed_data.get("connections", []), var_names
        )
        if parsed_data.get("matrices") and len(parsed_data["matrices"]) > 5:
            parsed_data["matrices"] = parsed_data["matrices"][:5]
        parsed_data["_sampling_applied"] = {
            "original_variables": original_vars,
            "original_connections": original_conns,
            "sampled_variables": len(parsed_data["variables"]),
            "sampled_connections": len(parsed_data["connections"]),
        }

    visualizations: List[str] = []

    if len(parsed_data.get("variables", [])) <= 200:
        try:
            visualizations.extend(
                generate_network_visualizations(parsed_data, model_dir, model_name)
            )
        except Exception as e:
            if verbose:
                print(f"Network visualization failed for {model_name}: {e}")
    elif verbose:
        print(f"Skipping network visualizations for {model_name} - too many nodes")

    try:
        visualizations.extend(
            generate_variable_parameter_bipartite(parsed_data, model_dir, model_name)
        )
    except Exception as e:
        if verbose:
            logger.debug("Bipartite visualization skipped: %s", e)

    try:
        mv = MatrixVisualizer()
        matrices: Dict[str, Any] = {}
        parameters = parsed_data.get("parameters", [])
        if parameters:
            matrices = mv.extract_matrix_data_from_parameters(parameters)
        if not matrices:
            matrices = mv.extract_matrix_data_from_parameters(
                parsed_data.get("variables", [])
            )
        if not matrices:
            for m_info in parsed_data.get("matrices", []):
                if isinstance(m_info, dict) and "data" in m_info:
                    m_name = m_info.get("name", f"matrix_{len(matrices)}")
                    try:
                        import numpy as np

                        m_data = np.array(m_info["data"], dtype=float)
                        matrices[m_name] = m_data
                    except (ValueError, TypeError) as e:
                        logger.debug(
                            "Skipping non-numeric matrix data for %s: %s",
                            m_info.get("name", "?"),
                            e,
                        )

        if matrices:
            for m_name, m_data in matrices.items():
                if m_data.ndim == 3:
                    m_path = model_dir / f"{model_name}_{m_name}_tensor.png"
                    if mv.generate_3d_tensor_visualization(
                        m_name, m_data, m_path, tensor_type="transition"
                    ):
                        visualizations.append(str(m_path))
                    analysis_path = model_dir / f"{model_name}_{m_name}_analysis.png"
                    mv.generate_pomdp_transition_analysis(m_data, analysis_path)
                    visualizations.append(str(analysis_path))
                else:
                    m_path = model_dir / f"{model_name}_{m_name}_heatmap.png"
                    if mv.generate_matrix_heatmap(m_name, m_data, m_path):
                        visualizations.append(str(m_path))
            if verbose:
                logger.info(
                    "Generated %s matrix visualizations for %s",
                    len(matrices),
                    model_name,
                )
        elif verbose:
            logger.warning(
                "No matrix data found for %s - checked parameters, variables, matrices",
                model_name,
            )
    except Exception as e:
        if verbose:
            logger.exception("Matrix visualization failed for %s: %s", model_name, e)

    try:
        visualizations.extend(
            generate_combined_analysis(parsed_data, model_dir, model_name)
        )
    except Exception as e:
        if verbose:
            print(f"Combined analysis failed for {model_name}: {e}")

    if should_sample and visualizations:
        try:
            sampling_note = model_dir / f"{model_name}_sampling_note.txt"
            with open(sampling_note, "w") as f:
                f.write(f"Sampling applied to {model_name}:\n")
                f.write(
                    f"Original variables: {parsed_data['_sampling_applied']['original_variables']}\n"
                )
                f.write(
                    f"Sampled variables: {parsed_data['_sampling_applied']['sampled_variables']}\n"
                )
                f.write(
                    f"Original connections: {parsed_data['_sampling_applied']['original_connections']}\n"
                )
                f.write(
                    f"Sampled connections: {parsed_data['_sampling_applied']['sampled_connections']}\n"
                )
        except OSError as e:
            logger.debug("Could not write sampling note for %s: %s", model_name, e)

    manifest_path = model_dir / f"{model_name}_viz_manifest.json"
    try:
        meta = parsed_data.get("_viz_meta") or {}
        manifest: Dict[str, Any] = {
            "model_name": model_name,
            "viz_meta": meta,
            "artifact_count": len(visualizations),
            "artifacts": list(visualizations),
            "variable_count": len(parsed_data.get("variables") or []),
            "connection_count": len(parsed_data.get("connections") or []),
            "parameter_count": len(parsed_data.get("parameters") or []),
            "ontology_label_count": len(parsed_data.get("ontology_labels") or {}),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        visualizations.append(str(manifest_path))
    except (OSError, TypeError, ValueError) as e:
        logger.debug("Could not write viz manifest for %s: %s", model_name, e)

    return visualizations
