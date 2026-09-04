#!/usr/bin/env python3
"""
Advanced visualization module for GNN pipeline (real implementations).

Naming convention: module-level functions use ``create_*`` to signal they
assemble and return data-structure dictionaries (not files/figures), while
class methods that produce output files or matplotlib figures use
``generate_*``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# Use non-interactive backend for server/CI environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Use local data extraction and visualization utilities
try:
    from .data_extractor import VisualizationDataExtractor

    VIS_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.debug(
        "VisualizationDataExtractor unavailable; using reduced visualization path."
    )
    VIS_PROCESSOR_AVAILABLE = False  # graceful degradation without data extractor


class ExtractorLike(Protocol):
    """Structural type for the extractor injection seam.

    Any object exposing ``extract_from_content(content) -> dict`` qualifies:
    the real ``VisualizationDataExtractor`` or lightweight test stubs.
    """

    def extract_from_content(self, content: str) -> Dict[str, Any]:
        """Extract structured model data from raw GNN content."""
        ...


class AdvancedVisualizer:
    """
    Real advanced visualizer that composes multiple visualization backends
    to generate a comprehensive set of artifacts per GNN file.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        extractor: Optional[ExtractorLike] = None,
        strict_validation: bool = True,
    ) -> None:
        """Initialize the instance.

        ``extractor`` accepts any object exposing
        ``extract_from_content(content) -> dict`` (a real
        ``VisualizationDataExtractor`` or a test stub). When omitted, one is
        built lazily per run; if the import is unavailable the visualizer
        degrades to the recovery path (previous module-global behavior).

        ``strict_validation`` (default ``True``, per the documented API)
        maps onto the lazily built ``VisualizationDataExtractor``; it is
        ignored when an ``extractor`` is injected (injection wins).
        """
        self.logger = logger or logging.getLogger("advanced_visualization")
        self._extractor = extractor
        self._strict_validation = strict_validation

    def _get_extractor(self) -> Optional[ExtractorLike]:
        """Return the injected extractor, or build a real one lazily.

        ``None`` means extraction is unavailable (nothing injected and the
        import failed) and the caller must take the recovery path.
        """
        if self._extractor is not None:
            return self._extractor
        if not VIS_PROCESSOR_AVAILABLE:
            return None
        return VisualizationDataExtractor(strict_validation=self._strict_validation)

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

        extractor = self._get_extractor()
        if extractor is None:
            # No extractor available (not injected, import failed): recovery.
            try:
                fallback_files = self._generate_fallback_visualizations(
                    content, model_name, model_output_dir
                )
                generated.extend(fallback_files)
            except Exception as e:
                self.logger.warning(
                    f"Recovery visualizations failed for {model_name}: {e}"
                )
            return generated

        extracted_data = extractor.extract_from_content(content)

        if not extracted_data.get("success", False):
            self.logger.warning(
                f"Data extraction failed for {model_name}, using recovery"
            )
            try:
                fallback_files = self._generate_fallback_visualizations(
                    content, model_name, model_output_dir
                )
                generated.extend(fallback_files)
            except Exception as e:
                self.logger.warning(
                    f"Recovery visualizations failed for {model_name}: {e}"
                )
            return generated

        # Stage dispatch: each stage is a (label, creator) pair; failures in one
        # stage never abort the others (matches the prior per-stage try/except).
        stages: list[
            tuple[str, Callable[[Dict[str, Any], str, Path], Optional[str]]]
        ] = [
            ("Statistical visualizations", self._create_statistics_plot),
            ("Network visualizations", self._create_network_graph),
            ("Matrix visualizations", self._create_matrix_heatmap),
        ]
        for label, create_fn in stages:
            generated.extend(
                self._run_stage(
                    label, create_fn, extracted_data, model_name, model_output_dir
                )
            )

        # Optional HTML summary page that links artifacts (real, non-interactive)
        if "html" in export_formats:
            try:
                html_path = self._generate_summary_html(
                    model_name, model_output_dir, generated
                )
                if html_path:
                    generated.append(str(html_path))
            except Exception as e:
                self.logger.warning(
                    f"Summary HTML generation failed for {model_name}: {e}"
                )

        # Optional JSON manifest of generated files
        if "json" in export_formats:
            try:
                manifest: dict[str, Any] = {
                    "model": model_name,
                    "generated": generated,
                    "timestamp": datetime.now().isoformat(),
                }
                manifest_path = (
                    model_output_dir / f"{model_name}_advanced_viz_manifest.json"
                )
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                generated.append(str(manifest_path))
            except Exception as e:
                self.logger.warning(f"Manifest JSON write failed for {model_name}: {e}")

        return generated

    def _generate_summary_html(
        self, model_name: str, model_output_dir: Path, files: List[str]
    ) -> Optional[Path]:
        """Generate a simple HTML page linking to produced artifacts."""
        try:
            rel_files = [Path(f) for f in files]
            # Build HTML content with embedded previews for PNGs
            items: list[Any] = []
            for f in rel_files:
                name = f.name
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                    items.append(
                        f"<div class='item'><h4>{name}</h4><img src='{name}' style='max-width:100%'></div>"
                    )
                else:
                    items.append(
                        f"<div class='item'><a href='{name}' target='_blank'>{name}</a></div>"
                    )
            html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{model_name} Advanced Visualizations</title>
<style>body{{font-family:Arial,sans-serif;padding:20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}.item{{background:#f8f9fa;padding:10px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}h2{{margin:0 0 10px 0}}</style>
</head><body>
<h2>Advanced Visualizations: {model_name}</h2>
<div class='grid'>
{"".join(items)}
</div>
</body></html>
            """
            out = model_output_dir / f"{model_name}_advanced_summary.html"
            with open(out, "w", encoding="utf-8") as output_file:
                output_file.write(html)
            return out
        except (OSError, ValueError, TypeError) as e:
            self.logger.debug(f"HTML summary generation failed: {e}")
            return None  # HTML summary generation is best-effort

    def _generate_fallback_visualizations(
        self, content: str, model_name: str, output_dir: Path
    ) -> List[str]:
        """Generate recovery visualizations when advanced libraries aren't available"""
        generated: list[Any] = []

        try:
            # Create a simple text-based summary
            summary_file = output_dir / f"{model_name}_fallback_summary.html"
            html_content = f"""
<!DOCTYPE html>
<html><head><title>{model_name} - Recovery Visualization</title>
<style>body {{ font-family: Arial, sans-serif; margin: 20px; }}
.content {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
pre {{ background: white; padding: 15px; border-radius: 5px; white-space: pre-wrap; }}
</style></head>
<body>
<h1>{model_name} - Recovery Visualization</h1>
<div class="content">
<h2>Model Content Summary</h2>
<pre>{content[:1000]}{"..." if len(content) > 1000 else ""}</pre>
</div>
</body></html>
"""
            with open(summary_file, "w") as f:
                f.write(html_content)
            generated.append(str(summary_file))

        except Exception as e:
            self.logger.error(f"Failed to generate recovery visualization: {e}")

        return generated

    def _run_stage(
        self,
        label: str,
        create_fn: Callable[[Dict[str, Any], str, Path], Optional[str]],
        extracted_data: Dict[str, Any],
        model_name: str,
        output_dir: Path,
    ) -> List[str]:
        """Run one visualization stage; return ``[path]`` on success, ``[]`` otherwise.

        Centralizes the try/except + optional-path bookkeeping that was previously
        triplicated across ``_generate_statistical/network/matrix_visualizations``.
        """
        try:
            path = create_fn(extracted_data, model_name, output_dir)
            return [path] if path else []
        except Exception as e:
            self.logger.warning(f"{label} failed for {model_name}: {e}")
            return []

    def _create_statistics_plot(
        self, extracted_data: Dict[str, Any], model_name: str, output_dir: Path
    ) -> Optional[str]:
        """Create statistical analysis plot"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract statistics
            blocks = extracted_data.get("blocks", [])
            extracted_data.get("connections", [])

            # Create bar chart of variable types
            if blocks:
                type_counts: dict[Any, Any] = {}
                for block in blocks:
                    var_type = block.get("type", "unknown")
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1

                types = list(type_counts.keys())
                counts = list(type_counts.values())

                ax.bar(types, counts, alpha=0.7)
                ax.set_title(f"Model Variable Types: {model_name}")
                ax.set_xlabel("Variable Type")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            output_file = output_dir / f"{model_name}_statistics.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            return str(output_file)
        except Exception as e:
            self.logger.error(f"Failed to create statistics plot: {e}")
            return None

    def _create_network_graph(
        self, extracted_data: Dict[str, Any], model_name: str, output_dir: Path
    ) -> Optional[str]:
        """Create network graph visualization"""
        try:
            # Simple network visualization
            fig, ax = plt.subplots(figsize=(8, 6))

            blocks = extracted_data.get("blocks", [])
            connections = extracted_data.get("connections", [])

            if blocks:
                # Create simple node positions
                n_nodes = len(blocks)
                rng = np.random.default_rng(42)
                positions = rng.random((n_nodes, 2)) * 10

                # Plot nodes
                for i, block in enumerate(blocks):
                    ax.scatter(positions[i, 0], positions[i, 1], s=100, alpha=0.7)
                    ax.annotate(
                        block.get("name", f"Node {i}"),
                        (positions[i, 0], positions[i, 1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

                # Plot connections: resolve real node indices via a
                # name→index map over ``blocks``. Connection dicts carry
                # ``source_variables``/``target_variables`` lists (extractor
                # format); legacy ``source``/``target`` scalars are accepted
                # as singletons. Unresolvable pairs are skipped silently.
                name_to_idx = {
                    b.get("name"): i for i, b in enumerate(blocks) if b.get("name")
                }
                for conn in connections:
                    if not isinstance(conn, dict):
                        continue
                    sources = conn.get("source_variables")
                    if sources is None:
                        sources = (
                            [conn["source"]] if conn.get("source") is not None else []
                        )
                    targets = conn.get("target_variables")
                    if targets is None:
                        targets = (
                            [conn["target"]] if conn.get("target") is not None else []
                        )
                    for s in sources:
                        for t in targets:
                            s_idx = name_to_idx.get(s)
                            t_idx = name_to_idx.get(t)
                            if s_idx is None or t_idx is None:
                                continue
                            ax.plot(
                                [positions[s_idx, 0], positions[t_idx, 0]],
                                [positions[s_idx, 1], positions[t_idx, 1]],
                                "r-",
                                alpha=0.5,
                            )

            ax.set_title(f"Network Graph: {model_name}")
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)

            plt.tight_layout()
            output_file = output_dir / f"{model_name}_network.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            return str(output_file)
        except Exception as e:
            self.logger.error(f"Failed to create network graph: {e}")
            return None

    def _create_matrix_heatmap(
        self, extracted_data: Dict[str, Any], model_name: str, output_dir: Path
    ) -> Optional[str]:
        try:
            # Use a real matrix from extracted parameters when available; fall back
            # to a deterministic demo matrix (seeded) so output is reproducible.
            sample_data = None
            for param in extracted_data.get("parameters", []):
                value = param.get("value") if isinstance(param, dict) else None
                if isinstance(value, list) and value and isinstance(value[0], list):
                    try:
                        sample_data = np.array(value, dtype=float)
                        break
                    except (ValueError, TypeError):
                        continue
            if sample_data is None:
                rng = np.random.default_rng(42)
                sample_data = rng.random((5, 5))

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(sample_data, cmap="viridis", aspect="auto")

            ax.set_title(f"Matrix Heatmap: {model_name}")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            plt.colorbar(im)

            plt.tight_layout()
            output_file = output_dir / f"{model_name}_heatmap.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            # Export matrix data to CSV for accessibility
            csv_file = output_dir / f"{model_name}_heatmap_data.csv"
            try:
                import csv

                with open(csv_file, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"Matrix Heatmap Data: {model_name}"])
                    writer.writerow([f"Shape: {sample_data.shape}"])
                    writer.writerow([f"Data type: {sample_data.dtype}"])
                    writer.writerow([])  # Empty row

                    # Write matrix data
                    writer.writerow([f"Col {j}" for j in range(sample_data.shape[1])])
                    for i, row in enumerate(sample_data):
                        writer.writerow([f"Row {i}"] + row.tolist())
            except Exception as e:
                self.logger.warning(f"Failed to export matrix data to CSV: {e}")

            return str(
                output_file
            )  # Return PNG file path, CSV file is saved but not returned
        except Exception as e:
            self.logger.error(f"Failed to create matrix heatmap: {e}")
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

    except (KeyError, ValueError, TypeError) as e:
        logging.getLogger(__name__).debug(f"Visualization creation failed: {e}")
        return None  # visualization creation is best-effort


def create_dashboard_section(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a dashboard section from data."""
    try:
        section: dict[str, Any] = {
            "title": data.get("title", "Section"),
            "type": data.get("type", "text"),
            "content": data.get("content", ""),
            "metrics": data.get("metrics", {}),
        }

        return section

    except (KeyError, TypeError) as e:
        logging.getLogger(__name__).debug(f"Dashboard section creation failed: {e}")
        return None  # malformed data, skip section


def create_network_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a network visualization."""
    try:
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Create network visualization data
        viz_data: dict[str, Any] = {
            "type": "network",
            "nodes": nodes,
            "edges": edges,
            "layout": "force_directed",
            "options": {
                "node_size": 10,
                "edge_width": 1,
                "node_color": "blue",
                "edge_color": "gray",
            },
        }

        return viz_data

    except Exception as e:
        return {"error": str(e)}


def create_timeline_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a timeline visualization."""
    try:
        events = data.get("events", [])

        # Create timeline visualization data
        viz_data: dict[str, Any] = {
            "type": "timeline",
            "events": events,
            "options": {"height": 400, "width": 800, "show_labels": True},
        }

        return viz_data

    except Exception as e:
        return {"error": str(e)}


def create_heatmap_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a heatmap visualization."""
    try:
        matrix = data.get("matrix", [])

        # Create heatmap visualization data
        viz_data: dict[str, Any] = {
            "type": "heatmap",
            "matrix": matrix,
            "options": {
                "colormap": "viridis",
                "show_values": True,
                "aspect_ratio": "auto",
            },
        }

        return viz_data

    except Exception as e:
        return {"error": str(e)}


def create_default_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a default visualization."""
    try:
        # Create a simple chart visualization
        viz_data: dict[str, Any] = {
            "type": "chart",
            "data": data,
            "options": {
                "chart_type": "line",
                "title": "GNN Analysis",
                "x_label": "Time",
                "y_label": "Value",
            },
        }

        return viz_data

    except Exception as e:
        return {"error": str(e)}
