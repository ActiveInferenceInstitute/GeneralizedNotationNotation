#!/usr/bin/env python3
"""
RxInfer.jl Results Parser for GNN Execute Pipeline.

Parses JSON output from executed RxInfer.jl scripts, extracts posterior distributions,
convergence metrics, and formats results for downstream pipeline steps.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_rxinfer_output(output_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse JSON output produced by a RxInfer.jl script.

    RxInfer.jl scripts are expected to write a JSON file with this structure:
    {
        "model_name": "...",
        "iterations": N,
        "converged": true/false,
        "free_energy": [...],
        "posteriors": {
            "variable_name": {"mean": [...], "variance": [...]}
        },
        "elapsed_seconds": ...
    }

    Args:
        output_path: Path to the JSON output file from RxInfer.jl

    Returns:
        Parsed results dict, or None if parsing fails
    """
    if not output_path.exists():
        logger.warning(f"RxInfer output file not found: {output_path}")
        return None

    try:
        with open(output_path, 'r') as f:
            raw = json.load(f)

        # Normalize: handle both snake_case and camelCase keys from Julia output
        normalized = {
            "model_name": raw.get("model_name") or raw.get("modelName", "unknown"),
            "iterations": raw.get("iterations") or raw.get("n_iterations", 0),
            "converged": raw.get("converged", False),
            "free_energy": _extract_free_energy(raw),
            "posteriors": _extract_posteriors(raw),
            "elapsed_seconds": raw.get("elapsed_seconds") or raw.get("elapsed", 0.0),
            "framework": "RxInfer.jl",
            "parse_timestamp": datetime.now().isoformat()
        }

        return normalized

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {output_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {output_path}: {e}")
        return None


def _extract_free_energy(raw: Dict[str, Any]) -> List[float]:
    """Extract free energy trajectory from raw RxInfer output."""
    # RxInfer may output free_energy as list, or as nested dict with "values"
    fe = raw.get("free_energy") or raw.get("freeEnergy") or raw.get("F")

    if fe is None:
        return []
    if isinstance(fe, list):
        return [float(v) for v in fe if v is not None]
    if isinstance(fe, dict):
        values = fe.get("values") or fe.get("data") or []
        return [float(v) for v in values if v is not None]
    try:
        return [float(fe)]
    except (TypeError, ValueError):
        return []


def _extract_posteriors(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract posterior distribution parameters from raw RxInfer output.

    Handles NormalMeanVariance, Dirichlet, and Categorical distributions
    as output by RxInfer.jl's standard inference routines.
    """
    raw_posteriors = raw.get("posteriors") or raw.get("q") or {}
    result = {}

    for var_name, dist_data in raw_posteriors.items():
        if not isinstance(dist_data, dict):
            # Scalar output
            try:
                result[var_name] = {"type": "scalar", "value": float(dist_data)}
            except (TypeError, ValueError):
                pass
            continue

        dist_type = dist_data.get("type") or dist_data.get("distribution", "unknown")

        parsed = {"type": dist_type}

        # NormalMeanVariance (Gaussian)
        if any(k in dist_data for k in ["mean", "μ", "mu"]):
            mean = dist_data.get("mean") or dist_data.get("μ") or dist_data.get("mu")
            var = (dist_data.get("variance") or dist_data.get("σ²") or
                   dist_data.get("sigma2") or dist_data.get("cov"))
            parsed["mean"] = _to_float_list(mean)
            parsed["variance"] = _to_float_list(var) if var is not None else None

        # Dirichlet
        if any(k in dist_data for k in ["alpha", "α", "concentration"]):
            conc = (dist_data.get("alpha") or dist_data.get("α") or
                    dist_data.get("concentration"))
            parsed["concentration"] = _to_float_list(conc)
            # Compute mean of Dirichlet
            conc_list = parsed.get("concentration", [])
            if conc_list:
                total = sum(conc_list)
                parsed["mean"] = [c / total for c in conc_list] if total > 0 else []

        # Categorical
        if any(k in dist_data for k in ["p", "probs", "probabilities"]):
            probs = (dist_data.get("p") or dist_data.get("probs") or
                     dist_data.get("probabilities"))
            parsed["probabilities"] = _to_float_list(probs)
            parsed["mean"] = parsed["probabilities"]

        result[var_name] = parsed

    return result


def _to_float_list(value: Any) -> List[float]:
    """Convert a value (scalar, list, or nested list) to a flat float list."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, (int, float)):
                result.append(float(item))
            elif isinstance(item, list):
                result.extend(_to_float_list(item))
        return result
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def extract_convergence_metrics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute convergence metrics from parsed RxInfer results.

    Args:
        parsed: Output from parse_rxinfer_output()

    Returns:
        Dict with convergence indicators:
        - converged: bool
        - final_free_energy: float
        - free_energy_change: float (change from first to last iteration)
        - relative_change: float (|change| / |first_fe|)
        - iterations_to_convergence: int or None
    """
    metrics = {
        "converged": parsed.get("converged", False),
        "total_iterations": parsed.get("iterations", 0),
        "final_free_energy": None,
        "free_energy_change": None,
        "relative_change": None,
        "iterations_to_convergence": None,
        "monotone_decrease": None
    }

    fe_trace = parsed.get("free_energy", [])

    if not fe_trace:
        return metrics

    metrics["final_free_energy"] = fe_trace[-1]

    if len(fe_trace) >= 2:
        fe_change = fe_trace[-1] - fe_trace[0]
        metrics["free_energy_change"] = fe_change

        if abs(fe_trace[0]) > 1e-10:
            metrics["relative_change"] = abs(fe_change) / abs(fe_trace[0])

        # Check monotone decrease (VFE should decrease during VBEM)
        decreases = sum(1 for i in range(1, len(fe_trace)) if fe_trace[i] < fe_trace[i-1])
        metrics["monotone_decrease"] = decreases == len(fe_trace) - 1

        # Estimate iterations to convergence (where |ΔFE| < 0.01 * |FE_0|)
        threshold = 0.01 * abs(fe_trace[0]) if abs(fe_trace[0]) > 1e-10 else 0.001
        for i in range(1, len(fe_trace)):
            if abs(fe_trace[i] - fe_trace[i-1]) < threshold:
                metrics["iterations_to_convergence"] = i
                break

    return metrics


def summarize_posteriors(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate human-readable posterior summary.

    Args:
        parsed: Output from parse_rxinfer_output()

    Returns:
        Summary dict mapping variable names to their posterior statistics
    """
    posteriors = parsed.get("posteriors", {})
    summary = {}

    for var_name, dist in posteriors.items():
        var_summary = {"type": dist.get("type", "unknown")}

        mean = dist.get("mean")
        if mean:
            if len(mean) == 1:
                var_summary["mean"] = mean[0]
            else:
                var_summary["mean"] = mean
                # Add entropy-like measure for categorical distributions
                if dist.get("type") in ["categorical", "dirichlet"]:
                    import math
                    var_summary["argmax"] = mean.index(max(mean))
                    # Entropy
                    entropy = -sum(p * math.log(p + 1e-10) for p in mean)
                    var_summary["entropy"] = round(entropy, 4)

        variance = dist.get("variance")
        if variance:
            if len(variance) == 1:
                var_summary["std"] = variance[0] ** 0.5
            else:
                var_summary["variance"] = variance

        summary[var_name] = var_summary

    return summary


def collect_rxinfer_results(
    output_dir: Path,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Collect and parse all RxInfer result JSON files from an output directory.

    Args:
        output_dir: Directory containing RxInfer output JSON files
        model_name: Optional filter by model name

    Returns:
        List of parsed result dicts with added file metadata
    """
    results = []

    if not output_dir.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return results

    # Find all JSON output files
    json_files = list(output_dir.glob("**/*_rxinfer*.json")) + \
                 list(output_dir.glob("**/*rxinfer*.json")) + \
                 list(output_dir.glob("**/rxinfer_output*.json"))

    for json_file in json_files:
        parsed = parse_rxinfer_output(json_file)
        if parsed is None:
            continue

        # Filter by model name if specified
        if model_name and parsed.get("model_name") != model_name:
            continue

        # Add file metadata
        parsed["source_file"] = str(json_file)
        parsed["convergence_metrics"] = extract_convergence_metrics(parsed)
        parsed["posterior_summary"] = summarize_posteriors(parsed)

        results.append(parsed)
        logger.info(f"Parsed RxInfer results from {json_file.name}: "
                   f"converged={parsed['converged']}, "
                   f"iterations={parsed['iterations']}")

    return results


def format_rxinfer_report(results: List[Dict[str, Any]]) -> str:
    """
    Format RxInfer results as a markdown report.

    Args:
        results: List of parsed results from collect_rxinfer_results()

    Returns:
        Markdown-formatted report string
    """
    if not results:
        return "# RxInfer.jl Results\n\nNo results found.\n"

    lines = ["# RxInfer.jl Inference Results\n",
             f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
             f"**Models analyzed**: {len(results)}\n\n"]

    for r in results:
        model = r.get("model_name", "unknown")
        conv = r.get("convergence_metrics", {})

        lines.append(f"## {model}\n")
        lines.append(f"- **Converged**: {'Yes' if conv.get('converged') else 'No'}")
        lines.append(f"- **Iterations**: {conv.get('total_iterations', 'N/A')}")

        if conv.get("final_free_energy") is not None:
            lines.append(f"- **Final Free Energy**: {conv['final_free_energy']:.4f}")
        if conv.get("iterations_to_convergence") is not None:
            lines.append(f"- **Iterations to convergence**: {conv['iterations_to_convergence']}")

        # Posterior summary
        post_summary = r.get("posterior_summary", {})
        if post_summary:
            lines.append("\n### Posterior Estimates\n")
            lines.append("| Variable | Type | Mean/Value |")
            lines.append("|----------|------|------------|")
            for var, stats in list(post_summary.items())[:10]:  # Top 10
                mean = stats.get("mean", "N/A")
                if isinstance(mean, list):
                    mean_str = f"[{', '.join(f'{v:.3f}' for v in mean[:4])}{'...' if len(mean) > 4 else ''}]"
                else:
                    mean_str = f"{mean:.4f}" if isinstance(mean, float) else str(mean)
                lines.append(f"| {var} | {stats.get('type', 'unknown')} | {mean_str} |")

        lines.append("")

    return "\n".join(lines)
