#!/usr/bin/env python3
"""
Research Processor module for GNN Processing Pipeline.

Generates experimental hypotheses via rule-based static analysis (default)
and optional LLM-powered semantic analysis when an LLM provider is configured.

The FEATURES['fallback_mode'] flag indicates the module operates without
LLM dependencies — this is expected behavior, not a limitation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import re
import json

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

logger = logging.getLogger(__name__)

FEATURES = {
    "fallback_mode": True,  # Rule-based mode; LLM used opportunistically when available
    "model_family_detection": True,
    "dimension_aware_analysis": True,
    "llm_hypothesis_generation": False,  # Set True if LLM provider configured
}


def detect_model_family(content: str) -> str:
    """
    Detect the Active Inference model family from GNN content.

    Returns one of: 'pomdp', 'hmm', 'hierarchical', 'continuous', 'factor_graph', 'unknown'
    """
    content_lower = content.lower()

    # Check GNNSection header first
    section_match = re.search(r'## GNNSection\s*\n\s*(\S+)', content)
    if section_match:
        section = section_match.group(1).lower()
        if 'pomdp' in section:
            return 'pomdp'
        elif 'hmm' in section:
            return 'hmm'
        elif 'hierarchical' in section:
            return 'hierarchical'
        elif 'continuous' in section:
            return 'continuous'
        elif 'factor' in section:
            return 'factor_graph'

    # Detect from state space variables
    has_A = bool(re.search(r'^A\s*\[', content, re.MULTILINE))
    has_B = bool(re.search(r'^B\s*\[', content, re.MULTILINE))
    has_C = bool(re.search(r'^C\s*\[', content, re.MULTILINE))
    has_D = bool(re.search(r'^D\s*\[', content, re.MULTILINE))
    has_pi = bool(re.search(r'^π\s*\[|^pi\s*\[', content, re.MULTILINE))
    has_G = bool(re.search(r'^G\s*\[', content, re.MULTILINE))

    # Hierarchical: multiple levels of A/B or explicit nesting
    level_count = len(re.findall(r'level\d|layer\d|hierarchical', content_lower))
    if level_count > 1:
        return 'hierarchical'

    # Full POMDP: has A, B, C, D and policy
    if has_A and has_B and has_pi and has_G:
        return 'pomdp'

    # HMM: A and B but no policy/action selection
    if has_A and has_B and not has_pi:
        return 'hmm'

    # Continuous: no discrete B matrix, uses continuous dynamics
    if 'continuous' in content_lower or 'gaussian' in content_lower:
        return 'continuous'

    # Factor graph
    if 'factor' in content_lower and has_A:
        return 'factor_graph'

    return 'unknown'


def extract_state_space_dims(content: str) -> Dict[str, List[int]]:
    """
    Extract variable dimensions from GNN StateSpaceBlock.
    Only extracts integer dimensions (not symbolic like pi).
    """
    dims = {}
    pattern = r'^([A-Za-z_][A-Za-z0-9_\']*)\s*\[([^\]]+)\]'

    in_state_space = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## StateSpaceBlock"):
            in_state_space = True
            continue
        elif stripped.startswith("##") and in_state_space:
            in_state_space = False
            continue

        if in_state_space and not stripped.startswith('#'):
            match = re.match(pattern, stripped)
            if match:
                var_name = match.group(1)
                dim_str = match.group(2)
                var_dims = []
                for part in dim_str.split(","):
                    part = part.strip()
                    if part.startswith("type="):
                        continue
                    try:
                        var_dims.append(int(part))
                    except ValueError:
                        pass
                if var_dims:
                    dims[var_name] = var_dims

    return dims


def count_connections(content: str) -> Dict[str, int]:
    """Count directed and undirected connections in the Connections section."""
    in_connections = False
    directed = 0
    undirected = 0

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Connections"):
            in_connections = True
            continue
        elif stripped.startswith("##") and in_connections:
            in_connections = False
            continue

        if in_connections and not stripped.startswith('#') and stripped:
            directed += len(re.findall(r'>', stripped))
            undirected += len(re.findall(r'(?<![>])-(?![>])', stripped))

    return {"directed": directed, "undirected": undirected, "total": directed + undirected}


def generate_rule_based_hypotheses(
    content: str,
    model_family: str,
    dims: Dict[str, List[int]],
    connections: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Generate research hypotheses via rule-based static analysis.

    Rules are domain-specific to Active Inference / generative model research.
    """
    hypotheses = []

    # Rule 1: High-Dimensionality -- only flag actual matrix variables, not arbitrary integers
    max_dim = 0
    large_vars = []
    for name, var_dims in dims.items():
        total = 1
        for d in var_dims:
            total *= d
        if total > 100:
            large_vars.append((name, var_dims, total))
            max_dim = max(max_dim, total)

    if max_dim > 1000:
        hypotheses.append({
            "type": "dimensionality_reduction",
            "description": f"Apply structured mean-field or amortized inference for large variables: {[v[0] for v in large_vars]}",
            "rationale": f"Variables {[f'{v[0]}{v[1]}' for v in large_vars[:3]]} have high parameter counts. "
                        f"Structured approximations can maintain accuracy while reducing computational cost.",
            "priority": "high"
        })
    elif max_dim > 100:
        hypotheses.append({
            "type": "dimensionality_check",
            "description": "Consider whether full joint inference is necessary for all state factors",
            "rationale": f"Moderate dimensionality detected (max {max_dim} elements). "
                        f"Factored inference may be more efficient.",
            "priority": "medium"
        })

    # Rule 2: Sparse Connectivity
    # Use structured dims when available; fall back to `name:` counting for non-standard GNN formats
    total_vars = len(dims)
    if total_vars == 0:
        total_vars = len(re.findall(r'name:', content))
    total_conns = connections.get("total", 0)
    if total_conns == 0:
        total_conns = len(re.findall(r'->', content))
    if total_vars > 2 and total_conns > 0:
        density = total_conns / (total_vars * (total_vars - 1))
        if density < 0.3:
            hypotheses.append({
                "type": "connectivity_enrichment",
                "description": "Investigate potential missing causal links between model components",
                "rationale": f"Graph density is {density:.2f} ({total_conns} connections, {total_vars} variables). "
                            f"Sparse graphs may miss important dependencies.",
                "priority": "medium"
            })

    # Rule 3: Model-family-specific suggestions
    if model_family == 'pomdp':
        # Check for precision parameters
        has_precision = bool(re.search(r'precision|omega|gamma|alpha', content, re.IGNORECASE))
        if not has_precision:
            hypotheses.append({
                "type": "precision_modulation",
                "description": "Add precision parameters to modulate sensory and policy uncertainty",
                "rationale": "The model lacks precision parameters. Active Inference with precision weighting "
                            "better captures attentional modulation and epistemic confidence.",
                "priority": "high"
            })

        # Check for planning horizon
        horizon_match = re.search(r'ModelTimeHorizon\s*=\s*(\w+)', content)
        if horizon_match and horizon_match.group(1) == 'Unbounded':
            hypotheses.append({
                "type": "planning_horizon",
                "description": "Define explicit planning horizon T for tractable policy inference",
                "rationale": "Unbounded time horizon requires truncation for policy selection. "
                            "Setting T=3-5 enables efficient Expected Free Energy computation.",
                "priority": "medium"
            })

        # Check for learning (parameter updating)
        has_learning = bool(re.search(r'learning|update.*param|concentration|dirichlet', content, re.IGNORECASE))
        if not has_learning:
            hypotheses.append({
                "type": "parameter_learning",
                "description": "Add Dirichlet concentration parameters for online model learning",
                "rationale": "Static A and B matrices cannot adapt. Adding concentration parameters "
                            "(a, b) enables Bayesian learning from experience.",
                "priority": "medium"
            })

    elif model_family == 'hmm':
        hypotheses.append({
            "type": "upgrade_to_pomdp",
            "description": "Consider extending HMM to full POMDP with action-dependent transitions",
            "rationale": "HMMs have no action selection. Adding a B[states,states,actions] tensor "
                        "and preference vector C enables Active Inference policy optimization.",
            "priority": "low"
        })

    elif model_family == 'hierarchical':
        hypotheses.append({
            "type": "message_passing",
            "description": "Verify belief propagation schedule across hierarchical levels",
            "rationale": "Hierarchical models require careful message-passing ordering. "
                        "Top-down predictions must precede bottom-up updates.",
            "priority": "high"
        })

    # Rule 4: Missing ontology annotation
    has_ontology = "## ActInfOntologyAnnotation" in content
    if not has_ontology:
        hypotheses.append({
            "type": "ontology_annotation",
            "description": "Add ActInfOntologyAnnotation section for variable semantic labeling",
            "rationale": "Ontology annotations enable automatic cross-framework compatibility "
                        "checks and improve model documentation.",
            "priority": "low"
        })

    # Rule 5: Missing initial parameterization
    has_params = "## InitialParameterization" in content
    if not has_params and dims:
        hypotheses.append({
            "type": "parameterization",
            "description": "Add InitialParameterization section with concrete matrix values",
            "rationale": "Without initial parameters, rendering steps cannot produce executable code. "
                        "Define A, B, C, D values for complete model specification.",
            "priority": "high"
        })

    return hypotheses


async def _generate_llm_hypotheses(
    content: str,
    model_family: str,
    dims: Dict[str, List[int]],
    logger: logging.Logger
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate hypotheses using LLM when available. Returns None on any failure.
    """
    try:
        from llm.llm_processor import initialize_global_processor
        from llm.providers.base_provider import LLMMessage
        processor = await initialize_global_processor()
        if not processor:
            return None
    except Exception:
        return None

    dim_summary = ", ".join(f"{k}{v}" for k, v in list(dims.items())[:6])
    prompt = f"""You are an Active Inference research assistant analyzing a GNN model specification.

Model family: {model_family}
Key variables: {dim_summary}

Based on this Active Inference generative model, generate 2-3 specific, actionable research hypotheses.
Focus on: model extensions, inference improvements, or experimental validations.

Respond with a JSON array of hypothesis objects, each with:
- type: short identifier (snake_case)
- description: one sentence action (20 words or less)
- rationale: explanation referencing specific model structure (50 words or less)
- priority: "high", "medium", or "low"

JSON only, no prose:"""

    try:
        messages = [LLMMessage(role="user", content=prompt)]
        response = await processor.get_response(
            messages=messages,
            model_name="gemma3:4b",
            max_tokens=800
        )
        # Parse JSON response
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except Exception as e:
        logger.debug(f"LLM hypothesis generation failed: {e}")
        return None


def process_research(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process research for GNN files.

    Generates deterministic experimental hypotheses based on static analysis rules.
    LLM-powered hypotheses are added opportunistically when an LLM provider is available.
    """
    logger = logging.getLogger("research")

    try:
        log_step_start(logger, "Processing research")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "processed_files": 0,
            "success": True,
            "hypotheses_generated": [],
            "model_families_detected": {},
            "errors": [],
            "analysis_mode": "rule_based"
        }

        gnn_files = list(target_dir.glob("*.md"))
        results["processed_files"] = len(gnn_files)

        for gnn_file in gnn_files:
            try:
                content = gnn_file.read_text()

                # Detect model family
                model_family = detect_model_family(content)
                results["model_families_detected"][gnn_file.name] = model_family

                if verbose:
                    logger.info(f"{gnn_file.name}: detected as '{model_family}' model")

                # Extract structured dimensions (not naive integer extraction)
                dims = extract_state_space_dims(content)
                connections = count_connections(content)

                # Rule-based hypotheses (always available)
                hypotheses = generate_rule_based_hypotheses(content, model_family, dims, connections)

                # Attempt LLM-powered hypotheses (opportunistic)
                llm_hypotheses = None
                if FEATURES.get("llm_hypothesis_generation"):
                    try:
                        import asyncio
                        llm_hypotheses = asyncio.run(
                            _generate_llm_hypotheses(content, model_family, dims, logger)
                        )
                        if llm_hypotheses:
                            results["analysis_mode"] = "llm_enhanced"
                            # Merge: LLM hypotheses first (more specific), then rules
                            hypotheses = llm_hypotheses + [
                                h for h in hypotheses
                                if not any(lh.get("type") == h.get("type") for lh in llm_hypotheses)
                            ]
                    except Exception as e:
                        logger.debug(f"LLM hypotheses unavailable for {gnn_file.name}: {e}")

                if hypotheses:
                    results["hypotheses_generated"].append({
                        "file": gnn_file.name,
                        "model_family": model_family,
                        "dimension_count": len(dims),
                        "hypotheses": hypotheses
                    })

            except Exception as e:
                logger.warning(f"Could not generate hypotheses for {gnn_file}: {e}")
                results["errors"].append({"file": str(gnn_file.name), "error": str(e)})

        # Save results
        results_file = results_dir / "research_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate markdown report
        report_lines = ["# Research Hypotheses Report\n"]
        report_lines.append(f"**Analysis mode**: {results['analysis_mode']}\n\n")

        for entry in results["hypotheses_generated"]:
            report_lines.append(f"## {entry['file']} ({entry['model_family']} model)\n")

            # Group by priority
            high = [h for h in entry['hypotheses'] if h.get('priority') == 'high']
            medium = [h for h in entry['hypotheses'] if h.get('priority') == 'medium']
            low = [h for h in entry['hypotheses'] if h.get('priority') == 'low']

            for priority_label, hyps in [("High Priority", high), ("Medium Priority", medium), ("Low Priority", low)]:
                if hyps:
                    report_lines.append(f"### {priority_label}\n")
                    for h in hyps:
                        report_lines.append(f"- **{h['type']}**: {h['description']}\n")
                        report_lines.append(f"  - *Rationale*: {h['rationale']}\n")
            report_lines.append("\n")

        (results_dir / "research_report.md").write_text("".join(report_lines))

        log_step_success(logger, "Research processing completed successfully")
        return results["success"]

    except Exception as e:
        log_step_error(logger, "Research processing failed", {"error": str(e)})
        return False
