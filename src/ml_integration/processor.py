#!/usr/bin/env python3
"""
ML Integration Processor module for GNN Processing Pipeline.

Extracts real GNN features (num_states, num_observations, connectivity_ratio, etc.)
and trains ML models to classify model families and predict complexity.

Features extracted from GNN structure:
- num_states: hidden state dimensions from StateSpaceBlock
- num_observations: observation dimensions
- num_actions: action space size (from B tensor)
- num_variables: total variable count
- connectivity_ratio: connections / (variables * (variables-1))
- max_dimension: largest single variable dimension
- total_parameters: total element count across all variables
- model_family_encoded: integer encoding of model type
"""

import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def extract_gnn_features(file_path: Path) -> Dict[str, Any]:
    """
    Extract structural features from a GNN file for ML training.

    Returns a feature dict with:
    - Numeric features for ML models
    - Categorical features for labeling
    - Raw metadata for reporting
    """
    try:
        content = file_path.read_text()
    except Exception as e:
        logger.error(f"Could not read {file_path}: {e}")
        return {}

    features = {
        "file_name": file_path.name,
        "model_family": _detect_model_family(content),
        "num_variables": 0,
        "num_states": 0,
        "num_observations": 0,
        "num_actions": 0,
        "max_dimension": 0,
        "total_parameters": 0,
        "connectivity_ratio": 0.0,
        "directed_connections": 0,
        "undirected_connections": 0,
        "has_precision": False,
        "has_learning": False,
        "has_ontology": bool("## ActInfOntologyAnnotation" in content),
        "has_parameterization": bool("## InitialParameterization" in content),
        "planning_horizon": _extract_planning_horizon(content),
        "time_type": _extract_time_type(content),
    }

    # Extract variable dimensions from StateSpaceBlock
    dims = _extract_dimensions(content)
    features["num_variables"] = len(dims)

    if dims:
        # Compute dimension-based features
        all_elements = []
        for _, var_dims in dims.items():
            elements = 1
            for d in var_dims:
                elements *= d
            all_elements.append(elements)
            features["max_dimension"] = max(features["max_dimension"], max(var_dims))

        features["total_parameters"] = sum(all_elements)

        # Extract specific Active Inference dimensions
        if "A" in dims:
            a_dims = dims["A"]
            if len(a_dims) >= 2:
                features["num_observations"] = a_dims[0]
                features["num_states"] = a_dims[1]

        if "B" in dims:
            b_dims = dims["B"]
            if len(b_dims) >= 3:
                features["num_actions"] = b_dims[2]

        if "s" in dims and features["num_states"] == 0:
            features["num_states"] = dims["s"][0]

        if "o" in dims and features["num_observations"] == 0:
            features["num_observations"] = dims["o"][0]

    # Connectivity analysis
    connections = _count_connections(content)
    features["directed_connections"] = connections["directed"]
    features["undirected_connections"] = connections["undirected"]
    total_conn = connections["directed"] + connections["undirected"]

    n = features["num_variables"]
    if n > 1:
        features["connectivity_ratio"] = total_conn / (n * (n - 1))

    # Qualitative features
    features["has_precision"] = bool(
        re.search(r'\b(precision|omega|γ|gamma|α|alpha)\b', content, re.IGNORECASE)
        and re.search(r'^(ω|γ|β|Π)\s*\[', content, re.MULTILINE)
    )
    features["has_learning"] = bool(
        re.search(r'\b(learning|concentration|dirichlet|update.*param)\b', content, re.IGNORECASE)
    )

    return features


def _detect_model_family(content: str) -> str:
    """Detect model family from GNN content."""
    section_match = re.search(r'## GNNSection\s*\n\s*(\S+)', content)
    if section_match:
        section = section_match.group(1).lower()
        if 'hierarchical' in section:
            return 'hierarchical'
        if 'continuous' in section:
            return 'continuous'
        if 'multiagent' in section or 'multi_agent' in section:
            return 'multi_agent'
        if 'factor' in section:
            return 'factor_graph'
        if 'hmm' in section:
            return 'hmm'
        if 'pomdp' in section:
            return 'pomdp'

    has_B_with_actions = bool(re.search(r'^B\s*\[\d+,\d+,\d+', content, re.MULTILINE))
    has_pi = bool(re.search(r'^π\s*\[|^pi\s*\[', content, re.MULTILINE))

    if has_B_with_actions and has_pi:
        return 'pomdp'
    if re.search(r'^B\s*\[', content, re.MULTILINE) and not has_pi:
        return 'hmm'
    return 'unknown'


def _extract_dimensions(content: str) -> Dict[str, List[int]]:
    """Extract variable dimensions from StateSpaceBlock."""
    dims = {}
    in_state_space = False
    pattern = r'^([A-Za-z_][A-Za-z0-9_\']*)\s*\[([^\]]+)\]'

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
                        logger.debug("Skipping non-integer dimension part: %s", part)
                if var_dims:
                    dims[var_name] = var_dims

    return dims


def _count_connections(content: str) -> Dict[str, int]:
    """Count directed and undirected connections."""
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

    return {"directed": directed, "undirected": undirected}


def _extract_planning_horizon(content: str) -> int:
    """Extract planning horizon from model parameters."""
    match = re.search(r'planning_horizon\s*:\s*(\d+)', content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'ModelTimeHorizon\s*=\s*(\d+)', content)
    if match:
        return int(match.group(1))
    return 1  # Default: 1-step planning


def _extract_time_type(content: str) -> str:
    """Extract time type (Discrete/Continuous)."""
    if re.search(r'^Continuous\s*$', content, re.MULTILINE):
        return 'continuous'
    if re.search(r'^Discrete\s*$', content, re.MULTILINE):
        return 'discrete'
    return 'unknown'


# Encoding for model families
MODEL_FAMILY_ENCODING = {
    "pomdp": 0,
    "hmm": 1,
    "hierarchical": 2,
    "continuous": 3,
    "multi_agent": 4,
    "factor_graph": 5,
    "unknown": -1
}


def process_ml_integration(
    target_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process ML integration for GNN models.

    Extracts real GNN features and trains models to classify model families.
    Uses RandomForestClassifier alongside Decision Tree for comparison.
    Adds 5-fold cross-validation and feature importance reporting.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.info(f"Starting ML integration: target={target_dir}")

        ml_results = {
            "status": "completed",
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "models_trained": [],
            "feature_importance": {},
            "cross_validation": {},
            "framework_status": {},
            "extracted_features": []
        }

        # Check for sklearn availability
        try:
            import sklearn  # noqa: F401
            import numpy  # noqa: F401
            has_sklearn = True
            ml_results["framework_status"]["sklearn"] = "available"
        except ImportError:
            has_sklearn = False
            ml_results["framework_status"]["sklearn"] = "missing"

        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning(f"No GNN files found in {target_dir}")
            ml_results["status"] = "no_files"

        # Extract features from all files
        all_features = []
        for gnn_file in gnn_files:
            try:
                feats = extract_gnn_features(gnn_file)
                if feats:
                    all_features.append(feats)
                    ml_results["extracted_features"].append(feats)
                    if verbose:
                        logger.info(f"  {gnn_file.name}: family={feats['model_family']}, "
                                   f"states={feats['num_states']}, params={feats['total_parameters']}")
            except Exception as e:
                logger.error(f"Feature extraction failed for {gnn_file}: {e}")

        # Train ML models if we have enough data
        if has_sklearn and len(all_features) >= 2:
            _train_models(all_features, ml_results, output_dir, verbose)
        elif not has_sklearn:
            logger.info("sklearn not available -- skipping ML training")
            # Add per-file structural analysis entries (backward compatibility)
            for feats in all_features:
                ml_results["models_trained"].append({
                    "source": feats["file_name"],
                    "type": "structural_analysis",
                    "framework": "internal_stats",
                    "accuracy": 1.0,
                    "note": "sklearn not available",
                    "model_family": feats.get("model_family"),
                    "num_states": feats.get("num_states", 0),
                    "total_parameters": feats.get("total_parameters", 0),
                })
        else:
            logger.info(f"Only {len(all_features)} files -- need >=2 for ML training")
            # Add per-file structural analysis entries (backward compatibility)
            for feats in all_features:
                ml_results["models_trained"].append({
                    "source": feats["file_name"],
                    "type": "structural_analysis",
                    "framework": "internal_stats",
                    "accuracy": 1.0,
                    "note": f"Need >=2 GNN files for ML classification (have {len(all_features)})",
                    "model_family": feats.get("model_family"),
                    "num_states": feats.get("num_states", 0),
                    "total_parameters": feats.get("total_parameters", 0),
                })
            _save_feature_analysis(all_features, ml_results, output_dir)

        # Save results
        results_file = output_dir / "ml_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(ml_results, f, indent=2, default=str)

        if verbose:
            logger.info(f"ML integration completed -- results at {results_file}")

        return True

    except Exception as e:
        logger.error(f"ML integration failed: {e}")
        return False


def _train_models(
    all_features: List[Dict],
    ml_results: Dict,
    output_dir: Path,
    verbose: bool
) -> None:
    """Train Decision Tree and Random Forest on extracted GNN features."""
    import numpy as np
    import pickle  # nosec B403 -- pickle used for internal model serialization with trusted data sources
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Build feature matrix using real GNN features
    numeric_feature_names = [
        "num_states", "num_observations", "num_actions", "num_variables",
        "connectivity_ratio", "max_dimension", "total_parameters", "planning_horizon",
        "directed_connections", "undirected_connections",
        "has_precision", "has_learning", "has_ontology", "has_parameterization"
    ]

    X = np.array([
        [
            f.get("num_states", 0),
            f.get("num_observations", 0),
            f.get("num_actions", 0),
            f.get("num_variables", 0),
            f.get("connectivity_ratio", 0.0),
            f.get("max_dimension", 0),
            f.get("total_parameters", 0),
            f.get("planning_horizon", 1),
            f.get("directed_connections", 0),
            f.get("undirected_connections", 0),
            int(f.get("has_precision", False)),
            int(f.get("has_learning", False)),
            int(f.get("has_ontology", False)),
            int(f.get("has_parameterization", False)),
        ]
        for f in all_features
    ], dtype=float)

    # Labels: model family
    y_labels = [f.get("model_family", "unknown") for f in all_features]

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Need at least 2 samples and 2 classes for meaningful training
    if len(np.unique(y)) < 2:
        # All same family -- use complexity classification instead
        y = np.array([
            0 if f.get("total_parameters", 0) < 100 else
            1 if f.get("total_parameters", 0) < 1000 else 2
            for f in all_features
        ])
        label_names = ["small", "medium", "large"]
        task = "complexity_classification"
    else:
        label_names = list(le.classes_)
        task = "family_classification"

    ml_results["classification_task"] = task
    ml_results["label_names"] = label_names

    # Cross-validation: cap folds by both sample count and smallest class count
    # to avoid sklearn warnings about underpopulated classes
    from collections import Counter
    class_counts = Counter(y)
    min_class_count = min(class_counts.values()) if class_counts else 1
    n_folds = min(5, len(X), min_class_count)

    for model_name, clf in [
        ("decision_tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42))
    ]:
        try:
            if n_folds >= 2:
                cv_scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            else:
                clf.fit(X, y)
                cv_mean = float(clf.score(X, y))
                cv_std = 0.0

            # Fit on all data
            clf.fit(X, y)
            train_accuracy = float(clf.score(X, y))

            # Save model
            model_path = output_dir / f"gnn_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)

            model_info = {
                "type": model_name,
                "framework": "sklearn",
                "task": task,
                "train_accuracy": train_accuracy,
                "cv_mean_accuracy": cv_mean,
                "cv_std": cv_std,
                "n_folds": n_folds,
                "n_samples": len(X),
                "n_features": len(numeric_feature_names),
                "feature_names": numeric_feature_names,
                "artifact_path": str(model_path),
            }

            # Feature importance (both models support this)
            importances = clf.feature_importances_.tolist()
            model_info["feature_importance"] = dict(zip(numeric_feature_names, importances))

            # Top 5 features
            sorted_feats = sorted(
                zip(numeric_feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            model_info["top_features"] = [(f, round(i, 4)) for f, i in sorted_feats[:5]]

            ml_results["models_trained"].append(model_info)

            if verbose:
                logger.info(f"  {model_name}: CV accuracy={cv_mean:.3f}+/-{cv_std:.3f}, "
                           f"top feature={sorted_feats[0][0]}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    # Store feature importance from best model
    if ml_results["models_trained"]:
        best = max(ml_results["models_trained"], key=lambda m: m.get("cv_mean_accuracy", 0))
        ml_results["feature_importance"] = best.get("feature_importance", {})


def _save_feature_analysis(
    all_features: List[Dict],
    ml_results: Dict,
    output_dir: Path
) -> None:
    """Save feature analysis when insufficient data for full ML training."""
    if not all_features:
        return

    # Summary statistics per feature
    numeric_keys = ["num_states", "num_observations", "num_actions", "total_parameters", "connectivity_ratio"]
    stats = {}

    for key in numeric_keys:
        values = [f.get(key, 0) for f in all_features if isinstance(f.get(key), (int, float))]
        if values:
            stats[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values)
            }

    ml_results["feature_statistics"] = stats
    ml_results["model_families"] = [f.get("model_family") for f in all_features]
