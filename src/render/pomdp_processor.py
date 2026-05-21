#!/usr/bin/env python3
"""
POMDP Processor for Render Module

This module provides specialized processing capabilities for injecting POMDP state spaces
into various rendering implementations (PyMDP, RxInfer, ActiveInference.jl, etc.).
"""

import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import numpy as np

from .framework_registry import get_pomdp_framework_configs
from .pomdp_contract import build_canonical_pomdp_spec

if TYPE_CHECKING:
    from gnn.pomdp_extractor import POMDPStateSpace

logger = logging.getLogger(__name__)


def _normalise_prob_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).flatten()
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0:
        return np.ones(max(vector.shape[0], 1), dtype=np.float64) / max(
            vector.shape[0], 1
        )
    return vector / total


def _normalise_columns(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {out.shape}")
    column_sums = out.sum(axis=0, keepdims=True)
    zero_columns = column_sums <= 0
    column_sums = np.where(zero_columns, 1.0, column_sums)
    out = out / column_sums
    if zero_columns.any():
        rows = out.shape[0]
        for column in np.where(zero_columns.flatten())[0]:
            out[:, column] = 1.0 / rows
    return out


def count_code_metrics(file_path: Path) -> Dict[str, int]:
    """
    Calculate code metrics for a generated file.

    Args:
        file_path: Path to the code file

    Returns:
        Dictionary with lines_of_code, functions, classes counts
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Count non-empty, non-comment lines
        loc = sum(
            1 for line in lines if line.strip() and not line.strip().startswith("#")
        )

        # Count functions (Python: def, Julia: function)
        functions = sum(
            1
            for line in lines
            if line.strip().startswith("def ")
            or line.strip().startswith("function ")
            or "@jit" in line
        )  # JAX decorated functions

        # Count classes (Python: class)
        classes = sum(1 for line in lines if line.strip().startswith("class "))

        return {
            "lines_of_code": loc,
            "total_lines": len(lines),
            "functions": functions,
            "classes": classes,
        }
    except Exception as e:
        logger.warning(f"Could not count code metrics for {file_path}: {e}")
        return {"lines_of_code": 0, "total_lines": 0, "functions": 0, "classes": 0}


class POMDPRenderProcessor:
    """
    Processes POMDP state spaces and injects them into framework-specific renderers.

    Features:
    - Modular injection of POMDP state spaces into renderers
    - Framework-specific output directory management
    - Structured approach to render coordination
    - Validation of POMDP-renderer compatibility
    """

    def __init__(self, base_output_dir: Path) -> None:
        """
        Initialize POMDP render processor.

        Args:
            base_output_dir: Base output directory for all renderers
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)

        self.framework_configs = get_pomdp_framework_configs()

    def process_pomdp_for_all_frameworks(
        self,
        pomdp_space: "POMDPStateSpace",
        gnn_file_path: Optional[Path] = None,
        frameworks: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process POMDP state space for all or specified frameworks.

        Args:
            pomdp_space: Extracted POMDP state space
            gnn_file_path: Original GNN file path (for reference)
            frameworks: List of frameworks to render for (default: all)
            **kwargs: Additional processing options

        Returns:
            Dictionary with processing results for each framework
        """
        if frameworks is None:
            frameworks = list(self.framework_configs.keys())

        results: dict[Any, Any] = {}
        overall_success = True

        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create processing summary
        processing_summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "source_file": str(gnn_file_path) if gnn_file_path else None,
            "model_name": pomdp_space.model_name,
            "pomdp_dimensions": {
                "num_states": pomdp_space.num_states,
                "num_observations": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
            },
            "frameworks_requested": frameworks,
            "frameworks_processed": [],
            "frameworks_failed": [],
        }

        self.logger.info(
            f"Processing POMDP '{pomdp_space.model_name}' for frameworks: {frameworks}"
        )

        for framework in frameworks:
            try:
                self.logger.info(f"Processing framework: {framework}")
                framework_result = self._process_single_framework(
                    pomdp_space, framework, gnn_file_path, **kwargs
                )

                results[framework] = framework_result

                if framework_result["success"]:
                    processing_summary["frameworks_processed"].append(framework)
                    self.logger.info(f"✅ {framework}: {framework_result['message']}")
                else:
                    processing_summary["frameworks_failed"].append(framework)
                    self.logger.error(f"❌ {framework}: {framework_result['message']}")

            except Exception as e:
                error_msg = f"Unexpected error processing {framework}: {e}"
                self.logger.error(error_msg)
                results[framework] = {
                    "success": False,
                    "message": error_msg,
                    "output_files": [],
                    "warnings": [],
                }
                processing_summary["frameworks_failed"].append(framework)

        total_frameworks = len(frameworks)
        successful_frameworks = len(processing_summary["frameworks_processed"])
        success_rate = (
            successful_frameworks / total_frameworks if total_frameworks > 0 else 0
        )
        overall_success = successful_frameworks == total_frameworks

        if not overall_success:
            self.logger.warning(
                "Framework rendering incomplete: %d/%d succeeded (%.1f%%)",
                successful_frameworks,
                total_frameworks,
                success_rate * 100,
            )

        # Save processing summary
        summary_file = self.base_output_dir / "processing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(processing_summary, f, indent=2)

        return {
            "overall_success": overall_success,
            "framework_results": results,
            "summary_file": str(summary_file),
            "output_directory": str(self.base_output_dir),
        }

    def _process_single_framework(
        self,
        pomdp_space: "POMDPStateSpace",
        framework: str,
        gnn_file_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process POMDP state space for a single framework.

        Args:
            pomdp_space: POMDP state space data
            framework: Target framework name
            gnn_file_path: Original GNN file path
            **kwargs: Additional options

        Returns:
            Processing result dictionary
        """
        if framework not in self.framework_configs:
            return {
                "success": False,
                "message": f"Unknown framework: {framework}",
                "output_files": [],
                "warnings": [],
            }

        config = self.framework_configs[framework]

        # Validate POMDP compatibility with framework
        validation_result = self._validate_pomdp_framework_compatibility(
            pomdp_space, framework
        )
        if not validation_result["compatible"]:
            return {
                "success": False,
                "message": f"POMDP not compatible with {framework}: {validation_result['reason']}",
                "output_files": [],
                "warnings": validation_result.get("warnings", []),
            }

        # Create framework-specific output directory
        framework_output_dir = self.base_output_dir / str(config["output_subdir"])
        framework_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert POMDP to GNN spec format expected by renderers
        gnn_spec = self._pomdp_to_gnn_spec(pomdp_space, **kwargs)

        # Get framework-specific renderer
        try:
            renderer_result = self._call_framework_renderer(
                framework, gnn_spec, framework_output_dir, **kwargs
            )

            if renderer_result["success"]:
                # Create framework-specific documentation
                self._create_framework_documentation(
                    framework, pomdp_space, framework_output_dir, renderer_result
                )

                # Calculate code metrics for generated files
                code_metrics: dict[Any, Any] = {}
                for output_file in renderer_result.get("artifacts", []):
                    file_path = Path(output_file)
                    if file_path.exists():
                        code_metrics = count_code_metrics(file_path)
                        break  # Use first file's metrics

                return {
                    "success": True,
                    "message": renderer_result["message"],
                    "output_files": renderer_result.get("artifacts", []),
                    "output_directory": str(framework_output_dir),
                    "warnings": validation_result.get("warnings", []),
                    "code_metrics": code_metrics,
                }
            else:
                return {
                    "success": False,
                    "message": renderer_result["message"],
                    "output_files": [],
                    "warnings": validation_result.get("warnings", []),
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Framework renderer failed: {e}",
                "output_files": [],
                "warnings": validation_result.get("warnings", []),
            }

    def _validate_pomdp_framework_compatibility(
        self, pomdp_space: "POMDPStateSpace", framework: str
    ) -> Dict[str, Any]:
        """
        Validate that POMDP is compatible with target framework.

        Args:
            pomdp_space: POMDP state space data
            framework: Target framework name

        Returns:
            Validation result dictionary
        """
        config = self.framework_configs[framework]
        warnings: list[Any] = []

        # Check required matrices are present, allowing factored matrices when
        # they can be composed into a canonical execution contract.
        missing_matrices: list[Any] = []
        required_matrices = cast(list[str], config["requires_matrices"])
        for required_matrix in required_matrices:
            if not self._has_matrix_or_factored_matrix(pomdp_space, required_matrix):
                missing_matrices.append(required_matrix)

        if missing_matrices:
            return {
                "compatible": False,
                "reason": f"Missing required matrices: {missing_matrices}",
                "warnings": warnings,
            }

        if framework == "pymdp":
            try:
                self._build_canonical_initialparameterization(pomdp_space)
            except ValueError as exc:
                return {"compatible": False, "reason": str(exc), "warnings": warnings}

        # Framework-specific checks
        if framework == "rxinfer" and not config["supports_multi_modality"]:
            if pomdp_space.num_observations > 1:  # This is a simplistic check
                warnings.append(f"{framework} has limited multi-modality support")

        # Check dimension limits
        max_reasonable_dim = 100  # Reasonable limit for most frameworks
        if (
            pomdp_space.num_states > max_reasonable_dim
            or pomdp_space.num_observations > max_reasonable_dim
            or pomdp_space.num_actions > max_reasonable_dim
        ):
            warnings.append("Large state spaces may cause performance issues")

        return {"compatible": True, "reason": None, "warnings": warnings}

    def _has_matrix_or_factored_matrix(
        self, pomdp_space: "POMDPStateSpace", matrix_name: str
    ) -> bool:
        matrices = getattr(pomdp_space, "matrices", None) or {}
        if matrix_name in matrices:
            return True
        if matrix_name == "B" and self._time_indexed_transition_key(matrices):
            return True
        return any(key.startswith(f"{matrix_name}_") for key in matrices)

    def _time_indexed_transition_key(self, matrices: Dict[str, Any]) -> Optional[str]:
        """Return the single supported time-indexed transition tensor key."""
        if "B_t" in matrices:
            return "B_t"
        time_keys = sorted(
            key for key in matrices if key.startswith("B_t") and key[3:].isdigit()
        )
        if len(time_keys) == 1:
            return time_keys[0]
        return None

    def _build_canonical_initialparameterization(
        self,
        pomdp_space: "POMDPStateSpace",
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Build the strict canonical A/B/C/D/E contract used by renderers."""
        matrices = getattr(pomdp_space, "matrices", None) or {}
        provenance = dict(getattr(pomdp_space, "matrix_provenance", None) or {})
        has_canonical = all(key in matrices for key in ("A", "B", "C", "D"))

        if has_canonical:
            initial = {key: matrices[key] for key in ("A", "B", "C", "D")}
            if "E" in matrices:
                initial["E"] = matrices["E"]
            gnn_spec = {
                "initialparameterization": initial,
                "model_parameters": getattr(pomdp_space, "model_parameters", None)
                or {},
                "num_states": pomdp_space.num_states,
                "num_observations": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
                "matrix_provenance": provenance,
            }
            canonical = build_canonical_pomdp_spec(gnn_spec)
            return (
                canonical["initialparameterization"],
                canonical["matrix_provenance"],
            )

        time_indexed_b = self._time_indexed_transition_key(matrices)
        if time_indexed_b and all(key in matrices for key in ("A", "C", "D")):
            b_tensor = self._canonicalise_time_indexed_B(
                matrices[time_indexed_b],
                max(1, int(getattr(pomdp_space, "num_actions", 1))),
            )
            initial = {
                "A": matrices["A"],
                "B": b_tensor.tolist(),
                "C": matrices["C"],
                "D": matrices["D"],
            }
            if "E" in matrices:
                initial["E"] = matrices["E"]
            provenance["B"] = {
                "source": "time_indexed_transition_projection",
                "source_key": time_indexed_b,
                "shape": list(b_tensor.shape),
                "derived": True,
                "reason": "PyMDP static transition contract uses the declared B_t tensor for execution",
            }
            gnn_spec = {
                "initialparameterization": initial,
                "model_parameters": getattr(pomdp_space, "model_parameters", None)
                or {},
                "num_states": pomdp_space.num_states,
                "num_observations": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
                "matrix_provenance": provenance,
            }
            canonical = build_canonical_pomdp_spec(gnn_spec)
            return (
                canonical["initialparameterization"],
                canonical["matrix_provenance"],
            )

        joint, joint_provenance = self._compose_factored_pomdp(pomdp_space)
        initial = {
            "A": joint["A"],
            "B": joint["B"],
            "C": joint["C"],
            "D": joint["D"],
        }
        if "E" in matrices:
            initial["E"] = matrices["E"]
        provenance.update(joint_provenance)
        gnn_spec = {
            "initialparameterization": initial,
            "model_parameters": getattr(pomdp_space, "model_parameters", None) or {},
            "num_states": len(initial["D"]),
            "num_observations": len(initial["A"]),
            "num_actions": pomdp_space.num_actions,
            "matrix_provenance": provenance,
        }
        canonical = build_canonical_pomdp_spec(gnn_spec)
        return (
            canonical["initialparameterization"],
            canonical["matrix_provenance"],
        )

    def _compose_factored_pomdp(
        self, pomdp_space: "POMDPStateSpace"
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Compose factored POMDP matrices into a joint PyMDP model without dropping factors."""
        matrices = getattr(pomdp_space, "matrices", None) or {}
        state_factors = [
            factor
            for factor in (getattr(pomdp_space, "state_factors", None) or [])
            if factor.get("size")
        ]
        obs_modalities = [
            modality
            for modality in (getattr(pomdp_space, "observation_modalities", None) or [])
            if modality.get("size")
        ]

        if not state_factors:
            raise ValueError("Factored POMDP is missing state factor metadata")
        if not obs_modalities:
            raise ValueError("Factored POMDP is missing observation modality metadata")

        state_sizes = [int(factor["size"]) for factor in state_factors]
        obs_sizes = [int(modality["size"]) for modality in obs_modalities]
        state_tuples = list(itertools.product(*[range(size) for size in state_sizes]))
        obs_tuples = list(itertools.product(*[range(size) for size in obs_sizes]))
        num_states = len(state_tuples)
        num_obs = len(obs_tuples)
        num_actions = max(1, int(getattr(pomdp_space, "num_actions", 1)))

        a_keys = sorted(key for key in matrices if key.startswith("A_"))
        b_keys = sorted(key for key in matrices if key.startswith("B_"))
        c_keys = sorted(key for key in matrices if key.startswith("C_"))
        d_keys = sorted(key for key in matrices if key.startswith("D_"))

        if not a_keys:
            raise ValueError("Factored POMDP is missing A_* likelihood matrices")
        if not b_keys:
            raise ValueError("Factored POMDP is missing B_* transition matrices")
        if not d_keys:
            raise ValueError("Factored POMDP is missing D_* prior vectors")

        A_joint = np.ones((num_obs, num_states), dtype=np.float64)
        for key in a_keys:
            matrix = np.asarray(matrices[key], dtype=np.float64)
            if matrix.ndim not in (2, 3):
                raise ValueError(
                    f"{key} must be 2D or 3D for PyMDP composition, got shape {matrix.shape}"
                )
            obs_index = self._match_descriptor_index(
                key, obs_modalities, matrix.shape[0]
            )
            state_indices = self._match_state_indices_for_matrix(
                key, state_factors, matrix.shape[1:]
            )
            for obs_flat, obs_tuple in enumerate(obs_tuples):
                for state_flat, state_tuple in enumerate(state_tuples):
                    matrix_index: list[Any] = [obs_tuple[obs_index]]
                    matrix_index.extend(state_tuple[index] for index in state_indices)
                    A_joint[obs_flat, state_flat] *= float(matrix[tuple(matrix_index)])
        A_joint = _normalise_columns(A_joint)

        B_joint = np.ones((num_states, num_states, num_actions), dtype=np.float64)
        for key in b_keys:
            factor_index = self._match_descriptor_index(key, state_factors)
            factor_size = state_sizes[factor_index]
            tensor = self._canonicalise_factored_B(
                matrices[key], factor_size, num_actions
            )
            for action in range(num_actions):
                source_action = action if tensor.shape[2] > 1 else 0
                for prev_flat, prev_tuple in enumerate(state_tuples):
                    for next_flat, next_tuple in enumerate(state_tuples):
                        B_joint[next_flat, prev_flat, action] *= float(
                            tensor[
                                next_tuple[factor_index],
                                prev_tuple[factor_index],
                                source_action,
                            ]
                        )
        for action in range(num_actions):
            B_joint[:, :, action] = _normalise_columns(B_joint[:, :, action])

        if c_keys:
            C_joint = np.zeros(num_obs, dtype=np.float64)
            for key in c_keys:
                vector = np.asarray(matrices[key], dtype=np.float64).flatten()
                obs_index = self._match_descriptor_index(
                    key, obs_modalities, vector.shape[0]
                )
                for obs_flat, obs_tuple in enumerate(obs_tuples):
                    C_joint[obs_flat] += float(vector[obs_tuple[obs_index]])
        elif getattr(pomdp_space, "passive_model", False):
            C_joint = np.zeros(num_obs, dtype=np.float64)
        else:
            raise ValueError("Factored POMDP is missing C_* preference vectors")

        D_joint = np.ones(num_states, dtype=np.float64)
        for key in d_keys:
            vector = _normalise_prob_vector(np.asarray(matrices[key], dtype=np.float64))
            factor_index = self._match_descriptor_index(
                key, state_factors, vector.shape[0]
            )
            for state_flat, state_tuple in enumerate(state_tuples):
                D_joint[state_flat] *= float(vector[state_tuple[factor_index]])
        D_joint = _normalise_prob_vector(D_joint)

        provenance: dict[str, Any] = {
            "A": {
                "source": "factored_joint_composition",
                "source_keys": a_keys,
                "shape": list(A_joint.shape),
                "derived": True,
            },
            "B": {
                "source": "factored_joint_composition",
                "source_keys": b_keys,
                "shape": list(B_joint.shape),
                "derived": True,
            },
            "C": {
                "source": "factored_joint_composition",
                "source_keys": c_keys,
                "shape": list(C_joint.shape),
                "derived": True,
            },
            "D": {
                "source": "factored_joint_composition",
                "source_keys": d_keys,
                "shape": list(D_joint.shape),
                "derived": True,
            },
        }

        return (
            {
                "A": A_joint.tolist(),
                "B": B_joint.tolist(),
                "C": C_joint.tolist(),
                "D": D_joint.tolist(),
            },
            provenance,
        )

    def _canonicalise_time_indexed_B(self, value: Any, num_actions: int) -> np.ndarray:
        """Canonicalize B_t to (next_state, previous_state, action)."""
        raw = np.asarray(value, dtype=np.float64)
        if raw.ndim == 2:
            tensor = raw[:, :, np.newaxis]
        elif raw.ndim == 3:
            if raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
                tensor = raw.transpose(1, 2, 0)
            elif raw.shape[0] == raw.shape[1] and raw.shape[2] in {1, num_actions}:
                tensor = raw
            else:
                raise ValueError(
                    f"B_t must be action-first or canonical 3D tensor, got shape {raw.shape}"
                )
        else:
            raise ValueError(f"B_t must be 2D or 3D, got shape {raw.shape}")
        for action in range(tensor.shape[2]):
            tensor[:, :, action] = _normalise_columns(tensor[:, :, action])
        return tensor

    def _canonicalise_factored_B(
        self, value: Any, factor_size: int, num_actions: int
    ) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if raw.ndim == 2:
            tensor = raw[:, :, np.newaxis]
        elif raw.ndim == 3:
            if raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
                tensor = raw.transpose(2, 1, 0)
            elif raw.shape[0] == 1 and raw.shape[1] == raw.shape[2]:
                tensor = raw.transpose(2, 1, 0)
            elif raw.shape[-1] in {1, num_actions} and raw.shape[0] == raw.shape[1]:
                tensor = raw
            else:
                tensor = raw
        else:
            raise ValueError(f"B factor must be 2D or 3D, got shape {raw.shape}")
        if tensor.shape[0] != factor_size or tensor.shape[1] != factor_size:
            raise ValueError(
                f"B factor shape {tensor.shape} does not match state factor size {factor_size}"
            )
        for action in range(tensor.shape[2]):
            tensor[:, :, action] = _normalise_columns(tensor[:, :, action])
        return tensor

    def _match_descriptor_index(
        self,
        matrix_key: str,
        descriptors: List[Dict[str, Any]],
        required_size: Optional[int] = None,
    ) -> int:
        suffix = (
            matrix_key.split("_", 1)[1].lower()
            if "_" in matrix_key
            else matrix_key.lower()
        )
        matches = [
            index
            for index, descriptor in enumerate(descriptors)
            if suffix in str(descriptor.get("name", "")).lower()
        ]
        if required_size is not None:
            matches = [
                index
                for index in matches
                if int(descriptors[index].get("size") or -1) == int(required_size)
            ] or [
                index
                for index, descriptor in enumerate(descriptors)
                if int(descriptor.get("size") or -1) == int(required_size)
            ]
        if len(matches) == 1:
            return matches[0]
        if not matches and len(descriptors) == 1:
            return 0
        raise ValueError(
            f"Could not map {matrix_key} to descriptors {[d.get('name') for d in descriptors]}"
        )

    def _match_state_indices_for_matrix(
        self,
        matrix_key: str,
        state_factors: List[Dict[str, Any]],
        matrix_state_shape: tuple[int, ...],
    ) -> List[int]:
        if len(matrix_state_shape) == 1:
            return [
                self._match_descriptor_index(
                    matrix_key, state_factors, matrix_state_shape[0]
                )
            ]
        indices: List[int] = []
        used: set[int] = set()
        for size in matrix_state_shape:
            matches = [
                index
                for index, factor in enumerate(state_factors)
                if index not in used and int(factor.get("size") or -1) == int(size)
            ]
            if not matches:
                raise ValueError(
                    f"Could not map {matrix_key} state shape {matrix_state_shape} to state factors"
                )
            index = matches[0]
            used.add(index)
            indices.append(index)
        return indices

    def _pomdp_to_gnn_spec(
        self, pomdp_space: "POMDPStateSpace", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Convert POMDP state space to GNN spec format expected by renderers.

        Args:
            pomdp_space: POMDP state space data
            **kwargs: Additional options like timesteps

        Returns:
            GNN specification dictionary
        """
        # Extract optional config params
        timesteps = kwargs.get("timesteps")
        if timesteps is None and hasattr(pomdp_space, "num_timesteps"):
            timesteps = pomdp_space.num_timesteps

        sim_params = kwargs.get("simulation_params", "{}")
        try:
            parsed_sim_params = (
                json.loads(sim_params) if isinstance(sim_params, str) else sim_params
            )
        except json.JSONDecodeError:
            self.logger.warning(
                f"Invalid simulation_params string: {sim_params}. Using empty dict."
            )
            parsed_sim_params = {}

        initial_parameterization, matrix_provenance = (
            self._build_canonical_initialparameterization(pomdp_space)
        )
        raw_model_parameters = getattr(pomdp_space, "model_parameters", None) or {}
        state_factors = getattr(pomdp_space, "state_factors", None) or []
        observation_modalities = (
            getattr(pomdp_space, "observation_modalities", None) or []
        )
        control_factors = getattr(pomdp_space, "control_factors", None) or []

        gnn_spec: dict[str, Any] = {
            "name": pomdp_space.model_name or "POMDP_Model",
            "model_name": pomdp_space.model_name or "POMDP_Model",
            "description": pomdp_space.model_annotation or "Extracted POMDP model",
            "model_parameters": {
                "num_hidden_states": pomdp_space.num_states,
                "num_obs": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
                "num_state_factors": len(state_factors)
                or raw_model_parameters.get("num_state_factors"),
                "num_modalities": len(observation_modalities)
                or raw_model_parameters.get("num_modalities"),
                "state_factors": state_factors,
                "observation_modalities": observation_modalities,
                "control_factors": control_factors,
                "passive_model": getattr(pomdp_space, "passive_model", False),
                "simulation_params": parsed_sim_params,
                **raw_model_parameters,
                **({"num_timesteps": timesteps} if timesteps else {}),
            },
            "initialparameterization": initial_parameterization,
            "structured_pomdp": {
                "matrices": getattr(pomdp_space, "matrices", None) or {},
                "matrix_provenance": matrix_provenance,
                "state_factors": state_factors,
                "observation_modalities": observation_modalities,
                "control_factors": control_factors,
                "adapter_notes": getattr(pomdp_space, "adapter_notes", None) or [],
            },
            "matrix_provenance": matrix_provenance,
            "canonical_pomdp_schema": "canonical_pomdp_v1",
            "variables": [],
            "connections": [],
        }

        # Add variable definitions
        if pomdp_space.state_variables:
            gnn_spec["variables"].extend(pomdp_space.state_variables)
        if pomdp_space.observation_variables:
            gnn_spec["variables"].extend(pomdp_space.observation_variables)
        if pomdp_space.action_variables:
            gnn_spec["variables"].extend(pomdp_space.action_variables)

        # Add connections
        if pomdp_space.connections:
            gnn_spec["connections"] = [
                {"source": conn[0], "relation": conn[1], "target": conn[2]}
                for conn in pomdp_space.connections
            ]

        # Add ontology mapping if available
        if pomdp_space.ontology_mapping:
            gnn_spec["ontology_mapping"] = pomdp_space.ontology_mapping

        return gnn_spec

    def _call_framework_renderer(
        self, framework: str, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Call the appropriate framework renderer.

        Args:
            framework: Target framework name
            gnn_spec: GNN specification
            output_dir: Output directory for this framework
            **kwargs: Additional renderer options

        Returns:
            Renderer result dictionary
        """
        if framework == "pymdp":
            return self._call_pymdp_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "rxinfer":
            return self._call_rxinfer_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "activeinference_jl":
            return self._call_activeinference_jl_renderer(
                gnn_spec, output_dir, **kwargs
            )
        elif framework == "jax":
            return self._call_jax_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "discopy":
            return self._call_discopy_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "pytorch":
            return self._call_pytorch_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "numpyro":
            return self._call_numpyro_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "stan":
            return self._call_stan_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == "bnlearn":
            return self._call_bnlearn_renderer(gnn_spec, output_dir, **kwargs)
        else:
            return {
                "success": False,
                "message": f"No renderer implemented for {framework}",
                "artifacts": [],
            }

    def _call_pymdp_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call PyMDP renderer."""
        try:
            from .pymdp.pymdp_renderer import render_gnn_to_pymdp

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_pymdp.py"

            # Validate state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(gnn_spec, "pymdp")
            if not validation_result["valid"]:
                warnings = validation_result.get("warnings", [])
                if validation_result.get("critical", False):
                    return {
                        "success": False,
                        "message": f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        "artifacts": [],
                        "warnings": warnings,
                    }

            success, message, warnings = render_gnn_to_pymdp(
                gnn_spec, output_file, kwargs
            )

            # Post-render validation: verify state spaces are in generated script
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(
                    output_file, gnn_spec
                )
                if not post_validation["valid"]:
                    warnings.extend(post_validation.get("warnings", []))

            return {
                "success": success,
                "message": message,
                "artifacts": [str(output_file)] if success else [],
                "warnings": warnings,
            }

        except ImportError:
            return {
                "success": False,
                "message": "PyMDP renderer not available",
                "artifacts": [],
            }

    def _call_rxinfer_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call RxInfer renderer."""
        try:
            from .rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_rxinfer.jl"

            # Validate state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(gnn_spec, "rxinfer")
            if not validation_result["valid"]:
                warnings = validation_result.get("warnings", [])
                if validation_result.get("critical", False):
                    return {
                        "success": False,
                        "message": f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        "artifacts": [],
                        "warnings": warnings,
                    }

            success, message, warnings = render_gnn_to_rxinfer(
                gnn_spec, output_file, kwargs
            )

            # Post-render validation
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(
                    output_file, gnn_spec
                )
                if not post_validation["valid"]:
                    warnings.extend(post_validation.get("warnings", []))

            return {
                "success": success,
                "message": message,
                "artifacts": [str(output_file)] if success else [],
                "warnings": warnings,
            }

        except ImportError:
            return {
                "success": False,
                "message": "RxInfer renderer not available",
                "artifacts": [],
            }

    def _call_activeinference_jl_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call ActiveInference.jl renderer."""
        try:
            from .activeinference_jl.activeinference_renderer import (
                render_gnn_to_activeinference_jl,
            )

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_activeinference.jl"

            # Validate state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(
                gnn_spec, "activeinference_jl"
            )
            warnings: list[Any] = []
            if not validation_result["valid"]:
                warnings = validation_result.get("warnings", [])
                if validation_result.get("critical", False):
                    return {
                        "success": False,
                        "message": f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        "artifacts": [],
                        "warnings": warnings,
                    }

            success, message, artifacts = render_gnn_to_activeinference_jl(
                gnn_spec, output_file, kwargs
            )

            # Post-render validation
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(
                    output_file, gnn_spec
                )
                if not post_validation["valid"]:
                    warnings.extend(post_validation.get("warnings", []))

            return {
                "success": success,
                "message": message,
                "artifacts": artifacts,
                "warnings": warnings,
            }

        except ImportError:
            return {
                "success": False,
                "message": "ActiveInference.jl renderer not available",
                "artifacts": [],
            }

    def _call_jax_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call JAX renderer."""
        try:
            from .jax.jax_renderer import render_gnn_to_jax

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_jax.py"

            # Pre-render validation: verify state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(gnn_spec, "jax")
            warnings: list[Any] = []
            if not validation_result["valid"]:
                warnings = validation_result.get("warnings", [])
                if validation_result.get("critical", False):
                    return {
                        "success": False,
                        "message": f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        "artifacts": [],
                        "warnings": warnings,
                    }

            success, message, artifacts = render_gnn_to_jax(
                gnn_spec, output_file, kwargs
            )

            # Post-render validation: verify state spaces are in generated script
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(
                    output_file, gnn_spec
                )
                if not post_validation["valid"]:
                    warnings.extend(post_validation.get("warnings", []))

            return {
                "success": success,
                "message": message,
                "artifacts": artifacts,
                "warnings": warnings,
            }

        except ImportError:
            return {
                "success": False,
                "message": "JAX renderer not available",
                "artifacts": [],
            }

    def _call_discopy_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call DisCoPy renderer."""
        try:
            from .discopy.discopy_renderer import render_gnn_to_discopy

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_discopy.py"

            success, message, warnings = render_gnn_to_discopy(
                gnn_spec, output_file, kwargs
            )

            return {
                "success": success,
                "message": message,
                "artifacts": [str(output_file)] if success else [],
                "warnings": warnings,
            }

        except ImportError:
            return {
                "success": False,
                "message": "DisCoPy renderer not available",
                "artifacts": [],
            }

    def _call_bnlearn_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call bnlearn renderer."""
        try:
            from .generators import generate_bnlearn_code

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_bnlearn.py"

            code = generate_bnlearn_code(gnn_spec, output_file)
            success = bool(code)

            return {
                "success": success,
                "message": "bnlearn code generated"
                if success
                else "Failed to generate bnlearn code",
                "artifacts": [str(output_file)] if success else [],
                "warnings": [],
            }
        except ImportError as e:
            return {
                "success": False,
                "message": f"bnlearn generator not available: {e}",
                "artifacts": [],
            }

    def _call_pytorch_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call PyTorch renderer."""
        try:
            from .pytorch.pytorch_renderer import render_gnn_to_pytorch

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_pytorch.py"

            # Build options dict with timesteps if available
            options: dict[Any, Any] = {}
            model_params = gnn_spec.get("model_parameters", {})
            if "num_timesteps" in model_params:
                options["num_timesteps"] = model_params["num_timesteps"]

            success, message, artifacts = render_gnn_to_pytorch(
                gnn_spec, output_file, options or None
            )

            return {
                "success": success,
                "message": message,
                "artifacts": artifacts,
                "warnings": [],
            }

        except ImportError:
            return {
                "success": False,
                "message": "PyTorch renderer not available",
                "artifacts": [],
            }

    def _call_numpyro_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call NumPyro renderer."""
        try:
            from .numpyro.numpyro_renderer import render_gnn_to_numpyro

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_numpyro.py"

            # Build options dict with timesteps if available
            options: dict[Any, Any] = {}
            model_params = gnn_spec.get("model_parameters", {})
            if "num_timesteps" in model_params:
                options["num_timesteps"] = model_params["num_timesteps"]

            success, message, artifacts = render_gnn_to_numpyro(
                gnn_spec, output_file, options or None
            )

            return {
                "success": success,
                "message": message,
                "artifacts": artifacts,
                "warnings": [],
            }

        except ImportError:
            return {
                "success": False,
                "message": "NumPyro renderer not available",
                "artifacts": [],
            }

    def _call_stan_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call Stan renderer."""
        try:
            from .stan import render_stan

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{model_name}_stan.stan"
            code = render_stan(
                list(gnn_spec.get("variables", [])),
                list(gnn_spec.get("connections", [])),
                model_name=model_name,
            )
            output_file.write_text(code, encoding="utf-8")
            return {
                "success": True,
                "message": "Stan model generated",
                "artifacts": [str(output_file)],
                "warnings": [],
            }
        except ImportError:
            return {
                "success": False,
                "message": "Stan renderer not available",
                "artifacts": [],
            }

    def _validate_state_spaces_in_spec(
        self, gnn_spec: Dict[str, Any], framework: str
    ) -> Dict[str, Any]:
        """
        Validate that state spaces are present in GNN spec.

        Args:
            gnn_spec: GNN specification dictionary
            framework: Target framework name

        Returns:
            Validation result dictionary
        """
        warnings: list[Any] = []
        initial_params = gnn_spec.get("initialparameterization", {})
        config = self.framework_configs[framework]

        # Check required matrices
        missing_required: list[Any] = []
        required_matrices = cast(list[str], config["requires_matrices"])
        for required_matrix in required_matrices:
            if required_matrix not in initial_params:
                missing_required.append(required_matrix)

        if missing_required:
            return {
                "valid": False,
                "critical": True,
                "reason": f"Missing required matrices: {missing_required}",
                "warnings": warnings,
            }

        # Check optional matrices
        optional_matrices = cast(list[str], config.get("optional_matrices", []))
        for optional_matrix in optional_matrices:
            if optional_matrix not in initial_params:
                warnings.append(f"Optional matrix {optional_matrix} not found")

        return {"valid": True, "critical": False, "warnings": warnings}

    def _validate_state_spaces_in_script(
        self, script_path: Path, gnn_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that state spaces are present in generated script.

        Args:
            script_path: Path to generated script
            gnn_spec: Original GNN specification

        Returns:
            Validation result dictionary
        """
        warnings: list[Any] = []

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()

            initial_params = gnn_spec.get("initialparameterization", {})

            # Check if matrices are referenced in script
            for matrix_name in ["A", "B", "C", "D", "E"]:
                if matrix_name in initial_params:
                    # Check if matrix is present in script (as variable or in data structure)
                    if (
                        matrix_name not in script_content
                        and f'"{matrix_name}"' not in script_content
                    ):
                        warnings.append(
                            f"Matrix {matrix_name} may not be properly injected into script"
                        )

            return {"valid": len(warnings) == 0, "warnings": warnings}
        except Exception as e:
            return {"valid": False, "warnings": [f"Failed to validate script: {e}"]}

    def _create_framework_documentation(
        self,
        framework: str,
        pomdp_space: "POMDPStateSpace",
        output_dir: Path,
        render_result: Dict[str, Any],
    ) -> None:
        """
        Create framework-specific documentation.

        Args:
            framework: Framework name
            pomdp_space: POMDP state space
            output_dir: Output directory
            render_result: Rendering result
        """
        try:
            doc_file = output_dir / "README.md"

            # Get model annotation safely
            model_annotation = getattr(pomdp_space, "model_annotation", None) or "N/A"

            def _shape_text(value: Any) -> str | None:
                if value is None:
                    return None
                try:
                    array = np.asarray(value)
                    if array.size == 0:
                        return None
                    return "×".join(str(dim) for dim in array.shape)
                except (TypeError, ValueError):
                    if isinstance(value, (list, tuple)) and value:
                        nested_shapes: list[str] = []
                        for item in value:
                            nested_shape = _shape_text(item)
                            if nested_shape is not None:
                                nested_shapes.append(nested_shape)
                        if nested_shapes:
                            unique_shapes = sorted(set(nested_shapes))
                            return f"{len(value)} blocks ({', '.join(unique_shapes)})"
                    return None

            def _vector_length(value: Any) -> int | None:
                if value is None:
                    return None
                array = np.asarray(value)
                if array.size == 0:
                    return None
                return int(array.size)

            doc_content = f"""# {framework.upper()} Rendering Results

Generated from GNN POMDP Model: **{pomdp_space.model_name}**

## Model Information

- **Model Name**: {pomdp_space.model_name}
- **Model Description**: {model_annotation}
- **Generation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## POMDP Dimensions

- **Number of States**: {pomdp_space.num_states}
- **Number of Observations**: {pomdp_space.num_observations}
- **Number of Actions**: {pomdp_space.num_actions}

## Active Inference Matrices

### Available Matrices/Vectors:
"""

            # Safely check for matrices/vectors
            A_matrix = getattr(pomdp_space, "A_matrix", None)
            A_shape = _shape_text(A_matrix)
            if A_shape is not None:
                doc_content += f"- **A Matrix (Likelihood)**: {A_shape} - Maps hidden states to observations\n"

            B_matrix = getattr(pomdp_space, "B_matrix", None)
            B_shape = _shape_text(B_matrix)
            if B_shape is not None:
                doc_content += f"- **B Matrix (Transition)**: {B_shape} - State transitions given actions\n"

            C_vector = getattr(pomdp_space, "C_vector", None)
            C_length = _vector_length(C_vector)
            if C_length is not None:
                doc_content += f"- **C Vector (Preferences)**: Length {C_length} - Preferences over observations\n"

            D_vector = getattr(pomdp_space, "D_vector", None)
            D_length = _vector_length(D_vector)
            if D_length is not None:
                doc_content += f"- **D Vector (Prior)**: Length {D_length} - Prior beliefs over states\n"

            E_vector = getattr(pomdp_space, "E_vector", None)
            E_length = _vector_length(E_vector)
            if E_length is not None:
                doc_content += (
                    f"- **E Vector (Habits)**: Length {E_length} - Policy priors\n"
                )

            doc_content += """

## Generated Files

"""

            for artifact in render_result.get("artifacts", []):
                artifact_path = Path(artifact)
                doc_content += (
                    f"- `{artifact_path.name}` - {framework} simulation script\n"
                )

            if render_result.get("warnings"):
                doc_content += """

## Warnings

"""
                for warning in render_result["warnings"]:
                    doc_content += f"- ⚠️ {warning}\n"

            doc_content += f"""

## Usage

Refer to the main {framework} documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: {framework}
- **File Extension**: {self.framework_configs[framework]["file_extension"]}
- **Multi-Modality Support**: {"✅" if self.framework_configs[framework]["supports_multi_modality"] else "❌"}
- **Multi-Factor Support**: {"✅" if self.framework_configs[framework]["supports_multi_factor"] else "❌"}
"""

            with open(doc_file, "w") as f:
                f.write(doc_content)

            self.logger.info(f"Created documentation: {doc_file}")

        except Exception as e:
            self.logger.warning(f"Failed to create documentation for {framework}: {e}")


def process_pomdp_for_frameworks(
    pomdp_space: "POMDPStateSpace",
    output_dir: Union[str, Path],
    frameworks: Optional[List[str]] = None,
    gnn_file_path: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience function to process POMDP for multiple frameworks.

    Args:
        pomdp_space: POMDP state space data
        output_dir: Base output directory
        frameworks: List of frameworks to process (default: all)
        gnn_file_path: Original GNN file path
        **kwargs: Additional processing options

    Returns:
        Processing results dictionary
    """
    processor = POMDPRenderProcessor(Path(output_dir))
    return processor.process_pomdp_for_all_frameworks(
        pomdp_space, gnn_file_path, frameworks, **kwargs
    )
