#!/usr/bin/env python3
"""
POMDP State Space Extractor for GNN Active Inference Models

This module provides specialized parsing and extraction capabilities for POMDP
(Partially Observable Markov Decision Process) state spaces from GNN specifications,
with focus on Active Inference model structures.

Dependency contract
-------------------
This module is stdlib-only at import time. The only non-stdlib-adjacent import
(``utils.safe_eval`` for literal matrix evaluation) is performed lazily inside
_parse_parameter_value; the heavy pipeline (numpy, jax, pymdp, renderers, ...)
is NOT required. Headless consumers can use ``gnn.extract`` (CLI) or import
``gnn.pomdp_extractor`` directly under a blocked-import environment.

Stability promise: the mapping produced by :meth:`POMDPStateSpace.to_dict` is
versioned via its ``extraction_schema_version`` key (currently "1.0.0"). The 26
pre-existing keys keep identical semantics within schema version 1.x; new keys
may be appended in minor versions.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, overload

CANONICAL_B_ORDER = ["next_state", "previous_state", "action"]
"""Canonical B tensor semantic order: B[s', s, a] per pymdp 1.0.0
(control.py: B[f][s, v, u]; per-action slices column-stochastic over next_state)."""

ON_ERROR_MODES = ("lenient", "raise", "collect")
OnErrorMode = Literal["lenient", "raise", "collect"]


@dataclass
class GNNExtractionError(Exception):
    """Structured error emitted by the POMDP extractor.

    Local, hermetic type: structurally compatible with
    ``gnn.schema.GNNParseError`` (code/message/line + severity) so consumers can
    normalize across both surfaces, but deliberately not imported from there —
    probing ``from gnn.schema import GNNParseError`` shows schema.py itself is
    import-light (ast/logging/re/dataclasses/typing only), yet routing through
    the ``gnn`` package ``__init__`` drags pipeline weight, and this module must
    stay importable headless. Line numbers are best-effort (relative to the
    enclosing GNN section, offset toward file-absolute when the section header
    is locatable).

    Codes: GNN-E002 (shape/orientation contradiction), GNN-E006 (parameter
    parse failure), GNN-E999 (unexpected extraction failure). Warning codes use
    the GNN-W* namespace.
    """

    code: str
    message: str
    line: Optional[int] = None
    section: Optional[str] = None

    @property
    def severity(self) -> str:
        """'warning' for GNN-W* codes, 'error' otherwise."""
        return "warning" if self.code.startswith("GNN-W") else "error"

    def __str__(self) -> str:
        location = f" (line {self.line})" if self.line is not None else ""
        section = f" [{self.section}]" if self.section else ""
        return f"{self.code}{section}{location}: {self.message}"


def _gnn_distribution_version() -> str:
    """Best-effort installed version of the generalized-notation-notation package."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("generalized-notation-notation")
    except Exception:  # noqa: BLE001 - any metadata failure degrades to "unknown"
        return "unknown"


logger = logging.getLogger(__name__)


@dataclass
class POMDPStateSpace:
    """Represents extracted POMDP state space information."""

    # Core dimensions
    num_states: int
    num_observations: int
    num_actions: int

    # Active Inference matrices and vectors
    A_matrix: Optional[List[List[float]]] = None  # Likelihood: P(o|s)
    B_matrix: Optional[List[List[List[float]]]] = None  # Transition: P(s'|s,a)
    C_vector: Optional[List[float]] = None  # Preferences over observations
    D_vector: Optional[List[float]] = None  # Prior beliefs over states
    E_vector: Optional[List[float]] = None  # Policy priors

    # State space variables
    state_variables: Optional[List[Dict[str, Any]]] = None
    observation_variables: Optional[List[Dict[str, Any]]] = None
    action_variables: Optional[List[Dict[str, Any]]] = None
    state_factors: Optional[List[Dict[str, Any]]] = None
    observation_modalities: Optional[List[Dict[str, Any]]] = None
    control_factors: Optional[List[Dict[str, Any]]] = None

    # Connections/relationships
    connections: Optional[List[Tuple[str, str, str]]] = (
        None  # (source, relation, target)
    )

    # Metadata
    model_name: Optional[str] = None
    model_annotation: Optional[str] = None
    gnn_section: Optional[str] = None  # raw ## GNNSection value (e.g. ActInfPOMDP)
    ontology_mapping: Optional[Dict[str, str]] = None
    num_timesteps: Optional[int] = None  # Simulation timesteps (from ModelParameters)
    model_parameters: Optional[Dict[str, Any]] = None
    matrices: Optional[Dict[str, Any]] = None
    matrix_provenance: Optional[Dict[str, Dict[str, Any]]] = None
    passive_model: bool = False
    adapter_notes: Optional[List[str]] = None
    initial_parameterization: Optional[Dict[str, Any]] = None
    # "discrete" (categorical POMDP/HMM) or "continuous" (linear-Gaussian
    # state-space model declared via F/H/Q/R + prior_mean/prior_cov).
    model_kind: str = "discrete"

    # Computed factor counts (bookkeeping excluded). A descriptor counts as a
    # factor unless its name matches *_prime (next-state/next-observation
    # aliases like s_prime/o_prime) or it is the policy symbol pi/π; the action
    # variable u DOES count as the control factor. The state_factors /
    # observation_modalities / control_factors lists keep ALL entries
    # (including bookkeeping, each tagged role='factor'|'bookkeeping'); only
    # these num_* counts exclude bookkeeping.
    num_state_factors: Optional[int] = None
    num_observation_modalities: Optional[int] = None
    num_control_factors: Optional[int] = None

    # Which _extract_dimensions priority level produced each core dimension:
    # {name: {"value": ..., "source": "ModelParameters|inferred_from_B_shape|
    # variable_dimensions|default"}}.
    dimension_provenance: Optional[Dict[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_states": self.num_states,
            "num_observations": self.num_observations,
            "num_actions": self.num_actions,
            "A_matrix": self.A_matrix,
            "B_matrix": self.B_matrix,
            "C_vector": self.C_vector,
            "D_vector": self.D_vector,
            "E_vector": self.E_vector,
            "state_variables": self.state_variables,
            "observation_variables": self.observation_variables,
            "action_variables": self.action_variables,
            "state_factors": self.state_factors,
            "observation_modalities": self.observation_modalities,
            "control_factors": self.control_factors,
            "connections": self.connections,
            "model_name": self.model_name,
            "model_annotation": self.model_annotation,
            "ontology_mapping": self.ontology_mapping,
            "num_timesteps": self.num_timesteps,
            "model_parameters": self.model_parameters,
            "matrices": self.matrices,
            "matrix_provenance": self.matrix_provenance,
            "passive_model": self.passive_model,
            "adapter_notes": self.adapter_notes,
            "initial_parameterization": self.initial_parameterization,
            "model_kind": self.model_kind,
            "gnn_version": _gnn_distribution_version(),
            "extraction_schema_version": "1.0.0",
            "num_state_factors": self.num_state_factors,
            "num_observation_modalities": self.num_observation_modalities,
            "num_control_factors": self.num_control_factors,
            "dimension_provenance": self.dimension_provenance,
        }


class POMDPExtractor:
    """
    Specialized extractor for POMDP state spaces from GNN specifications.

    Features:
    - Parses Active Inference matrix structures (A, B, C, D, E)
    - Extracts state space dimensions and variable definitions
    - Handles initial parameterization with matrix values
    - Maps ontology annotations to Active Inference concepts
    - Validates POMDP structural consistency
    """

    def __init__(self, strict_validation: bool = True) -> None:
        """
        Initialize POMDP extractor.

        Args:
            strict_validation: Enable strict validation of POMDP structure
        """
        self.strict_validation = strict_validation
        self.logger = logging.getLogger(__name__)

        # Error-collection state (reset at the start of each extraction call,
        # but initialized here so private helpers are safe in isolation).
        self._on_error: str = "lenient"
        self._errors: List[GNNExtractionError] = []
        self._parse_failures: Dict[str, Dict[str, Any]] = {}
        self._section_line_offset: int = 0
        self._dimension_sources: Dict[str, str] = {}

        # Patterns for parsing GNN content
        self.SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        self.VARIABLE_PATTERN = re.compile(
            r"^([A-Za-z_π][A-Za-z0-9_π]*)\[([^\]]+)\](?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$"
        )
        self.CONNECTION_PATTERN = re.compile(
            r"^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$"
        )
        self.PARAMETER_PATTERN = re.compile(
            r"^([A-Za-z_π][A-Za-z0-9_π]*)\s*=\s*\{(.+)\}", re.MULTILINE | re.DOTALL
        )

    @overload
    def extract_from_gnn_content(
        self,
        content: str,
        *,
        on_error: Literal["lenient", "raise"] = ...,
        insert_default_c: bool = ...,
    ) -> Optional[POMDPStateSpace]: ...

    @overload
    def extract_from_gnn_content(
        self,
        content: str,
        *,
        on_error: Literal["collect"],
        insert_default_c: bool = ...,
    ) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...

    def extract_from_gnn_content(
        self,
        content: str,
        *,
        on_error: str = "lenient",
        insert_default_c: bool = True,
    ) -> Union[
        Optional[POMDPStateSpace],
        Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
    ]:
        """
        Extract POMDP state space from GNN content.

        Args:
            content: Raw GNN file content
            on_error: Failure policy.
                - 'lenient' (default): log faults and keep going; parse-error
                  provenance/adapter_notes records are still written.
                - 'raise': raise GNNExtractionError at the first fault.
                - 'collect': return (spec_or_None, list[GNNExtractionError]).
                Invalid values raise ValueError.
            insert_default_c: When True (default) a passive model without a C
                vector keeps the zero-preference adapter behavior. When False
                the faithful read is preserved: C_vector stays None and no
                passive_model_adapter provenance/adapter_notes entry is made.

        Returns:
            POMDPStateSpace (lenient/raise modes) or a
            (POMDPStateSpace | None, list[GNNExtractionError]) tuple
            (collect mode).
        """
        if on_error not in ON_ERROR_MODES:
            raise ValueError(
                f"on_error must be one of {ON_ERROR_MODES}, got {on_error!r}"
            )
        self._on_error = on_error
        self._errors = []
        self._parse_failures = {}
        self._section_line_offset = self._section_line_offset_for(
            content, "InitialParameterization"
        )
        try:
            sections = self._parse_sections(content)

            # Extract basic information
            model_name = self._extract_model_name(sections)
            model_annotation = self._extract_model_annotation(sections)
            gnn_section = sections.get("GNNSection", "").strip() or None

            # Parse state space block
            state_space_info = self._parse_state_space_block(
                sections.get("StateSpaceBlock", "")
            )

            # Parse initial parameterization FIRST (needed for dimension inference)
            initial_params = self._parse_initial_parameterization(
                sections.get("InitialParameterization", "")
            )
            model_parameters = self._parse_model_parameters(
                sections.get("ModelParameters", "")
            )

            # Continuous-state (linear-Gaussian) models declare F/H/Q/R and a
            # Gaussian prior instead of categorical A/B/C/D. Their dimensions
            # come from the system matrices, never from a discrete stand-in.
            is_continuous = self._is_continuous_model(gnn_section, initial_params)

            if is_continuous:
                num_states, num_observations, num_actions, num_timesteps = (
                    self._extract_continuous_dimensions(
                        state_space_info, initial_params, model_parameters
                    )
                )
            else:
                # Extract dimensions (now with access to sections and initial_params for better inference)
                num_states, num_observations, num_actions, num_timesteps = (
                    self._extract_dimensions(
                        state_space_info,
                        sections=sections,
                        initial_params=initial_params,
                    )
                )

            # Parse connections
            connections = self._parse_connections(sections.get("Connections", ""))

            # Parse ontology mapping
            ontology_mapping = self._parse_ontology_annotations(
                sections.get("ActInfOntologyAnnotation", "")
            )

            if is_continuous:
                matrices = self._collect_continuous_parameters(initial_params)
            else:
                matrices = self._collect_matrix_parameters(initial_params)
            matrix_provenance = self._build_matrix_provenance(matrices)
            state_factors = self._describe_variables(
                state_space_info.get("state_variables"), "state_factor"
            )
            observation_modalities = self._describe_variables(
                state_space_info.get("observation_variables"), "observation_modality"
            )
            control_factors = self._describe_variables(
                state_space_info.get("action_variables"), "control_factor"
            )
            if is_continuous:
                passive_model = not (
                    "goal_mean" in initial_params and "control_gain" in initial_params
                )
            else:
                passive_model = self._is_passive_model(
                    model_parameters=model_parameters,
                    initial_params=initial_params,
                    num_actions=num_actions,
                    connections=connections,
                )

            A_matrix = None if is_continuous else initial_params.get("A")
            B_matrix = None if is_continuous else initial_params.get("B")
            C_vector = None if is_continuous else initial_params.get("C")
            D_vector = None if is_continuous else initial_params.get("D")
            E_vector = None if is_continuous else initial_params.get("E")
            adapter_notes: list[Any] = []

            # Failed parameter blocks are never silently dropped: record
            # parse_error provenance + adapter_notes entries in every mode.
            for failed_name, failure in self._parse_failures.items():
                if failed_name in ("A", "B", "C", "D", "E") or failed_name.startswith(
                    ("A_", "B_", "C_", "D_", "E_")
                ):
                    matrix_provenance[failed_name] = {
                        "source": "parse_error",
                        "code": failure["code"],
                        "message": failure["message"],
                    }
                adapter_notes.append(
                    f"parse_error:{failed_name} [{failure['code']}]: {failure['message']}"
                )

            if (
                insert_default_c
                and not is_continuous
                and C_vector is None
                and passive_model
                and num_observations > 0
            ):
                C_vector = [0.0] * num_observations
                matrices["C"] = C_vector
                matrix_provenance["C"] = {
                    "source": "passive_model_adapter",
                    "shape": [num_observations],
                    "derived": True,
                    "reason": "zero preferences for passive HMM/Markov model",
                }
                adapter_notes.append("passive_model_zero_preferences")

            # B-orientation metadata (detection only — the stored B_matrix is
            # never transposed here; render/execute depend on as-written nesting).
            if not is_continuous and isinstance(B_matrix, (list, tuple)):
                b_provenance = matrix_provenance.get("B")
                if (
                    b_provenance is not None
                    and b_provenance.get("source") == "InitialParameterization"
                ):
                    orientation = self._analyze_b_orientation(
                        sections.get("StateSpaceBlock", ""),
                        sections.get("InitialParameterization", ""),
                        B_matrix,
                    )
                    b_provenance.update(orientation)
                    if orientation.get("contradiction"):
                        message = (
                            f"B orientation contradiction: {orientation.get('reason')}"
                        )
                        if self.strict_validation and on_error in ("raise", "collect"):
                            self._record_error(
                                "GNN-E002", message, section="StateSpaceBlock"
                            )
                        else:
                            self.logger.warning("B orientation: %s", message)

            # Computed factor counts (bookkeeping excluded; see dataclass).
            num_state_factors = sum(
                1 for descriptor in state_factors if descriptor.get("role") == "factor"
            )
            num_observation_modalities = sum(
                1
                for descriptor in observation_modalities
                if descriptor.get("role") == "factor"
            )
            num_control_factors = sum(
                1
                for descriptor in control_factors
                if descriptor.get("role") == "factor"
            )

            dimension_provenance = self._build_dimension_provenance(
                num_states, num_observations, num_actions, num_timesteps
            )

            # Create POMDP state space
            pomdp_space = POMDPStateSpace(
                num_states=num_states,
                num_observations=num_observations,
                num_actions=num_actions,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                C_vector=C_vector,
                D_vector=D_vector,
                E_vector=E_vector,
                state_variables=state_space_info.get("state_variables"),
                observation_variables=state_space_info.get("observation_variables"),
                action_variables=state_space_info.get("action_variables"),
                state_factors=state_factors,
                observation_modalities=observation_modalities,
                control_factors=control_factors,
                connections=connections,
                model_name=model_name,
                model_annotation=model_annotation,
                gnn_section=gnn_section,
                ontology_mapping=ontology_mapping,
                num_timesteps=num_timesteps,
                model_parameters=model_parameters,
                matrices=matrices,
                matrix_provenance=matrix_provenance,
                passive_model=passive_model,
                adapter_notes=adapter_notes,
                initial_parameterization=initial_params,
                model_kind="continuous" if is_continuous else "discrete",
                num_state_factors=num_state_factors,
                num_observation_modalities=num_observation_modalities,
                num_control_factors=num_control_factors,
                dimension_provenance=dimension_provenance,
            )

            # Validate if strict validation enabled (discrete contract only —
            # continuous models are shape-checked in _extract_continuous_dimensions)
            if self.strict_validation and not is_continuous:
                validation_result = self._validate_pomdp_structure(pomdp_space)
                if not validation_result["valid"]:
                    self.logger.warning(
                        f"POMDP validation warnings: {validation_result['warnings']}"
                    )

            if on_error == "collect":
                return (pomdp_space, self._errors)
            return pomdp_space

        except GNNExtractionError:
            raise
        except Exception as e:
            error = GNNExtractionError(
                code="GNN-E999",
                message=f"unexpected extraction failure: {e}",
            )
            if on_error == "raise":
                raise error from e
            self.logger.error(f"Failed to extract POMDP state space: {e}")
            self._errors.append(error)
            if on_error == "collect":
                return (None, self._errors)
            return None

    @overload
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        *,
        on_error: Literal["lenient", "raise"] = ...,
        insert_default_c: bool = ...,
    ) -> Optional[POMDPStateSpace]: ...

    @overload
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        *,
        on_error: Literal["collect"],
        insert_default_c: bool = ...,
    ) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...

    def extract_from_file(
        self,
        file_path: Union[str, Path],
        *,
        on_error: str = "lenient",
        insert_default_c: bool = True,
    ) -> Union[
        Optional[POMDPStateSpace],
        Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
    ]:
        """
        Extract POMDP state space from GNN file.

        Args:
            file_path: Path to GNN file
            on_error: 'lenient' (default) | 'raise' | 'collect' — see
                extract_from_gnn_content. Invalid values raise ValueError.
            insert_default_c: Preserve (True, default) or suppress (False) the
                passive-model zero-C adapter.

        Returns:
            POMDPStateSpace (lenient/raise modes) or a
            (POMDPStateSpace | None, list[GNNExtractionError]) tuple
            (collect mode).
        """
        if on_error not in ON_ERROR_MODES:
            raise ValueError(
                f"on_error must be one of {ON_ERROR_MODES}, got {on_error!r}"
            )
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                error = GNNExtractionError(
                    code="GNN-E999",
                    message=f"file not found: {file_path}",
                )
                if on_error == "raise":
                    raise error
                self.logger.error("File not found: %s", file_path)
                self._errors.append(error)
                if on_error == "collect":
                    return (None, self._errors)
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return self.extract_from_gnn_content(
                content,
                on_error=cast(OnErrorMode, on_error),
                insert_default_c=insert_default_c,
            )

        except GNNExtractionError:
            raise
        except Exception as e:
            error = GNNExtractionError(
                code="GNN-E999",
                message=f"failed to read file {file_path}: {e}",
            )
            if on_error == "raise":
                raise error from e
            self.logger.error("Failed to read file %s: %s", file_path, e)
            self._errors.append(error)
            if on_error == "collect":
                return (None, self._errors)
            return None

    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse GNN content into sections."""
        sections: dict[Any, Any] = {}
        current_section = None
        current_content: list[Any] = []

        for line in content.split("\n"):
            line = line.strip()

            # Check for section header
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = section_match.group(1).strip()
                current_content = []
            else:
                # Add line to current section
                if current_section and line:
                    current_content.append(line)

        # Save final section
        if current_section:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _extract_model_name(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract model name from sections."""
        return sections.get("ModelName", "").strip() or None

    def _extract_model_annotation(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract model annotation from sections."""
        return sections.get("ModelAnnotation", "").strip() or None

    def _parse_state_space_block(self, content: str) -> Dict[str, Any]:
        """Parse StateSpaceBlock section."""
        variables: dict[str, Any] = {
            "state_variables": [],
            "observation_variables": [],
            "action_variables": [],
        }

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = self.VARIABLE_PATTERN.match(line)
            if match:
                var_name = match.group(1)
                dimensions_str = match.group(2)
                var_type = match.group(3) or "float"
                comment = match.group(4)

                # Parse dimensions
                dimensions: list[Any] = []
                for dim in dimensions_str.split(","):
                    dim = dim.strip()
                    if "=" not in dim:  # Skip type specifications
                        try:
                            if dim == "π":  # Special handling for π
                                dimensions.append("π")
                            else:
                                dimensions.append(int(dim))
                        except ValueError:
                            dimensions.append(dim)  # Keep as string if not integer

                var_info: dict[str, Any] = {
                    "name": var_name,
                    "dimensions": dimensions,
                    "type": var_type,
                    "comment": comment,
                }

                # Categorize variables. Name prefixes are authoritative first
                # (s* = state, o* = observation, u/pi* = action); matrix/vector
                # parameters (A/B/C/D/E/F/G or A_*/B_*...) are never state-space
                # variables regardless of comment wording; comment keywords are
                # the fallback heuristic.
                name_lower = var_name.lower()
                if name_lower.startswith("s"):
                    variables["state_variables"].append(var_info)
                elif name_lower.startswith("o"):
                    variables["observation_variables"].append(var_info)
                elif name_lower in ["u", "π"] or name_lower.startswith(("u", "pi")):
                    variables["action_variables"].append(var_info)
                elif var_name in ["A", "B", "C", "D", "E", "F", "G"] or any(
                    var_name.startswith(f"{prefix}_")
                    for prefix in ("A", "B", "C", "D", "E", "F", "G")
                ):
                    # These are matrix/vector parameters, not state space variables
                    continue
                elif "state" in (comment or "").lower():
                    variables["state_variables"].append(var_info)
                elif "observation" in (comment or "").lower():
                    variables["observation_variables"].append(var_info)
                elif (
                    "action" in (comment or "").lower()
                    or "policy" in (comment or "").lower()
                ):
                    variables["action_variables"].append(var_info)
                else:
                    # Default categorization based on typical Active Inference naming
                    variables["state_variables"].append(var_info)

        return variables

    def _extract_dimensions(
        self,
        state_space_info: Dict[str, Any],
        sections: Optional[Dict[str, str]] = None,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int, Optional[int]]:
        """
        Extract core dimensions from state space information.

        Priority for num_actions:
        1. ModelParameters section (num_actions, num_controls)
        2. B matrix dimensions (inferred from shape)
        3. Action variables (u, π)
        4. Default (3)

        Which level fired for each of num_states / num_observations /
        num_actions / num_timesteps is recorded in
        ``self._dimension_sources`` (values: "ModelParameters" |
        "inferred_from_B_shape" | "variable_dimensions" | "default") for the
        dimension_provenance field. The return signature is unchanged.
        """
        num_states = 3  # Default
        num_observations = 3  # Default
        num_actions = None  # Will be determined by priority
        num_timesteps = None  # Simulation timesteps (optional)
        sources: Dict[str, str] = {
            "num_states": "default",
            "num_observations": "default",
            "num_actions": "default",
            "num_timesteps": "default",
        }
        self._dimension_sources = sources

        # Priority 1: Check ModelParameters section
        if sections:
            for key, value in self._parse_model_parameters(
                sections.get("ModelParameters", "")
            ).items():
                try:
                    int_value = int(value)
                except (ValueError, TypeError):
                    continue
                key_lower = key.lower()
                if key_lower in ["num_actions", "num_controls", "n_actions"]:
                    num_actions = int_value
                    sources["num_actions"] = "ModelParameters"
                elif key_lower in [
                    "num_hidden_states",
                    "num_states",
                    "n_states",
                    "num_locations",
                ]:
                    num_states = int_value
                    sources["num_states"] = "ModelParameters"
                elif key_lower in [
                    "num_obs",
                    "num_observations",
                    "n_obs",
                    "num_location_obs",
                ]:
                    num_observations = int_value
                    sources["num_observations"] = "ModelParameters"
                elif key_lower in ["num_timesteps", "n_timesteps", "timesteps"]:
                    num_timesteps = int_value
                    sources["num_timesteps"] = "ModelParameters"

        # Priority 2: Infer from B matrix dimensions if still None
        if num_actions is None and initial_params:
            action_candidates: list[Any] = []
            for key, matrix in initial_params.items():
                if key == "B" or key.startswith("B_"):
                    shape = self._nested_shape(matrix)
                    if len(shape) == 2:
                        action_candidates.append(1)
                    elif len(shape) == 3:
                        if shape[0] == shape[1]:
                            action_candidates.append(shape[2])
                        elif shape[1] == shape[2]:
                            action_candidates.append(shape[0])
                        else:
                            action_candidates.append(shape[-1])
            if action_candidates:
                num_actions = max(action_candidates)
                sources["num_actions"] = "inferred_from_B_shape"
                self.logger.info(
                    "Inferred num_actions=%d from B matrix dimensions", num_actions
                )

        # Priority 3: Try to extract from state variables
        for var in state_space_info.get("state_variables", []):
            if var["name"].lower() == "s":
                if len(var["dimensions"]) > 0 and isinstance(var["dimensions"][0], int):
                    if num_states == 3:  # Only override default
                        num_states = var["dimensions"][0]
                        sources["num_states"] = "variable_dimensions"

        # Try to extract from observation variables
        for var in state_space_info.get("observation_variables", []):
            if var["name"].lower() == "o":
                if len(var["dimensions"]) > 0 and isinstance(var["dimensions"][0], int):
                    if num_observations == 3:  # Only override default
                        num_observations = var["dimensions"][0]
                        sources["num_observations"] = "variable_dimensions"

        # Priority 4: Try to extract from action variables (if still None)
        if num_actions is None:
            for var in state_space_info.get("action_variables", []):
                if var["name"].lower() in ["u", "π"]:
                    if len(var["dimensions"]) > 0 and isinstance(
                        var["dimensions"][0], int
                    ):
                        # Only use if > 1 (u[1] means single action, not 1 possible action)
                        dim = var["dimensions"][0]
                        if dim > 1:
                            num_actions = dim
                            sources["num_actions"] = "variable_dimensions"

        # Final default
        if num_actions is None:
            num_actions = 3

        return num_states, num_observations, num_actions, num_timesteps

    def _parse_model_parameters(self, content: str) -> Dict[str, Any]:
        """Parse the ModelParameters section into typed scalar values."""
        params: Dict[str, Any] = {}
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            clean_value = value.split("#", 1)[0].strip()
            key = key.strip()
            if not clean_value:
                continue
            try:
                params[key] = int(clean_value)
                continue
            except ValueError as e:
                self.logger.debug("Model parameter %s is not an int: %s", key, e)
            try:
                params[key] = float(clean_value)
                continue
            except ValueError as e:
                self.logger.debug("Model parameter %s is not a float: %s", key, e)
            if clean_value.lower() in {"true", "false"}:
                params[key] = clean_value.lower() == "true"
            else:
                params[key] = clean_value
        return params

    CONTINUOUS_REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")
    CONTINUOUS_OPTIONAL_KEYS = ("goal_mean", "control_gain")

    def _is_continuous_model(
        self, gnn_section: Optional[str], initial_params: Dict[str, Any]
    ) -> bool:
        """Continuous when the section says so or the full LGSSM block is declared."""
        if gnn_section and "continuous" in gnn_section.lower():
            return True
        return all(key in initial_params for key in ("F", "H", "Q", "R"))

    def _extract_continuous_dimensions(
        self,
        state_space_info: Dict[str, Any],
        initial_params: Dict[str, Any],
        model_parameters: Dict[str, Any],
    ) -> Tuple[int, int, int, Optional[int]]:
        """Dimensions of a linear-Gaussian model: n from F, m from H."""
        missing = [k for k in self.CONTINUOUS_REQUIRED_KEYS if k not in initial_params]
        if missing:
            raise ValueError(
                f"continuous model is missing linear-Gaussian parameters {missing}"
            )
        f_shape = self._nested_shape(initial_params["F"])
        h_shape = self._nested_shape(initial_params["H"])
        if len(f_shape) != 2 or f_shape[0] != f_shape[1]:
            raise ValueError(f"F must be a square matrix, got shape {f_shape}")
        if len(h_shape) != 2 or h_shape[1] != f_shape[0]:
            raise ValueError(
                f"H must have shape [m, n] with n={f_shape[0]}, got {h_shape}"
            )
        num_states, num_observations = f_shape[0], h_shape[0]
        self._dimension_sources = {
            "num_states": "variable_dimensions",
            "num_observations": "variable_dimensions",
            "num_actions": "default",
            "num_timesteps": "default",
        }
        for key, expected in (
            ("Q", [num_states, num_states]),
            ("R", [num_observations, num_observations]),
            ("prior_cov", [num_states, num_states]),
            ("prior_mean", [num_states]),
        ):
            shape = self._nested_shape(initial_params[key])
            if shape != expected:
                raise ValueError(f"{key} has shape {shape}, expected {expected}")
        # One continuous control channel when a control variable is declared.
        num_actions = 1 if state_space_info.get("action_variables") else 0
        num_timesteps: Optional[int] = None
        raw_t = model_parameters.get("num_timesteps")
        if raw_t is not None:
            try:
                num_timesteps = int(raw_t)
                self._dimension_sources["num_timesteps"] = "ModelParameters"
            except (TypeError, ValueError):
                num_timesteps = None
        return num_states, num_observations, num_actions, num_timesteps

    def _collect_continuous_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return the linear-Gaussian parameter block (no discrete stand-in)."""
        keys = self.CONTINUOUS_REQUIRED_KEYS + self.CONTINUOUS_OPTIONAL_KEYS
        out: Dict[str, Any] = {}
        for key in keys:
            if key in params:
                value = params[key]
                if key == "control_gain":
                    # scalar declared as {(v)} parses to [v]
                    if isinstance(value, (list, tuple)):
                        while isinstance(value, (list, tuple)) and len(value) == 1:
                            value = value[0]
                    value = float(value)
                out[key] = value
        return out

    def _collect_matrix_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return all Active Inference matrices/vectors without collapsing factors."""
        matrices: Dict[str, Any] = {}
        for key, value in params.items():
            if key in {"A", "B", "C", "D", "E"} or any(
                key.startswith(f"{prefix}_") for prefix in ("A", "B", "C", "D", "E")
            ):
                matrices[key] = value
        return matrices

    def _build_matrix_provenance(
        self, matrices: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Describe where each extracted matrix came from and what shape it has."""
        return {
            key: {
                "source": "InitialParameterization",
                "shape": self._nested_shape(value),
                "derived": False,
            }
            for key, value in matrices.items()
        }

    def _nested_shape(self, value: Any) -> List[int]:
        """Return a best-effort shape for nested Python matrix data."""
        return _shape_of(value)

    def _describe_variables(
        self,
        variables: Optional[List[Dict[str, Any]]],
        fallback_prefix: str,
    ) -> List[Dict[str, Any]]:
        """Create factor/modality/control descriptors from parsed variables.

        Every descriptor carries 'role': 'factor' or 'bookkeeping'. Bookkeeping
        entries are next-state/next-observation aliases matching *_prime and
        the policy symbol π/pi; the action variable u is a real factor. Lists
        keep ALL entries; the num_* counts on POMDPStateSpace exclude
        bookkeeping.
        """
        descriptors: list[Any] = []
        for index, variable in enumerate(variables or []):
            name = variable.get("name") or f"{fallback_prefix}_{index}"
            name_lower = str(name).lower()
            if fallback_prefix == "state_factor" and not name_lower.startswith("s"):
                continue
            if fallback_prefix == "observation_modality" and not name_lower.startswith(
                "o"
            ):
                continue
            if (
                fallback_prefix == "control_factor"
                and name_lower not in {"u", "π", "pi"}
                and not name_lower.startswith("pi")
            ):
                continue
            dimensions = variable.get("dimensions") or []
            size = next((dim for dim in dimensions if isinstance(dim, int)), None)
            is_bookkeeping = name_lower.endswith("_prime") or (
                fallback_prefix == "control_factor" and name_lower in {"π", "pi"}
            )
            descriptors.append(
                {
                    "name": name,
                    "size": size,
                    "dimensions": dimensions,
                    "type": variable.get("type"),
                    "comment": variable.get("comment"),
                    "index": index,
                    "role": "bookkeeping" if is_bookkeeping else "factor",
                }
            )
        return descriptors

    def _is_passive_model(
        self,
        *,
        model_parameters: Dict[str, Any],
        initial_params: Dict[str, Any],
        num_actions: int,
        connections: List[Tuple[str, str, str]],
    ) -> bool:
        """Detect passive HMM/Markov models that have no control-dependent choices."""
        model_type = str(model_parameters.get("model_type", "")).lower()
        if any(term in model_type for term in ("hmm", "markov", "passive")):
            return True
        if int(num_actions) == 1:
            return True
        b_matrix = initial_params.get("B")
        if b_matrix is not None and len(self._nested_shape(b_matrix)) == 2:
            return True
        return False

    def _record_error(
        self,
        code: str,
        message: str,
        line: Optional[int] = None,
        section: Optional[str] = None,
    ) -> GNNExtractionError:
        """Record a structured fault; raise immediately in 'raise' mode."""
        error = GNNExtractionError(
            code=code, message=message, line=line, section=section
        )
        self._errors.append(error)
        if self._on_error == "raise":
            raise error
        if error.severity == "error":
            self.logger.error("structured error: %s", error)
        else:
            self.logger.warning("structured warning: %s", error)
        return error

    def _record_parameter_failure(
        self,
        param_name: str,
        exc: BaseException,
        line_no: Optional[int] = None,
    ) -> None:
        """Record a failed parameter parse (GNN-E006); never silently dropped."""
        if isinstance(exc, (ValueError, SyntaxError)):
            code = "GNN-E006"
        else:
            code = "GNN-E006"
        line = (
            self._section_line_offset + line_no
            if self._section_line_offset and line_no
            else line_no
        )
        self._parse_failures[param_name] = {
            "code": code,
            "message": f"{param_name}: {exc}",
        }
        self.logger.warning("Failed to parse parameter %s: %s", param_name, exc)
        self._record_error(
            code,
            f"failed to parse parameter '{param_name}': {exc}",
            line=line,
            section="InitialParameterization",
        )

    def _section_line_offset_for(self, content: str, section: str) -> int:
        """File-absolute 1-based line number of a section header (0 if absent)."""
        for index, raw_line in enumerate(content.split("\n"), start=1):
            if raw_line.strip().lower() == f"## {section.lower()}":
                return index
        return 0

    def _build_dimension_provenance(
        self,
        num_states: int,
        num_observations: int,
        num_actions: int,
        num_timesteps: Optional[int],
    ) -> Dict[str, Dict[str, Any]]:
        """Expose which _extract_dimensions priority level fired per dimension."""
        sources = self._dimension_sources or {}
        provenance: Dict[str, Dict[str, Any]] = {
            "num_states": {
                "value": num_states,
                "source": sources.get("num_states", "default"),
            },
            "num_observations": {
                "value": num_observations,
                "source": sources.get("num_observations", "default"),
            },
            "num_actions": {
                "value": num_actions,
                "source": sources.get("num_actions", "default"),
            },
        }
        provenance["num_timesteps"] = {
            "value": num_timesteps,
            "source": sources.get("num_timesteps", "default"),
        }
        return provenance

    # --- B-orientation metadata (detection only; never re-orients data) ---

    _AXIS_ALIASES = {
        "next_state": "next_state",
        "next": "next_state",
        "s_next": "next_state",
        "states_next": "next_state",
        "s'": "next_state",
        "previous_state": "previous_state",
        "prev_state": "previous_state",
        "previous": "previous_state",
        "prev": "previous_state",
        "s_prev": "previous_state",
        "states_previous": "previous_state",
        "action": "action",
        "actions": "action",
        "u": "action",
    }

    def _parse_declared_b_order(self, state_space_block: str) -> Optional[List[str]]:
        """Parse the declared B axis order from the StateSpaceBlock comment."""
        for line in state_space_block.split("\n"):
            if "B[" not in line:
                continue
            for match in re.finditer(r"B\[([^\]]+)\]", line):
                axes = [part.strip().lower() for part in match.group(1).split(",")]
                order: List[str] = []
                for axis in axes:
                    for alias, canonical in self._AXIS_ALIASES.items():
                        if axis == alias or axis.startswith(alias):
                            if canonical not in order:
                                order.append(canonical)
                            break
                if len(order) == 3:
                    return order
        return None

    def _parse_claimed_slice_convention(self, parameterization: str) -> Optional[str]:
        """Parse the claimed per-slice convention from the InitialParameterization B comment."""
        near_b = re.search(
            r"#\s*B:.*?(?=\n(?:[A-Za-zπ_]\w*\s*=)|$)",
            parameterization,
            re.DOTALL,
        )
        text = near_b.group(0).lower() if near_b else parameterization.lower()
        if ("rows are previous" in text or "rows as previous" in text) and (
            "columns are next" in text or "columns as next" in text
        ):
            return "rows_previous_columns_next"
        if "rows are next" in text or "rows as next" in text:
            return "rows_next_columns_previous"
        return None

    def _detect_b_order(self, b_matrix: Any) -> Optional[List[str]]:
        """Detect the stored tensor's axis order from stochasticity sums.

        Evidence tests over the stored (as-written) tensor T[d0][d1][d2]:
        - Doubly stochastic (dominant): every slice has row sums AND column
          sums = 1 (permutation-style data). Ambiguous — never decisive, and
          never a contradiction by itself.
        - H2 (canonical): sum over the OUTER axis at each (i, j) position = 1
          (law of total probability over next_state) -> stored is
          (next_state, previous_state, action).
        - H1 (action-outer): every slice is row-stochastic (row sums = 1) ->
          stored is (action, previous_state, next_state).
        Neither test decisive -> None.
        """
        shape = self._nested_shape(b_matrix)
        if len(shape) != 3 or 0 in shape:
            return None
        try:
            rows_stochastic = all(
                abs(sum(float(v) for v in row) - 1.0) <= 1e-6
                for slice_ in b_matrix
                for row in slice_
            )
            cols_stochastic = all(
                abs(sum(float(row[j]) for row in slice_) - 1.0) <= 1e-6
                for slice_ in b_matrix
                for j in range(len(slice_[0]))
            )
            doubly_stochastic = rows_stochastic and cols_stochastic
            h2 = all(
                abs(sum(float(s[i][j]) for s in b_matrix) - 1.0) <= 1e-6
                for i in range(shape[1])
                for j in range(shape[2])
            )
        except (TypeError, ValueError, IndexError):
            return None

        if doubly_stochastic:
            return None  # ambiguous — never decisive, never a contradiction
        if h2:
            return list(CANONICAL_B_ORDER)  # (next, prev, action)
        if rows_stochastic:
            return ["action", "previous_state", "next_state"]
        return None

    def _analyze_b_orientation(
        self,
        state_space_block: str,
        parameterization: str,
        b_matrix: Any,
    ) -> Dict[str, Any]:
        """Produce B-orientation metadata for matrix_provenance['B']."""
        declared = self._parse_declared_b_order(state_space_block)
        claimed = self._parse_claimed_slice_convention(parameterization)
        detected = self._detect_b_order(b_matrix)
        contradiction = False
        reason: Optional[str] = None

        # Contradiction requires decisive data (doubly-stochastic/ambiguous
        # data is NEVER a contradiction by itself): the detected orientation
        # must disagree with the declared axis order (or, absent a declaration,
        # with the claimed convention / canonical order).
        if detected is not None:
            reference = declared
            reference_label = "declared"
            if reference is None:
                reference = (
                    list(CANONICAL_B_ORDER)
                    if claimed is None
                    else (
                        ["action", "previous_state", "next_state"]
                        if claimed == "rows_previous_columns_next"
                        else list(CANONICAL_B_ORDER)
                    )
                )
                reference_label = "claimed" if claimed else "canonical"
            if detected != reference:
                contradiction = True
                reason = (
                    f"detected B orientation {detected} contradicts the "
                    f"{reference_label} order {reference}"
                )

        return {
            "declared_order": declared or list(CANONICAL_B_ORDER),
            "claimed_slice_convention": claimed,
            "detected_order": detected,
            "canonical_order": list(CANONICAL_B_ORDER),
            "contradiction": contradiction,
            "reason": reason,
        }

    def _parse_initial_parameterization(self, content: str) -> Dict[str, Any]:
        """Parse InitialParameterization section."""
        params: dict[Any, Any] = {}

        # Split content into lines and process each parameter block
        lines = content.split("\n")
        current_param = None
        current_value = ""
        in_param_block = False

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            # Skip comments
            if line.startswith("#") or not line:
                continue

            # Check if this line starts a matrix/vector block parameter definition.
            if "={" in line and not in_param_block:
                # Start of parameter block
                param_name = line.split("={")[0].strip()
                raw_value = line.split("=", 1)[1].strip()
                raw_inner = (
                    raw_value[1:-1].strip()
                    if raw_value.startswith("{") and raw_value.endswith("}")
                    else ""
                )
                if ":" in raw_inner:
                    try:
                        params[param_name] = self._parse_assignment_value(raw_value)
                    except (ValueError, SyntaxError) as e:
                        self._record_parameter_failure(param_name, e, line_no)
                    except Exception as e:  # unexpected — still never dropped
                        self._record_parameter_failure(param_name, e, line_no)
                    current_param = None
                    current_value = ""
                    continue

                current_param = param_name
                current_value = (
                    raw_value[1:] if raw_value.startswith("{") else raw_value
                )

                # Check if parameter ends on the same line
                if "}" in current_value:
                    # Single-line parameter
                    current_value = current_value.split("}")[0]
                    try:
                        parsed_value = self._parse_parameter_value(current_value)
                        params[current_param] = parsed_value
                    except (ValueError, SyntaxError) as e:
                        self._record_parameter_failure(current_param, e, line_no)
                    except Exception as e:  # unexpected — still never dropped
                        self._record_parameter_failure(current_param, e, line_no)
                    current_param = None
                    current_value = ""
                else:
                    # Multi-line parameter
                    in_param_block = True

            elif in_param_block and current_param:
                # Continue collecting parameter value
                if "}" in line:
                    # End of parameter block
                    current_value += " " + line.split("}")[0]
                    try:
                        parsed_value = self._parse_parameter_value(current_value)
                        params[current_param] = parsed_value
                    except (ValueError, SyntaxError) as e:
                        self._record_parameter_failure(current_param, e, line_no)
                    except Exception as e:  # unexpected — still never dropped
                        self._record_parameter_failure(current_param, e, line_no)
                    in_param_block = False
                    current_param = None
                    current_value = ""
                else:
                    # Add line to current value
                    current_value += " " + line

            elif "=" in line and not in_param_block:
                param_name, raw_value = line.split("=", 1)
                param_name = param_name.strip()
                raw_value = raw_value.strip()
                if not param_name:
                    continue
                try:
                    params[param_name] = self._parse_assignment_value(raw_value)
                except (ValueError, SyntaxError) as e:
                    self._record_parameter_failure(param_name, e, line_no)
                except Exception as e:  # unexpected — still never dropped
                    self._record_parameter_failure(param_name, e, line_no)

        return params

    def _parse_assignment_value(self, value_str: str) -> Any:
        """Parse a complete InitialParameterization assignment value."""
        value_str = value_str.strip()
        if value_str.startswith("{") and value_str.endswith("}"):
            inner = value_str[1:-1].strip()
            if ":" not in inner:
                value_str = inner
        return self._parse_parameter_value(value_str)

    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse parameter value string into appropriate data structure."""
        import ast

        value_str = value_str.strip()

        # Handle simple numeric values
        try:
            if re.match(r"^[-+]?\d*\.\d+$", value_str):
                return float(value_str)
            if re.match(r"^[-+]?\d+$", value_str):
                return int(value_str)
        except ValueError:
            self.logger.debug(
                "Value '%s' is not a simple numeric, trying structured formats",
                value_str[:40],
            )

        # Handle structured data (tuples/nested lists)
        if "(" in value_str or "[" in value_str:
            try:
                try:
                    from utils.safe_eval import MATRIX_MAX_LEN, safe_literal_eval
                except ImportError as e:
                    # Heavy pipeline not importable: eval-free path keeps
                    # working; a broken/suspicious safe_eval is a structured
                    # GNN-E006 fault, never a silent drop.
                    raise ImportError(
                        f"utils.safe_eval unavailable ({e}); cannot safely "
                        "evaluate structured parameter literal"
                    ) from e

                # Convert ( ) to [ ] for literal_eval if needed, or just let it handle tuples
                # Better to convert to a standard format
                clean_str = value_str.replace("(", "[").replace(")", "]")
                # Handle cases like ( (1,2), (3,4) ) -> [ [1,2], [3,4] ]
                # Remove extra commas if any (e.g., from trailing commas in GNN)
                clean_str = re.sub(r",\s*\]", "]", clean_str)
                # Matrix/tensor literals are shallow but legitimately large
                # (scaling-study B tensors reach ~2.6M chars), so use the
                # matrix length bound rather than the scalar default.
                return cast(
                    "list[Any] | float | int",
                    safe_literal_eval(clean_str, max_len=MATRIX_MAX_LEN),
                )
            except (ValueError, SyntaxError) as e:
                self.logger.warning(
                    f"ast.literal_eval failed for {value_str}: {e}. Falling back to manual parsing."
                )
                fallback = self._parse_nested_structure_safe(value_str)
                junk = self._find_string_tokens(fallback)
                if junk:
                    # Non-numeric tokens inside a bracketed literal are a
                    # genuine parse failure (e.g. '(0.05, 0.9, oops)'), not a
                    # tolerated string value: raise so the parameter is
                    # recorded as a structured parse_error, never silently
                    # degraded.
                    raise ValueError(
                        f"non-numeric token(s) {junk} in matrix literal"
                    ) from e
                return fallback

    @staticmethod
    def _find_string_tokens(value: Any) -> List[str]:
        """Collect string tokens that leaked into a parsed numeric structure."""
        found: List[str] = []
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.extend(POMDPExtractor._find_string_tokens(item))
        return found

    def _parse_nested_structure_safe(self, value_str: str) -> List:
        """
        Robust manual parser for nested structures as a last resort.
        Handles nested parentheses/brackets by tracking depth.
        """
        value_str = value_str.strip()
        if not value_str:
            return []

        result: list[Any] = []
        current = ""
        depth = 0

        # Normalize delimiters
        value_str = value_str.replace("(", "[").replace(")", "]")

        if value_str.startswith("[") and value_str.endswith("]"):
            content = value_str[1:-1].strip()
        else:
            content = value_str

        i = 0
        while i < len(content):
            char = content[i]
            if char == "[":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    # Found a complete nested group
                    group = content[start_idx : i + 1]
                    result.append(self._parse_nested_structure_safe(group))
            elif char == "," and depth == 0:
                if current.strip():
                    try:
                        val = current.strip()
                        if "." in val:
                            result.append(float(val))
                        else:
                            result.append(int(val))
                    except ValueError:
                        result.append(current.strip())
                    current = ""
            elif depth == 0:
                current += char
            i += 1

        if current.strip():
            try:
                val = current.strip()
                if "." in val:
                    result.append(float(val))
                else:
                    result.append(int(val))
            except ValueError:
                result.append(current.strip())

        return result

    def _parse_connections(self, content: str) -> List[Tuple[str, str, str]]:
        """Parse Connections section."""
        connections: list[Any] = []

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = self.CONNECTION_PATTERN.match(line)
            if match:
                source = match.group(1).strip()
                relation = match.group(2).strip()
                target = match.group(3).strip()
                connections.append((source, relation, target))

        return connections

    def _parse_ontology_annotations(self, content: str) -> Dict[str, str]:
        """Parse ActInfOntologyAnnotation section."""
        mapping: dict[Any, Any] = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    mapping[key] = value

        return mapping

    def _validate_pomdp_structure(self, pomdp_space: POMDPStateSpace) -> Dict[str, Any]:
        """Validate POMDP structure for consistency."""
        warnings: list[Any] = []

        # Check dimension consistency
        try:
            if pomdp_space.A_matrix and isinstance(pomdp_space.A_matrix, list):
                if len(pomdp_space.A_matrix) > 0 and isinstance(
                    pomdp_space.A_matrix[0], list
                ):
                    expected_a_dims = (
                        pomdp_space.num_observations,
                        pomdp_space.num_states,
                    )
                    actual_a_dims = (
                        len(pomdp_space.A_matrix),
                        len(pomdp_space.A_matrix[0]),
                    )
                    if expected_a_dims != actual_a_dims:
                        warnings.append(
                            f"A matrix dimensions {actual_a_dims} don't match expected {expected_a_dims}"
                        )
        except (TypeError, IndexError) as e:
            warnings.append(f"A matrix has invalid structure: {e}")

        try:
            if pomdp_space.B_matrix and isinstance(pomdp_space.B_matrix, list):
                if (
                    len(pomdp_space.B_matrix) > 0
                    and isinstance(pomdp_space.B_matrix[0], list)
                    and len(pomdp_space.B_matrix[0]) > 0
                    and isinstance(pomdp_space.B_matrix[0][0], list)
                ):
                    expected_b_dims = (
                        pomdp_space.num_states,
                        pomdp_space.num_states,
                        pomdp_space.num_actions,
                    )
                    actual_b_dims = (
                        len(pomdp_space.B_matrix[0]),
                        len(pomdp_space.B_matrix[0][0]),
                        len(pomdp_space.B_matrix),
                    )
                    if expected_b_dims != actual_b_dims:
                        warnings.append(
                            f"B matrix dimensions {actual_b_dims} don't match expected {expected_b_dims}"
                        )
        except (TypeError, IndexError) as e:
            warnings.append(f"B matrix has invalid structure: {e}")

        try:
            if pomdp_space.C_vector and isinstance(pomdp_space.C_vector, list):
                if len(pomdp_space.C_vector) != pomdp_space.num_observations:
                    warnings.append(
                        f"C vector length {len(pomdp_space.C_vector)} doesn't match num_observations {pomdp_space.num_observations}"
                    )
        except TypeError as e:
            warnings.append(f"C vector has invalid structure: {e}")

        try:
            if pomdp_space.D_vector and isinstance(pomdp_space.D_vector, list):
                if len(pomdp_space.D_vector) != pomdp_space.num_states:
                    warnings.append(
                        f"D vector length {len(pomdp_space.D_vector)} doesn't match num_states {pomdp_space.num_states}"
                    )
        except TypeError as e:
            warnings.append(f"D vector has invalid structure: {e}")

        return {"valid": len(warnings) == 0, "warnings": warnings}


@overload
def extract_pomdp_from_file(
    file_path: Union[str, Path],
    strict_validation: bool = ...,
    *,
    on_error: Literal["lenient", "raise"] = ...,
    insert_default_c: bool = ...,
) -> Optional[POMDPStateSpace]: ...


@overload
def extract_pomdp_from_file(
    file_path: Union[str, Path],
    strict_validation: bool = ...,
    *,
    on_error: Literal["collect"],
    insert_default_c: bool = ...,
) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...


def extract_pomdp_from_file(
    file_path: Union[str, Path],
    strict_validation: bool = True,
    *,
    on_error: str = "lenient",
    insert_default_c: bool = True,
) -> Union[
    Optional[POMDPStateSpace],
    Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
]:
    """
    Convenience function to extract POMDP state space from a GNN file.

    Args:
        file_path: Path to GNN file
        strict_validation: Enable strict validation
        on_error: 'lenient' (default) | 'raise' | 'collect'. 'raise' raises
            GNNExtractionError at the first fault; 'collect' returns
            (spec_or_None, list[GNNExtractionError]); invalid -> ValueError.
        insert_default_c: True (default) preserves the passive-model zero-C
            adapter; False keeps C None with no adapter provenance.

    Returns:
        POMDPStateSpace (lenient/raise) or a (spec | None, errors) tuple
        (collect mode).
    """
    extractor = POMDPExtractor(strict_validation=strict_validation)
    return extractor.extract_from_file(
        file_path,
        on_error=cast(OnErrorMode, on_error),
        insert_default_c=insert_default_c,
    )


@overload
def extract_pomdp_from_content(
    content: str,
    strict_validation: bool = ...,
    *,
    on_error: Literal["lenient", "raise"] = ...,
    insert_default_c: bool = ...,
) -> Optional[POMDPStateSpace]: ...


@overload
def extract_pomdp_from_content(
    content: str,
    strict_validation: bool = ...,
    *,
    on_error: Literal["collect"],
    insert_default_c: bool = ...,
) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...


def extract_pomdp_from_content(
    content: str,
    strict_validation: bool = True,
    *,
    on_error: str = "lenient",
    insert_default_c: bool = True,
) -> Union[
    Optional[POMDPStateSpace],
    Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
]:
    """
    Convenience function to extract POMDP state space from GNN content.

    Args:
        content: GNN file content
        strict_validation: Enable strict validation
        on_error: 'lenient' (default) | 'raise' | 'collect'. 'raise' raises
            GNNExtractionError at the first fault; 'collect' returns
            (spec_or_None, list[GNNExtractionError]); invalid -> ValueError.
        insert_default_c: True (default) preserves the passive-model zero-C
            adapter; False keeps C None with no adapter provenance.

    Returns:
        POMDPStateSpace (lenient/raise) or a (spec | None, errors) tuple
        (collect mode).
    """
    extractor = POMDPExtractor(strict_validation=strict_validation)
    return extractor.extract_from_gnn_content(
        content,
        on_error=cast(OnErrorMode, on_error),
        insert_default_c=insert_default_c,
    )


def canonicalize_pomdp(spec: POMDPStateSpace) -> POMDPStateSpace:
    """Return a NEW POMDPStateSpace with B in canonical (next, prev, action) order.

    Pure, stdlib-only copy: the input spec (and its B_matrix) is never
    mutated. Orientation is chosen from matrix_provenance['B']
    (detected_order/claimed_slice_convention) when decisive; a 3-D B stored
    as (action, previous_state, next_state) is transposed to
    (next_state, previous_state, action); canonical or ambiguous storage is
    copied unchanged. All other fields are copied as-is.
    """
    canonical = POMDPStateSpace(
        **{field: getattr(spec, field) for field in spec.__dataclass_fields__}
    )
    b_matrix = spec.B_matrix
    provenance = spec.matrix_provenance or {}
    b_meta = provenance.get("B") or {}
    stored_order = b_meta.get("detected_order") or (
        ["action", "previous_state", "next_state"]
        if b_meta.get("claimed_slice_convention") == "rows_previous_columns_next"
        else None
    )
    if (
        isinstance(b_matrix, (list, tuple))
        and len(self_shape := _shape_of(b_matrix)) == 3
        and stored_order
        and stored_order != list(CANONICAL_B_ORDER)
        and stored_order == ["action", "previous_state", "next_state"]
    ):
        # (action, prev, next) -> (next, prev, action):
        # canonical[n][p][a] = stored[a][p][n]
        canonical.B_matrix = [
            [
                [b_matrix[a][p][n] for a in range(self_shape[2])]
                for p in range(self_shape[1])
            ]
            for n in range(self_shape[0])
        ]
    return canonical


def _shape_of(value: Any) -> List[int]:
    """Best-effort nested shape (module-level twin of _nested_shape)."""
    shape: List[int] = []
    current: Any = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape
