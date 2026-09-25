#!/usr/bin/env python3
"""
Section and block parsing for the POMDP extractor.

Mechanical extraction from ``gnn.extract.pomdp_extractor`` (M-01 band split):
``POMDPSectionsMixin`` holds the verbatim GNN section/block parsing methods —
section splitting, model name and annotation extraction, StateSpaceBlock
parsing, dimension extraction, dimension provenance, connection parsing, and
ontology annotation parsing.

This module is stdlib-only at import time.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .pomdp_support import POMDPExtractorSupportMixin

if TYPE_CHECKING:
    import logging
    import re


class POMDPSectionsMixin(POMDPExtractorSupportMixin):
    """Verbatim section-parsing methods moved from ``POMDPExtractor``."""

    if TYPE_CHECKING:
        SECTION_PATTERN: re.Pattern[str]
        VARIABLE_PATTERN: re.Pattern[str]
        CONNECTION_PATTERN: re.Pattern[str]
        _dimension_sources: Dict[str, str]
        logger: logging.Logger

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
