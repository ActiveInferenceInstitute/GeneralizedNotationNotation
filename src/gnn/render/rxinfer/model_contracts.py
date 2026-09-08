"""RxInfer model-contract helpers migrated from the retired TOML emitter.

The retired ``render_gnn_to_rxinfer_toml`` entry point (and its private code
generator) were deleted with the dead ``rxinfer_toml`` render target. What
remains live here:

- GNN matrix/vector literal parsers (``_parse_gnn_matrix``,
  ``_parse_gnn_3d_matrix``, ``_parse_gnn_vector``) and their string helpers;
- Julia literal formatters, the exact-formatting TOML writer, the compact
  multi-agent config-structure builder (``agent_ids`` /
  ``agent_initial_positions`` / ``agent_target_positions``), and the
  fail-closed topology validation (``_validate_topology_references``).

These are pinned by ``tests/render/test_toml_matrix_parser.py``,
``tests/render/test_rxinfer_multiagent_contract.py``, and the capability
contract markers in ``scripts/check_capability_contracts.py``.
"""

import json
from typing import Any, Dict, List


def _strip_outer_braces(s: str) -> str:
    """Remove one layer of outer curly braces if the whole string is wrapped."""
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1].strip()
    return s


def _strip_outer_parens_wrapper(s: str) -> str:
    """
    Remove one outer pair of parentheses only when they wrap the entire content
    (the matching close paren is the final character). This lets us accept both
    '((a,b,c),(d,e,f))' and '(a,b,c),(d,e,f)' forms while leaving partially
    parenthesized input untouched.
    """
    s = s.strip()
    if not s.startswith("("):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if i == len(s) - 1:
                    return s[1:-1].strip()
                return s
    return s


def _split_top_level_groups(content: str) -> List[str]:
    """
    Split a string into top-level comma-separated groups, counting commas at
    parenthesis/brace depth 0 so that commas inside nested tuples are preserved.
    """
    groups: List[str] = []
    depth = 0
    brace_depth = 0
    start = 0
    for i, ch in enumerate(content):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == "," and depth == 0 and brace_depth == 0:
            groups.append(content[start:i].strip())
            start = i + 1
    groups.append(content[start:].strip())
    return groups


def _parse_float_row(token: str, context_name: str) -> List[float]:
    """Parse a single tuple token like '(0.9,0.05,0.05)' into a list of floats."""
    token = token.strip()
    if not token:
        raise ValueError(f"Failed to parse {context_name}: empty element")
    token = _strip_outer_parens_wrapper(token)
    elements = [float(x.strip()) for x in token.split(",") if x.strip() != ""]
    if not elements:
        raise ValueError(
            f"Failed to parse {context_name}: no numeric elements in {token!r}"
        )
    return elements


def _parse_gnn_matrix(matrix_str: str) -> List[List[float]]:
    """
    Parse GNN matrix notation into a Python list of lists.

    Accepts the parenthesized-tuple forms used in GNN exemplar files, e.g.:
      '((0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9))'
      '{(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)}'
    as well as the multi-line brace form:
      '{\n  (1.0, 0.0, 0.0),\n  (0.0, 1.0, 0.0)\n}'

    Raises:
        ValueError: if the input cannot be parsed into a rectangular matrix.
    """
    s = _strip_outer_braces(matrix_str)
    s = _strip_outer_parens_wrapper(s)
    row_tokens = _split_top_level_groups(s)
    if not row_tokens or all(not t for t in row_tokens):
        raise ValueError(f"Failed to parse matrix {matrix_str!r}: no rows found")

    matrix: List[List[float]] = []
    for token in row_tokens:
        matrix.append(_parse_float_row(token, f"matrix {matrix_str!r}"))

    # Validate the matrix is rectangular.
    if matrix:
        ncols = len(matrix[0])
        for row in matrix:
            if len(row) != ncols:
                raise ValueError(
                    f"Failed to parse matrix {matrix_str!r}: "
                    f"rows have inconsistent lengths"
                )
    return matrix


def _parse_gnn_3d_matrix(matrix_str: str) -> List[List[List[float]]]:
    """
    Parse GNN 3D matrix notation (e.g. the B transition tensor) into a Python
    list of 2D matrices.

    Accepts the nested parenthesized form used in the exemplars, e.g.:
      '{( (0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9) ), ( (...),... )}'

    Raises:
        ValueError: if the input cannot be parsed into a 3D tensor.
    """
    s = _strip_outer_braces(matrix_str)
    s = _strip_outer_parens_wrapper(s)
    action_tokens = _split_top_level_groups(s)
    if not action_tokens or all(not t for t in action_tokens):
        raise ValueError(
            f"Failed to parse 3D matrix {matrix_str!r}: no action matrices found"
        )

    tensor: List[List[List[float]]] = []
    for token in action_tokens:
        tensor.append(_parse_gnn_matrix(token))
    return tensor


def _parse_gnn_vector(vector_str: str) -> List[float]:
    """
    Parse GNN vector notation into a Python list of floats.

    Accepts e.g. '(0.25, 0.25, 0.25, 0.25)' or '{(0.0, 0.0, 0.0, 3.0)}'.

    Raises:
        ValueError: if the input cannot be parsed.
    """
    s = _strip_outer_braces(vector_str)
    s = _strip_outer_parens_wrapper(s)
    elements = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not elements:
        raise ValueError(f"Failed to parse vector {vector_str!r}: no numeric elements")
    return elements


def _matrix_to_julia(matrix: List[List[float]]) -> str:
    """Convert Python matrix to Julia matrix string."""
    rows: list[Any] = []
    for row in matrix:
        row_str = "[" + ", ".join(str(x) for x in row) + "]"
        rows.append(row_str)
    return "[" + ", ".join(rows) + "]"


def _tensor_to_julia(tensor: List[List[List[float]]]) -> str:
    """Convert Python 3D tensor to Julia tensor string."""
    if not tensor:
        return "[]"

    # Handle 3D tensor (actions x states x states)
    action_matrices: list[Any] = []
    for action_matrix in tensor:
        matrix_str = _matrix_to_julia(action_matrix)
        action_matrices.append(matrix_str)

    return "[" + ", ".join(action_matrices) + "]"


def _vector_to_julia(vector: List[float]) -> str:
    """Convert Python vector to Julia vector string."""
    return "[" + ", ".join(str(x) for x in vector) + "]"


def _create_dirichlet_prior(matrix: List[List[float]]) -> str:
    """Create Dirichlet prior for matrix."""
    rows: list[Any] = []
    for row in matrix:
        # Add small regularization to avoid zeros
        regularized_row = [x + 0.1 for x in row]
        row_str = "[" + ", ".join(str(x) for x in regularized_row) + "]"
        rows.append(row_str)
    return "[" + ", ".join(rows) + "]"


def _create_dirichlet_prior_3d(tensor: List[List[List[float]]]) -> str:
    """Create Dirichlet prior for 3D tensor."""
    if not tensor:
        return "[]"

    action_matrices: list[Any] = []
    for action_matrix in tensor:
        matrix_str = _create_dirichlet_prior(action_matrix)
        action_matrices.append(matrix_str)

    return "[" + ", ".join(action_matrices) + "]"


def _write_toml_with_exact_formatting(f: Any, config: Any) -> Any:
    """
    Write TOML with exact formatting to match the gold standard.
    This function writes sections in a specific order with comments and formatting.
    """
    # Model section
    f.write("#\n# Model parameters\n#\n")
    f.write("[model]\n")

    # Write model parameters
    f.write("# Time step for the state space model\n")
    f.write(f"dt = {config['model']['dt']}\n\n")

    f.write("# Constraint parameter for the Halfspace node\n")
    f.write(f"gamma = {config['model']['gamma']}\n\n")

    f.write("# Number of time steps in the trajectory\n")
    f.write(f"nr_steps = {config['model']['nr_steps']}\n\n")

    f.write("# Number of inference iterations\n")
    f.write(f"nr_iterations = {config['model']['nr_iterations']}\n\n")

    f.write("# Number of agents in the simulation\n")
    f.write(f"nr_agents = {config['model']['nr_agents']}\n\n")

    f.write("# Temperature parameter for the softmin function\n")
    f.write(f"softmin_temperature = {config['model']['softmin_temperature']}\n\n")

    f.write("# Intermediate results saving interval (every N iterations)\n")
    f.write(f"intermediate_steps = {config['model']['intermediate_steps']}\n\n")

    f.write("# Whether to save intermediate results\n")
    f.write(
        f"save_intermediates = {str(config['model']['save_intermediates']).lower()}\n\n"
    )

    # Matrices section
    f.write("#\n# State Space Matrices\n#\n")
    f.write("[model.matrices]\n")

    # State transition matrix
    f.write("# State transition matrix\n")
    f.write("# [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]\n")
    f.write("A = [\n")
    for i, row in enumerate(config["model"]["matrices"]["A"]):
        f.write(f"    {row}")
        if i < len(config["model"]["matrices"]["A"]) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n\n")

    # Control input matrix
    f.write("# Control input matrix\n")
    f.write("# [0 0; dt 0; 0 0; 0 dt]\n")
    f.write("B = [\n")
    for i, row in enumerate(config["model"]["matrices"]["B"]):
        f.write(f"    {row}")
        if i < len(config["model"]["matrices"]["B"]) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n\n")

    # Observation matrix
    f.write("# Observation matrix\n")
    f.write("# [1 0 0 0; 0 0 1 0]\n")
    f.write("C = [\n")
    for i, row in enumerate(config["model"]["matrices"]["C"]):
        f.write(f"    {row}")
        if i < len(config["model"]["matrices"]["C"]) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n\n")

    # Priors section
    f.write("#\n# Prior distributions\n#\n")
    f.write("[priors]\n")

    f.write("# Prior on initial state\n")
    f.write(
        f"initial_state_variance = {config['priors']['initial_state_variance']}\n\n"
    )

    f.write("# Prior on control inputs\n")
    f.write(f"control_variance = {config['priors']['control_variance']}\n\n")

    f.write("# Goal constraints variance\n")
    # Use scientific notation with exact format to match gold standard
    f.write(
        f"goal_constraint_variance = {config['priors']['goal_constraint_variance']:.1e}\n\n"
    )

    f.write("# Parameters for GammaShapeRate prior on constraint parameters\n")
    f.write(f"gamma_shape = {config['priors']['gamma_shape']}  # 3/2\n")
    f.write(
        f"gamma_scale_factor = {config['priors']['gamma_scale_factor']}  # γ^2/2\n\n"
    )

    # Visualization section
    f.write("#\n# Visualization parameters\n#\n")
    f.write("[visualization]\n")

    f.write("# Plot boundaries\n")
    f.write(f"x_limits = {config['visualization']['x_limits']}\n")
    f.write(f"y_limits = {config['visualization']['y_limits']}\n\n")

    f.write("# Animation frames per second\n")
    f.write(f"fps = {config['visualization']['fps']}\n\n")

    f.write("# Heatmap resolution\n")
    f.write(f"heatmap_resolution = {config['visualization']['heatmap_resolution']}\n\n")

    f.write("# Plot size\n")
    f.write(f"plot_width = {config['visualization']['plot_width']}\n")
    f.write(f"plot_height = {config['visualization']['plot_height']}\n\n")

    f.write("# Visualization alpha values\n")
    f.write(f"agent_alpha = {config['visualization']['agent_alpha']}\n")
    f.write(f"target_alpha = {config['visualization']['target_alpha']}\n\n")

    f.write("# Color palette\n")
    f.write(f'color_palette = "{config["visualization"]["color_palette"]}"\n\n')

    # Environments section
    f.write("#\n# Environment definitions\n#\n")

    # Door environment
    if "door" in config["environments"] and config["environments"]["door"]["obstacles"]:
        env = config["environments"]["door"]
        f.write("[environments.door]\n")
        f.write(f'description = "{env["description"]}"\n\n')
        for obstacle in env["obstacles"]:
            f.write("[[environments.door.obstacles]]\n")
            f.write(f"center = {obstacle['center']}\n")
            f.write(f"size = {obstacle['size']}\n\n")

    # Wall environment
    if "wall" in config["environments"] and config["environments"]["wall"]["obstacles"]:
        env = config["environments"]["wall"]
        f.write("[environments.wall]\n")
        f.write(f'description = "{env["description"]}"\n\n')
        for obstacle in env["obstacles"]:
            f.write("[[environments.wall.obstacles]]\n")
            f.write(f"center = {obstacle['center']}\n")
            f.write(f"size = {obstacle['size']}\n\n")

    # Combined environment
    if (
        "combined" in config["environments"]
        and config["environments"]["combined"]["obstacles"]
    ):
        env = config["environments"]["combined"]
        f.write("[environments.combined]\n")
        f.write(f'description = "{env["description"]}"\n\n')
        for obstacle in env["obstacles"]:
            f.write("[[environments.combined.obstacles]]\n")
            f.write(f"center = {obstacle['center']}\n")
            f.write(f"size = {obstacle['size']}\n\n")

    # Agents section
    f.write("#\n# Agent configurations\n#\n")

    for agent in config["agents"]:
        f.write("[[agents]]\n")
        f.write(f"id = {json_dumps_scalar(agent['id'])}\n")
        f.write(f"radius = {agent['radius']}\n")
        f.write(f"initial_position = {agent['initial_position']}\n")
        f.write(f"target_position = {agent['target_position']}\n\n")

    # Agent topology section
    topology = config.get("topology")
    if isinstance(topology, dict):
        f.write("#\n# Agent topology\n#\n")
        f.write("[topology]\n")
        f.write(f'type = "{topology.get("type", "agent_population")}"\n')
        f.write(f"agent_ids = {_toml_array(topology.get('agent_ids', []))}\n")
        if topology.get("message_passing"):
            f.write(f'message_passing = "{topology["message_passing"]}"\n')
        f.write("\n")
        for edge in topology.get("edges", []):
            if isinstance(edge, dict):
                f.write("[[topology.edges]]\n")
                f.write(f"source = {json_dumps_scalar(edge['source'])}\n")
                f.write(f"target = {json_dumps_scalar(edge['target'])}\n\n")
        for cluster in topology.get("clusters", []):
            if isinstance(cluster, dict):
                f.write("[[topology.clusters]]\n")
                f.write(f'name = "{cluster["name"]}"\n')
                f.write(f"agent_ids = {_toml_array(cluster.get('agent_ids', []))}\n\n")

    # Experiments section
    f.write("#\n# Experiment configurations\n#\n")
    f.write("[experiments]\n")

    f.write("# Random seeds for reproducibility\n")
    f.write(f"seeds = {config['experiments']['seeds']}\n\n")

    f.write("# Base directory for results\n")
    f.write(f'results_dir = "{config["experiments"]["results_dir"]}"\n\n')

    f.write("# Filename templates\n")
    f.write(f'animation_template = "{config["experiments"]["animation_template"]}"\n')
    f.write(
        f'control_vis_filename = "{config["experiments"]["control_vis_filename"]}"\n'
    )
    f.write(
        f'obstacle_distance_filename = "{config["experiments"]["obstacle_distance_filename"]}"\n'
    )
    f.write(
        f'path_uncertainty_filename = "{config["experiments"]["path_uncertainty_filename"]}"\n'
    )
    # Add a space at the end of the last line to match gold standard
    f.write(
        f'convergence_filename = "{config["experiments"]["convergence_filename"]}" '
    )


def _create_toml_config_structure(
    gnn_spec: Dict[str, Any], options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create the TOML configuration structure from a GNN specification.

    Args:
        gnn_spec: Dictionary containing the GNN specification
        options: Additional options for TOML generation

    Returns:
        Dictionary representing the TOML configuration
    """
    params = _initial_parameterization(gnn_spec)
    agents = _extract_agents(gnn_spec)

    # Start with a standard structure based on the config.toml example
    toml_config: dict[str, Any] = {
        "model": {
            "dt": params.get("dt", 1.0),
            "gamma": params.get("gamma", 1.0),
            "nr_steps": params.get("nr_steps", 40),
            "nr_iterations": params.get("nr_iterations", 350),
            "nr_agents": len(agents),
            "softmin_temperature": params.get("softmin_temperature", 10.0),
            "intermediate_steps": params.get("intermediate_steps", 10),
            "save_intermediates": str(params.get("save_intermediates", False))
            .lower()
            .strip()
            .startswith("true"),
            "matrices": _extract_matrices(gnn_spec),
        },
        "priors": {
            "initial_state_variance": params.get("initial_state_variance", 100.0),
            "control_variance": params.get("control_variance", 0.1),
            "goal_constraint_variance": params.get("goal_constraint_variance", 1e-5),
            "gamma_shape": params.get("gamma_shape", 1.5),
            "gamma_scale_factor": params.get("gamma_scale_factor", 0.5),
        },
        "visualization": {
            "x_limits": params.get("x_limits", [-20, 20]),
            "y_limits": params.get("y_limits", [-20, 20]),
            "fps": params.get("fps", 15),
            "heatmap_resolution": params.get("heatmap_resolution", 100),
            "plot_width": params.get("plot_width", 800),
            "plot_height": params.get("plot_height", 400),
            "agent_alpha": params.get("agent_alpha", 1.0),
            "target_alpha": params.get("target_alpha", 0.2),
            "color_palette": params.get("color_palette", "tab10"),
        },
        "environments": _extract_environments(gnn_spec),
        "agents": agents,
        "topology": _extract_agent_topology(params, agents),
        "experiments": _extract_experiments(gnn_spec),
    }

    return toml_config


def _toml_array(value: Any) -> str:
    """Return a TOML-compatible array literal for simple scalar lists."""
    if not isinstance(value, list):
        return "[]"
    return "[" + ", ".join(json_dumps_scalar(item) for item in value) + "]"


def json_dumps_scalar(value: Any) -> str:
    """Serialize scalar values using TOML-compatible JSON scalar syntax."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value)
    if isinstance(value, list):
        return _toml_array(value)
    return json.dumps(str(value))


def _initial_parameterization(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized initial-parameterization keys from public GNN specs."""
    for key in (
        "initialparameterization",
        "initial_parameterization",
        "InitialParameterization",
    ):
        value = gnn_spec.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _extract_matrices(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract state space matrices from the GNN specification."""
    matrices: dict[Any, Any] = {}
    params = _initial_parameterization(gnn_spec)

    # Use provided matrices if available, otherwise use defaults
    if "A" in params:
        matrices["A"] = params["A"]
    else:
        # Default state transition matrix [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
        matrices["A"] = [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    if "B" in params:
        matrices["B"] = params["B"]
    else:
        # Default control input matrix [0 0; dt 0; 0 0; 0 dt]
        matrices["B"] = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]

    if "C" in params:
        matrices["C"] = params["C"]
    else:
        # Default observation matrix [1 0 0 0; 0 0 1 0]
        matrices["C"] = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

    return matrices


def _extract_environments(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract environment definitions from the GNN specification."""
    params = _initial_parameterization(gnn_spec)

    environments: dict[str, Any] = {
        "door": {
            "description": "Two parallel walls with a gap between them",
            "obstacles": [],
        },
        "wall": {
            "description": "A single wall obstacle in the center",
            "obstacles": [],
        },
        "combined": {
            "description": "A combination of walls and obstacles",
            "obstacles": [],
        },
    }

    if "door_obstacle_center_1" in params and "door_obstacle_size_1" in params:
        environments["door"]["obstacles"].append(
            {
                "center": params["door_obstacle_center_1"],
                "size": params["door_obstacle_size_1"],
            }
        )
    if "door_obstacle_center_2" in params and "door_obstacle_size_2" in params:
        environments["door"]["obstacles"].append(
            {
                "center": params["door_obstacle_center_2"],
                "size": params["door_obstacle_size_2"],
            }
        )

    if "wall_obstacle_center" in params and "wall_obstacle_size" in params:
        environments["wall"]["obstacles"].append(
            {
                "center": params["wall_obstacle_center"],
                "size": params["wall_obstacle_size"],
            }
        )

    if "combined_obstacle_center_1" in params and "combined_obstacle_size_1" in params:
        environments["combined"]["obstacles"].append(
            {
                "center": params["combined_obstacle_center_1"],
                "size": params["combined_obstacle_size_1"],
            }
        )
    if "combined_obstacle_center_2" in params and "combined_obstacle_size_2" in params:
        environments["combined"]["obstacles"].append(
            {
                "center": params["combined_obstacle_center_2"],
                "size": params["combined_obstacle_size_2"],
            }
        )
    if "combined_obstacle_center_3" in params and "combined_obstacle_size_3" in params:
        environments["combined"]["obstacles"].append(
            {
                "center": params["combined_obstacle_center_3"],
                "size": params["combined_obstacle_size_3"],
            }
        )

    return environments


def _extract_agents(gnn_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract agent configurations from the GNN specification."""
    params = _initial_parameterization(gnn_spec)
    nr_agents = _coerce_positive_int(params.get("nr_agents"))

    if nr_agents > 0:
        compact_agents = _extract_compact_agents(params, nr_agents)
        if compact_agents is not None:
            return compact_agents

        indexed_agents = _extract_indexed_agents(params, nr_agents)
        if len(indexed_agents) == nr_agents:
            return indexed_agents

        raise ValueError(
            "nr_agents was provided but agent configuration is incomplete. "
            "Provide compact agent_ids/agent_initial_positions/agent_target_positions "
            "or complete agent{i}_id/agent{i}_initial_position/agent{i}_target_position keys."
        )

    # Recovery to default agents if extraction fails
    return [
        {
            "id": 1,
            "radius": 2.5,
            "initial_position": [-4.0, 10.0],
            "target_position": [-10.0, -10.0],
        },
        {
            "id": 2,
            "radius": 1.5,
            "initial_position": [-10.0, 5.0],
            "target_position": [10.0, -15.0],
        },
        {
            "id": 3,
            "radius": 1.0,
            "initial_position": [-15.0, -10.0],
            "target_position": [10.0, 10.0],
        },
        {
            "id": 4,
            "radius": 2.5,
            "initial_position": [0.0, -10.0],
            "target_position": [-10.0, 15.0],
        },
    ]


def _coerce_positive_int(value: Any) -> int:
    """Coerce a value to a positive int, returning 0 for missing/invalid values."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)


def _as_list(value: Any) -> list[Any] | None:
    """Return value as a list when it is list-like enough for TOML config."""
    return value if isinstance(value, list) else None


def _extract_compact_agents(
    params: Dict[str, Any], nr_agents: int
) -> List[Dict[str, Any]] | None:
    """Extract agents from compact vectorized InitialParameterization keys."""
    agent_ids = _as_list(params.get("agent_ids"))
    initial_positions = _as_list(params.get("agent_initial_positions"))
    target_positions = _as_list(params.get("agent_target_positions"))
    if agent_ids is None and initial_positions is None and target_positions is None:
        return None
    radii = _as_list(params.get("agent_radii")) or _as_list(params.get("agent_radius"))
    default_radius = params.get("agent_default_radius", 1.0)
    required = {
        "agent_ids": agent_ids,
        "agent_initial_positions": initial_positions,
        "agent_target_positions": target_positions,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing compact multi-agent keys: {', '.join(missing)}")
    assert agent_ids is not None
    assert initial_positions is not None
    assert target_positions is not None
    lengths = {
        "agent_ids": len(agent_ids),
        "agent_initial_positions": len(initial_positions),
        "agent_target_positions": len(target_positions),
    }
    if any(length != nr_agents for length in lengths.values()):
        raise ValueError(
            f"Compact multi-agent lengths must match nr_agents={nr_agents}: {lengths}"
        )
    if radii is not None and len(radii) != nr_agents:
        raise ValueError(
            f"agent_radii length {len(radii)} must match nr_agents={nr_agents}"
        )
    return [
        {
            "id": agent_ids[index],
            "radius": radii[index] if radii is not None else default_radius,
            "initial_position": initial_positions[index],
            "target_position": target_positions[index],
        }
        for index in range(nr_agents)
    ]


def _extract_indexed_agents(
    params: Dict[str, Any], nr_agents: int
) -> List[Dict[str, Any]]:
    """Extract indexed agent{i}_... agent definitions."""
    agents: List[Dict[str, Any]] = []
    for i in range(1, nr_agents + 1):
        agent_id = params.get(f"agent{i}_id")
        radius = params.get(f"agent{i}_radius", params.get("agent_default_radius", 1.0))
        initial_pos = params.get(f"agent{i}_initial_position")
        target_pos = params.get(f"agent{i}_target_position")

        if all(v is not None for v in [agent_id, radius, initial_pos, target_pos]):
            agents.append(
                {
                    "id": agent_id,
                    "radius": radius,
                    "initial_position": initial_pos,
                    "target_position": target_pos,
                }
            )
    return agents


def _extract_agent_topology(
    params: Dict[str, Any], agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract explicit multi-agent topology metadata for TOML and execution."""
    agent_ids = [agent["id"] for agent in agents if "id" in agent]
    raw_edges, edges_present = _first_present(
        params, ("agent_edges", "topology_edges", "edges"), []
    )
    raw_clusters, clusters_present = _first_present(
        params, ("agent_clusters", "topology_clusters", "clusters"), {}
    )
    edges = _normalize_topology_edges(raw_edges, strict=edges_present)
    clusters = _normalize_topology_clusters(raw_clusters, strict=clusters_present)
    topology_type = params.get("agent_topology_type") or params.get("topology_type")
    if topology_type is None:
        if clusters:
            topology_type = "clustered"
        elif edges:
            topology_type = "network"
        else:
            topology_type = "agent_population"
    topology: Dict[str, Any] = {
        "type": str(topology_type),
        "agent_ids": agent_ids,
        "edges": edges,
        "clusters": clusters,
    }
    message_passing = params.get("message_passing") or params.get(
        "agent_message_passing"
    )
    if message_passing:
        topology["message_passing"] = str(message_passing)
    _validate_topology_references(topology, agent_ids)
    return topology


def _validate_topology_references(
    topology: Dict[str, Any], agent_ids: List[Any]
) -> None:
    """Reject topology records that reference undeclared agents."""
    declared = set(agent_ids)
    for edge in topology.get("edges", []):
        if not isinstance(edge, dict):
            continue
        for endpoint_name in ("source", "target"):
            endpoint = edge.get(endpoint_name)
            if endpoint not in declared:
                raise ValueError(
                    f"Topology edge {endpoint_name} references undeclared agent {endpoint!r}"
                )
    for cluster in topology.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        for agent_id in cluster.get("agent_ids", []):
            if agent_id not in declared:
                raise ValueError(
                    f"Topology cluster {cluster.get('name', '<unnamed>')} "
                    f"references undeclared agent {agent_id!r}"
                )


def _first_present(
    params: Dict[str, Any], keys: tuple[str, ...], default: Any
) -> tuple[Any, bool]:
    """Return the first present parameter value and whether any key was present."""
    for key in keys:
        if key in params:
            return params[key], True
    return default, False


def _normalize_topology_edges(
    raw_edges: Any, *, strict: bool = False
) -> List[Dict[str, Any]]:
    """Normalize compact edge lists into explicit source/target records."""
    if not isinstance(raw_edges, list):
        if strict:
            raise ValueError("Topology edges must be a list")
        return []
    edges: List[Dict[str, Any]] = []
    for raw_edge in raw_edges:
        if isinstance(raw_edge, dict):
            source = raw_edge.get("source", raw_edge.get("from"))
            target = raw_edge.get("target", raw_edge.get("to"))
        elif isinstance(raw_edge, (list, tuple)) and len(raw_edge) >= 2:
            source, target = raw_edge[0], raw_edge[1]
        else:
            if strict:
                raise ValueError(f"Malformed topology edge: {raw_edge!r}")
            continue
        if source is None or target is None:
            if strict:
                raise ValueError(
                    f"Topology edge requires source and target: {raw_edge!r}"
                )
            continue
        edges.append({"source": source, "target": target})
    return edges


def _normalize_topology_clusters(
    raw_clusters: Any, *, strict: bool = False
) -> List[Dict[str, Any]]:
    """Normalize dict/list cluster definitions into TOML table records."""
    clusters: List[Dict[str, Any]] = []
    if isinstance(raw_clusters, dict):
        for name, agent_ids in raw_clusters.items():
            if isinstance(agent_ids, list):
                clusters.append({"name": str(name), "agent_ids": agent_ids})
            elif strict:
                raise ValueError(f"Topology cluster {name!r} members must be a list")
        return clusters
    if isinstance(raw_clusters, list):
        for index, raw_cluster in enumerate(raw_clusters, start=1):
            if isinstance(raw_cluster, dict):
                agent_ids = raw_cluster.get("agent_ids") or raw_cluster.get("agents")
                if isinstance(agent_ids, list):
                    clusters.append(
                        {
                            "name": str(raw_cluster.get("name", f"cluster_{index}")),
                            "agent_ids": agent_ids,
                        }
                    )
                elif strict:
                    raise ValueError(
                        f"Topology cluster {raw_cluster.get('name', index)!r} "
                        "members must be a list"
                    )
            elif strict:
                raise ValueError(f"Malformed topology cluster: {raw_cluster!r}")
        return clusters
    if strict:
        raise ValueError("Topology clusters must be a dict or list")
    return clusters


def _extract_experiments(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract experiment configurations from the GNN specification."""
    params = _initial_parameterization(gnn_spec)

    # Use experiment settings from GNN spec if available, otherwise use defaults
    experiments: dict[str, Any] = {
        "seeds": params.get("experiment_seeds", [42, 123]),
        "results_dir": params.get("results_dir", "results"),
        "animation_template": params.get(
            "animation_template", "{environment}_{seed}.gif"
        ),
        "control_vis_filename": params.get(
            "control_vis_filename", "control_signals.gif"
        ),
        "obstacle_distance_filename": params.get(
            "obstacle_distance_filename", "obstacle_distance.png"
        ),
        "path_uncertainty_filename": params.get(
            "path_uncertainty_filename", "path_uncertainty.png"
        ),
        "convergence_filename": params.get("convergence_filename", "convergence.png"),
    }
    return experiments
