"""Source intent must survive parser, direct renderer and Step 11 admission."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.render.execution_contracts import unsupported_contract
from gnn.render.planning_admission import unsupported_planning
from gnn.render.pomdp_processor import POMDPRenderProcessor
from gnn.render.processor import render_gnn_spec

ROOT = Path(__file__).resolve().parents[2]
DEEP = ROOT / "input/gnn_files/discrete/deep_planning_horizon.md"
PLANNERS = ("pymdp", "jax", "rxinfer", "activeinference_jl", "pytorch", "numpyro")
DIRECT = (
    ("pymdp.pymdp_renderer", "render_gnn_to_pymdp"),
    ("jax.jax_renderer", "render_gnn_to_jax"),
    ("jax.jax_renderer", "render_gnn_to_jax_factorized"),
    ("jax.jax_renderer", "render_gnn_to_jax_pomdp"),
    ("jax.jax_renderer", "render_gnn_to_jax_combined"),
    ("rxinfer.rxinfer_renderer", "render_gnn_to_rxinfer"),
    ("activeinference_jl.activeinference_renderer", "render_gnn_to_activeinference_jl"),
    ("pytorch.pytorch_renderer", "render_gnn_to_pytorch"),
    ("numpyro.numpyro_renderer", "render_gnn_to_numpyro"),
)


def parsed_deep():
    result = MarkdownGNNParser().parse_string(DEEP.read_text())
    assert result.success
    spec = result.model.to_dict()
    spec["model_name"] = "renamed_authored_planning_example"
    assert spec["model_parameters"]["planning_horizon"] == 5
    assert spec["model_parameters"]["num_policies"] == 64
    return spec


@pytest.mark.parametrize("backend", PLANNERS)
def test_real_source_is_unsupported_at_both_pipeline_interfaces(backend, tmp_path):
    original = DEEP.read_bytes()
    spec = parsed_deep()
    before = deepcopy(spec)
    ok, reason, files = render_gnn_spec(spec, backend, tmp_path / "single")
    assert not ok and not files and reason.startswith("unsupported-planning:")
    assert "=5" in reason and "pruned policy set" in reason
    space = extract_pomdp_from_file(DEEP, on_error="raise")
    receipt = POMDPRenderProcessor(tmp_path / "pipeline")._process_single_framework(
        space, backend
    )
    assert receipt["status"] == "unsupported" and receipt["unsupported"]
    assert receipt["message"].startswith("unsupported-planning:")
    assert not receipt["output_files"]
    assert not list(tmp_path.rglob("*.py")) and not list(tmp_path.rglob("*.jl"))
    assert spec == before and DEEP.read_bytes() == original


@pytest.mark.parametrize("module,function", DIRECT)
def test_direct_renderers_preserve_existing_destination(module, function, tmp_path):
    renderer = getattr(import_module("gnn.render." + module), function)
    destination = tmp_path / "existing.py"
    destination.write_bytes(b"preserve existing artifact exactly\n")
    before = destination.read_bytes()
    ok, reason, files = renderer(parsed_deep(), destination)
    assert not ok and not files and "unsupported-planning:" in reason
    assert destination.read_bytes() == before


@pytest.mark.parametrize("mode", ("pipeline", "standalone"))
def test_file_backed_pymdp_class_refuses_before_writing(mode, tmp_path):
    from gnn.render.pymdp.pymdp_renderer import PyMDPRenderer

    destination = tmp_path / "absent" / "model.py"
    ok, reason = PyMDPRenderer({"mode": mode}).render_file(DEEP, destination)
    assert not ok and "unsupported-planning:" in reason
    assert not destination.parent.exists()


@pytest.mark.parametrize("key", ("planning_horizon", "policy_len", "policy_horizon"))
@pytest.mark.parametrize("value", (True, 0, -1, 1.0, "1", None, 2))
def test_invalid_or_unimplemented_horizons_never_coerce(key, value):
    reason = unsupported_planning({"model_parameters": {key: value}}, "pymdp")
    assert reason and "unsupported-planning:" in reason and key in reason


@pytest.mark.parametrize(
    "container",
    (
        "spec",
        "model_parameters",
        "initialparameterization",
        "initial_parameterization",
        "parameters",
    ),
)
def test_every_parameter_surface_preserves_horizon(container):
    parameters = {"planning_horizon": 5}
    spec = parameters if container == "spec" else {container: parameters}
    if container == "parameters":
        spec = {container: [{"name": "planning_horizon", "value": 5}]}
    assert "=5" in unsupported_planning(spec, "pytorch")


def test_precedence_cannot_erase_conflicting_authored_horizon():
    spec = {
        "model_parameters": {"planning_horizon": 1},
        "initialparameterization": {"policy_len": 2},
    }
    assert "conflicting" in unsupported_planning(spec, "numpyro")


@pytest.mark.parametrize(
    "parameters",
    (
        {"num_policies": 64, "num_actions": 4},
        {"num_policies": True, "num_actions": 4},
        {"num_policies": 4},
        {"num_policies": 4, "num_actions": "4"},
        {"num_policies": 4, "num_actions": False},
        {"policies": [[0], [1]]},
        {"policy_sequences": [[0], [1]]},
    ),
)
def test_policy_identity_cannot_be_invented_or_ignored(parameters):
    assert unsupported_planning({"model_parameters": parameters}, "rxinfer")


def test_conflicting_action_counts_cannot_certify_policy_count():
    spec = {
        "model_parameters": {"num_policies": 2, "num_actions": 2},
        "initialparameterization": {"num_actions": 3},
    }
    assert "unambiguous" in unsupported_planning(spec, "pymdp")


@pytest.mark.parametrize("backend", PLANNERS)
def test_matching_one_step_declarations_are_admitted(backend):
    assert (
        unsupported_contract(
            {
                "model_parameters": {
                    "planning_horizon": 1,
                    "policy_len": 1,
                    "policy_horizon": 1,
                    "num_policies": 4,
                    "num_actions": 4,
                }
            },
            backend,
        )
        is None
    )


@pytest.mark.parametrize("backend", ("stan", "bnlearn", "discopy", "thrml", "cpomdp"))
def test_separate_inference_questions_are_not_relabelled_as_planning(backend):
    assert unsupported_planning(parsed_deep(), backend) is None


def test_separately_versioned_execution_contract_keeps_its_horizon():
    spec = {
        "model_parameters": {
            "execution_contract": "episodic_contingent_v1",
            "policy_horizon": 2,
        }
    }
    assert unsupported_contract(spec, "jax") is None
    assert "unsupported-execution-contract:" in unsupported_contract(spec, "pymdp")


@pytest.mark.parametrize("target", ("jax_pomdp", "jax_combined"))
def test_alternative_jax_defaults_cannot_certify_even_a_declared_one_step(target):
    reason = unsupported_planning({"model_parameters": {"planning_horizon": 1}}, target)
    assert "does not consume source-authored planning declarations" in reason
