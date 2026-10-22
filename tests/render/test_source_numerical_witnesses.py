"""E4 foundation: independent source reading, native comparisons and counterexamples."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.jax.jax_renderer import render_gnn_to_jax
from gnn.render.pomdp_processor import pomdp_to_gnn_spec
from tests.helpers.numerical_witness import (
    FINITE_AXES,
    GAUSSIAN_AXES,
    SCHEMA,
    compare_records,
    condition_joint_gaussian,
    enumerate_finite,
    read_source,
    record,
)


def authored_source(model="finite", question="filtering"):
    """Author asymmetric source bytes once; references reread them independently."""
    if model == "finite":
        parameters = {
            "A": [[0.8, 0.1], [0.15, 0.3], [0.05, 0.6]],
            "B": [[[0.9, 0.4], [0.2, 0.7]], [[0.1, 0.6], [0.8, 0.3]]],
            "C": [0, 0, 0],
            "D": [0.7, 0.3],
        }
        section, variables = (
            "ActInfPOMDP",
            "s[2,1,type=int]\no[3,1,type=int]\nu[2,1,type=int]",
        )
        equations = (
            "p(s_0)=D; p(o_t|s_t)=A[o_t,s_t]; p(s_(t+1)|s_t,a_t)=B[s_(t+1),s_t,a_t]"
        )
        axes, states, observations = (
            FINITE_AXES,
            ["left", "right"],
            ["red", "green", "blue"],
        )
        options = {
            "num_timesteps": 4,
            "random_seed": 19,
            "b_tensor_order": "next_state_previous_state_action",
        }
    else:
        parameters = {
            "F": [[0.8, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.7]],
            "H": [[1.0, 0.2, -0.1], [0.0, 0.3, 1.0]],
            "Q": [[0.1, 0.01, 0.0], [0.01, 0.2, 0.02], [0.0, 0.02, 0.15]],
            "R": [[0.2, 0.03], [0.03, 0.3]],
            "prior_mean": [-1.0, 0.2, 0.8],
            "prior_cov": [[0.8, 0.1, -0.1], [0.1, 0.6, 0.05], [-0.1, 0.05, 0.9]],
        }
        section, variables = "ActInfContinuous", "x[3,1,type=float]\ny[2,1,type=float]"
        equations = (
            "x_0~N(prior_mean,prior_cov); x_(t+1)=F*x_t+N(0,Q); y_t=H*x_t+N(0,R)"
        )
        axes, states, observations = (
            GAUSSIAN_AXES,
            ["position", "velocity", "bias"],
            ["sensor_a", "sensor_b"],
        )
        options = {"num_timesteps": 4, "random_seed": 19}
    declaration = {
        "schema": SCHEMA,
        "model": model,
        "question": question,
        "dtype": "float64",
        "initial_time": "first_observation_uses_prior",
        "axes": axes,
        "state_ids": states,
        "observation_ids": observations,
    }
    assignments = lambda mapping: "\n".join(
        f"{key} = {json.dumps(value)}" for key, value in mapping.items()
    )
    option_lines = "\n".join(f"{key}: {value}" for key, value in options.items())
    return (
        f"# Independent {model} witness\n\n## ModelName\nE4 asymmetric {model}\n\n## GNNSection\n{section}\n\n## StateSpaceBlock\n{variables}\n\n## InitialParameterization\n{assignments(parameters)}\n\n## ModelParameters\n{option_lines}\n\n## Equations\n{equations}\n\n## NumericalWitness\n{json.dumps(declaration)}\n"
    ).encode()


def extracted_spec(tmp_path, source):
    path = tmp_path / "source.md"
    path.write_bytes(source)
    extracted = extract_pomdp_from_file(path, strict_validation=True, on_error="raise")
    assert extracted is not None
    spec = pomdp_to_gnn_spec(extracted)
    contract = read_source(source)
    for name, reference in contract["parameters"].items():
        np.testing.assert_allclose(
            spec["initialparameterization"][name],
            reference,
            rtol=0,
            atol=0,
            err_msg=f"independent extraction: {name}",
        )
    assert (
        spec["model_parameters"]["num_timesteps"]
        == contract["model_parameters"]["num_timesteps"]
    )
    assert (
        spec["model_parameters"]["random_seed"]
        == contract["model_parameters"]["random_seed"]
    )
    return spec


def save_native_receipt(tmp_path, contract, expected, actual, script, result):
    comparisons = compare_records(expected, actual)
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        text=True,
    ).splitlines()
    assert (
        hashlib.sha256((tmp_path / "source.md").read_bytes()).hexdigest()
        == contract["source_sha256"]
    )
    assert result.get("jax_version", result.get("bnlearn_version"))
    source_bytes = (tmp_path / "source.md").read_bytes()
    script_bytes = script.read_bytes()
    result_bytes = (tmp_path / "result/simulation_results.json").read_bytes()
    assert json.loads(result_bytes) == result
    receipt = {
        "checkout_commit": revision,
        "checkout_dirty_paths": dirty,
        "schema": SCHEMA,
        "source_contract": contract,
        "source_utf8": source_bytes.decode("utf-8"),
        "expected": expected,
        "actual": actual,
        "comparisons": comparisons,
        "reference_algorithm": "finite_complete_path_enumeration"
        if contract["model"] == "finite"
        else "joint_gaussian_prefix_conditioning",
        "native": {
            "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
            "script_utf8": script_bytes.decode("utf-8"),
            "python": sys.version,
            "numpy": np.__version__,
            "framework": result["framework"],
            "framework_version": result.get(
                "jax_version", result.get("bnlearn_version")
            ),
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "result_utf8": result_bytes.decode("utf-8"),
        },
        "boundary": "One explicit source subset and finite fixture; no universal equivalence or new formal/native Lean proof.",
    }
    path = tmp_path / "numerical-witness.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    destination = os.environ.get("GNN_NUMERICAL_WITNESS_DIR")
    if destination:
        out = Path(destination)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"e4-{contract['model']}-filtering.json").write_bytes(path.read_bytes())


def test_finite_source_filtering_and_smoothing_have_distinct_conditioning(tmp_path):
    source = authored_source()
    contract = read_source(source)
    extracted_spec(tmp_path, source)
    observations, actions = [2, 0, 1, 2], [1, 0, 1]
    filtering = enumerate_finite(contract, observations, actions)
    smoothed = read_source(authored_source(question="smoothing"))
    smoothing = enumerate_finite(smoothed, observations, actions)
    # The first posterior is also checked by direct Bayes odds, independent of
    # path enumeration. The final filter and smoother condition on the same data.
    np.testing.assert_allclose(filtering[0], [7 / 43, 36 / 43], rtol=1e-14)
    np.testing.assert_allclose(filtering[-1], smoothing[-1], rtol=1e-14)
    assert not np.allclose(filtering[:-1], smoothing[:-1])
    filter_record = record(contract, observations, actions, {"beliefs": filtering})
    smooth_record = record(smoothed, observations, actions, {"beliefs": smoothing})
    assert filter_record["conditioning_horizon"] == [1, 2, 3, 4]
    assert smooth_record["conditioning_horizon"] == [4, 4, 4, 4]
    with pytest.raises(ValueError, match="source_sha256|question"):
        compare_records(filter_record, smooth_record)


@pytest.mark.needs_bnlearn
def test_native_finite_filter_matches_independent_source_enumeration(tmp_path):
    from gnn.render.bnlearn import generate_bnlearn_code

    source = authored_source()
    contract = read_source(source)
    spec = extracted_spec(tmp_path, source)
    script = tmp_path / "model.py"
    generate_bnlearn_code(spec, script)
    observations, actions = [2, 0, 1, 2], [1, 0, 1]
    for name, value in (("observations", observations), ("actions", actions)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    child = subprocess.run(
        [
            sys.executable,
            str(script),
            "--observations-json",
            str(tmp_path / "observations.json"),
            "--transition-actions-json",
            str(tmp_path / "actions.json"),
        ],
        env=dict(os.environ, BNLEARN_OUTPUT_DIR=str(tmp_path / "result")),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "result/simulation_results.json").read_text())
    expected = record(
        contract,
        observations,
        actions,
        {"beliefs": enumerate_finite(contract, observations, actions)},
    )
    actual = record(
        contract,
        result["observations"],
        result["transition_actions"],
        {"beliefs": result["beliefs"]},
    )
    save_native_receipt(tmp_path, contract, expected, actual, script, result)


def test_native_gaussian_filter_matches_independent_source_joint_conditioning(tmp_path):
    source = authored_source("gaussian")
    contract = read_source(source)
    spec = extracted_spec(tmp_path, source)
    script = tmp_path / "model_jax.py"
    success, reason, artifacts = render_gnn_to_jax(spec, script)
    assert success and str(script) in artifacts, reason
    # Run the real generated entrypoint while inspecting its actual JAX arrays
    # in the same process. No generated source, observations or outputs change.
    runtime_path = tmp_path / "native-runtime.json"
    runner = "import json,runpy,sys; ns=runpy.run_path(sys.argv[1]); runtime={'array_dtype': str(ns['arr']([0.0]).dtype), 'jax_enable_x64': bool(ns['jax'].config.jax_enable_x64)}; assert runtime['array_dtype']=='float64' and runtime['jax_enable_x64']; result=ns['run_simulation'](); assert result['validation']['all_valid']; open(sys.argv[2], 'x').write(json.dumps(runtime))"
    before = hashlib.sha256(script.read_bytes()).hexdigest()
    child = subprocess.run(
        [sys.executable, "-c", runner, str(script), str(runtime_path)],
        env=dict(os.environ, GNN_OUTPUT_DIR=str(tmp_path / "result")),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert hashlib.sha256(script.read_bytes()).hexdigest() == before
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "result/simulation_results.json").read_text())
    runtime = json.loads(runtime_path.read_text())
    assert runtime == {"array_dtype": "float64", "jax_enable_x64": True}
    observations = result["observations_continuous"]
    means, covariances = condition_joint_gaussian(contract, observations)
    expected = record(
        contract, observations, [], {"beliefs": means, "posterior_cov": covariances}
    )
    actual = record(
        contract,
        observations,
        [],
        {"beliefs": result["beliefs"], "posterior_cov": result["posterior_cov"]},
    )
    assert result["control_mode"] == "passive"
    assert result["validation"]["all_valid"] is True
    assert np.linalg.cond(np.asarray(contract["parameters"]["prior_cov"])) < 10
    actual["runtime_precision"] = runtime
    save_native_receipt(tmp_path, contract, expected, actual, script, result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future"),
        ("dtype", "float32"),
        ("initial_time", "predict_before_first_observation"),
        ("axes", {"A": ["state", "observation"], "B": FINITE_AXES["B"]}),
        ("question", "efe_with_reversed_sign"),
        ("state_ids", ["same", "same"]),
    ],
)
def test_reference_refuses_unsupported_or_ambiguous_source(field, value):
    text = authored_source().decode()
    prefix, declaration = text.split("## NumericalWitness\n")
    metadata = json.loads(declaration)
    metadata[field] = value
    with pytest.raises(
        ValueError, match="schema|dtype|initial_time|axes|question|state_ids"
    ):
        read_source((prefix + "## NumericalWitness\n" + json.dumps(metadata)).encode())


@pytest.mark.parametrize(
    "mutation", ["transition", "action_time", "horizon", "source", "precision", "axes"]
)
def test_finite_counterexamples_name_the_incompatible_quantity(mutation):
    contract = read_source(authored_source())
    observations, actions = [2, 0, 1, 2], [1, 0, 1]
    values = enumerate_finite(contract, observations, actions)
    expected = record(contract, observations, actions, {"beliefs": values})
    wrong = deepcopy(expected)
    if mutation in ("transition", "action_time"):
        changed = deepcopy(contract)
        if mutation == "transition":
            changed["parameters"]["B"][0][0][1] += 0.15
            changed["parameters"]["B"][1][0][1] -= 0.15
            wrong_values = enumerate_finite(changed, observations, actions)
        else:
            wrong_values = enumerate_finite(
                changed, observations, actions[1:] + actions[:1]
            )
        wrong["quantities"]["beliefs"] = wrong_values.tolist()
        with pytest.raises(AssertionError, match="beliefs"):
            compare_records(expected, wrong)
    else:
        field, value = {
            "horizon": ("conditioning_horizon", [4, 4, 4, 4]),
            "source": ("source_sha256", "0" * 64),
            "precision": ("dtype", "float32"),
            "axes": ("axes", {"A": ["state", "observation"]}),
        }[mutation]
        wrong[field] = value
        with pytest.raises(ValueError, match=field):
            compare_records(expected, wrong)


@pytest.mark.parametrize(
    "mutation",
    ["extra_prediction", "cross_covariance", "covariance_only", "future_conditioning"],
)
def test_gaussian_counterexamples_detect_timing_covariance_and_horizon(mutation):
    contract = read_source(authored_source("gaussian"))
    observations = [[0.2, -0.3], [0.8, 0.1], [-0.2, 0.5], [0.4, -0.7]]
    means, covariances = condition_joint_gaussian(contract, observations)
    expected = record(
        contract, observations, [], {"beliefs": means, "posterior_cov": covariances}
    )
    changed = deepcopy(contract)
    p = changed["parameters"]
    if mutation == "covariance_only":
        wrong = deepcopy(expected)
        wrong["quantities"]["posterior_cov"] = [
            np.diag(np.diag(covariance)).tolist() for covariance in covariances
        ]
        with pytest.raises(AssertionError, match="posterior_cov"):
            compare_records(expected, wrong)
        return
    if mutation == "extra_prediction":
        f = np.asarray(p["F"])
        p["prior_mean"] = (f @ np.asarray(p["prior_mean"])).tolist()
        p["prior_cov"] = (
            f @ np.asarray(p["prior_cov"]) @ f.T + np.asarray(p["Q"])
        ).tolist()
    elif mutation == "cross_covariance":
        for key in ("prior_cov", "Q", "R"):
            p[key] = np.diag(np.diag(p[key])).tolist()
    else:
        # Changing a future observation cannot alter an earlier filter result.
        alternate = deepcopy(observations)
        alternate[-1] = [20.0, -30.0]
        other_means, other_covariances = condition_joint_gaussian(contract, alternate)
        np.testing.assert_allclose(other_means[:-1], means[:-1], rtol=0, atol=0)
        np.testing.assert_allclose(other_covariances, covariances, rtol=0, atol=0)
        wrong = deepcopy(expected)
        wrong["conditioning_horizon"] = [4, 4, 4, 4]
        with pytest.raises(ValueError, match="conditioning_horizon"):
            compare_records(expected, wrong)
        return
    wrong_means, wrong_covariances = condition_joint_gaussian(changed, observations)
    wrong = record(
        contract,
        observations,
        [],
        {"beliefs": wrong_means, "posterior_cov": wrong_covariances},
    )
    with pytest.raises(AssertionError, match="beliefs|posterior_cov"):
        compare_records(expected, wrong)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "nonfinite", "boolean", "equation", "unsupported", "workload"],
)
def test_independent_reader_rejects_ambiguous_or_unbounded_parameter_sources(mutation):
    source = authored_source()
    if mutation == "duplicate":
        source = source.replace(b"D = [0.7, 0.3]", b"D = [0.7, 0.3]\nD = [0.7, 0.3]")
    elif mutation == "nonfinite":
        source = source.replace(b"D = [0.7, 0.3]", b"D = [NaN, 0.3]")
    elif mutation == "boolean":
        source = source.replace(b"D = [0.7, 0.3]", b"D = [true, false]")
    elif mutation == "equation":
        source = source.replace(b"A[o_t,s_t]", b"A[s_t,o_t]")
    elif mutation == "unsupported":
        source = source.replace(b"D = [0.7, 0.3]", b"D = {(0.7, 0.3)}")
    else:
        source = source.replace(b"num_timesteps: 4", b"num_timesteps: 100000")
    with pytest.raises(ValueError):
        read_source(source)


@pytest.mark.parametrize("model,name", [("finite", "A"), ("gaussian", "H")])
def test_reference_rejects_transposed_rectangular_source_axes(model, name):
    source = authored_source(model)
    contract = read_source(source)
    matrix = contract["parameters"][name]
    altered = source.replace(
        f"{name} = {json.dumps(matrix)}".encode(),
        f"{name} = {json.dumps(np.asarray(matrix).T.tolist())}".encode(),
    )
    assert altered != source
    with pytest.raises(ValueError, match="dimensions"):
        read_source(altered)


def test_source_identity_preserves_line_endings_and_final_byte():
    source = authored_source()
    original = read_source(source)
    observations, actions = [2, 0, 1, 2], [1, 0, 1]
    values = enumerate_finite(original, observations, actions)
    expected = record(original, observations, actions, {"beliefs": values})
    for altered in (source.replace(b"\n", b"\r\n"), source.rstrip(b"\n")):
        changed = read_source(altered)
        assert changed["parameters"] == original["parameters"]
        assert changed["source_sha256"] != original["source_sha256"]
        with pytest.raises(ValueError, match="source_sha256"):
            compare_records(
                expected, record(changed, observations, actions, {"beliefs": values})
            )


@pytest.mark.parametrize("missing", ["all", "posterior_cov"])
def test_unavailable_quantities_cannot_be_reported_as_agreement(missing):
    contract = read_source(authored_source("gaussian"))
    observations = [[0.2, -0.3], [0.8, 0.1], [-0.2, 0.5], [0.4, -0.7]]
    means, covariances = condition_joint_gaussian(contract, observations)
    expected = record(
        contract, observations, [], {"beliefs": means, "posterior_cov": covariances}
    )
    actual = deepcopy(expected)
    if missing == "all":
        expected["quantities"] = actual["quantities"] = {}
    else:
        del actual["quantities"][missing]
    with pytest.raises(ValueError, match="quantities"):
        compare_records(expected, actual)
