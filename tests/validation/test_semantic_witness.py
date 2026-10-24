"""Raw-artifact replay, native metadata admission and installed CLI contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from gnn.cli import main
from gnn.validation.semantic_witness import (
    build_semantic_witness,
    verify_semantic_witness,
)
from gnn.validation.witness_source import read_source as production_source
from tests.helpers.numerical_witness import (
    condition_joint_gaussian,
    enumerate_finite,
    read_source,
)
from tests.render.test_source_numerical_witnesses import authored_source


def reported_result(model: str) -> tuple[bytes, dict]:
    """Build identified reported values with a separate test-only numerical oracle."""
    source = authored_source(model)
    contract = read_source(source)
    finite = model == "finite"
    observations = (
        [2, 0, 1, 2] if finite else [[0.1, -0.2], [0.4, 0.2], [-0.1, 0.5], [0.3, 0.7]]
    )
    result = {
        "framework": "bnlearn" if finite else "jax",
        "dtype": "float64",
        "initial_time": contract["initial_time"],
        "axes": contract["axes"],
        "conditioning_horizon": [1, 2, 3, 4],
        "num_timesteps": 4,
        "num_states": len(contract["state_ids"]),
        "num_observations": len(contract["observation_ids"]),
        "random_seed": 19,
        "parameters": contract["parameters"],
        "inference_contract": "bnlearn_fixed_action_filter_v1"
        if finite
        else "gnn_lgssm_filter_v1",
        "inference_mode": "exact_categorical_filtering"
        if finite
        else "gaussian_filtering",
        "control_mode": "fixed_transition_actions" if finite else "passive",
    }
    if finite:
        result.update(
            observations=observations,
            transition_actions=[1, 0, 1],
            action_semantics="transition_into_next_state",
        )
        result["beliefs"] = enumerate_finite(
            contract, observations, result["transition_actions"]
        ).tolist()
    else:
        means, covariance = condition_joint_gaussian(contract, observations)
        result.update(
            observations_continuous=observations,
            beliefs=means.tolist(),
            posterior_cov=covariance.tolist(),
            controls=np.zeros((4, 3)).tolist(),
        )
    return source, result


@pytest.mark.parametrize("model", ["finite", "gaussian"])
def test_raw_artifact_reference_matches_separate_oracle_and_replays(model):
    source, result = reported_result(model)
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "passed"
    assert verify_semantic_witness(receipt) == []
    assert receipt["execution_evidence"]["execution_source_verified"] is False
    assert receipt["formal_evidence"] == {
        "status": "unavailable",
        "universal_equivalence": False,
    }
    assert "source_sha256" not in receipt["candidate"]
    assert "state_ids" not in receipt["candidate"]
    assert receipt["comparisons"]["beliefs"]["max_abs_error"] < 1e-14


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("initial_time", "predict_first"),
        ("conditioning_horizon", [4, 4, 4, 4]),
        ("random_seed", True),
        ("num_states", 2.0),
        ("axes", {"A": ["state", "observation"]}),
        ("inference_contract", "another_filter"),
        ("action_semantics", "arrival_action"),
    ],
)
def test_native_metadata_must_not_be_stamped_from_expected(field, value):
    source, result = reported_result("finite")
    result[field] = value
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "refused"
    assert receipt["comparisons"][f"identity:{field}"]["status"] == "refused"
    assert receipt["comparisons"]["beliefs"]["status"] == "refused"
    assert verify_semantic_witness(receipt) == []  # replay-valid refusal


def test_missing_native_metadata_and_quantity_remain_unavailable():
    source, result = reported_result("gaussian")
    del result["dtype"], result["posterior_cov"]
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "unavailable"
    assert receipt["comparisons"]["identity:dtype"]["status"] == "unavailable"
    assert receipt["candidate"]["dtype"] is None
    assert receipt["comparisons"]["posterior_cov"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "mutation", ["covariance", "transition", "action_time", "boolean", "shape"]
)
def test_quantity_and_parameter_counterexamples_are_specific(mutation):
    source, result = reported_result(
        "gaussian" if mutation == "covariance" else "finite"
    )
    target = "beliefs"
    if mutation == "covariance":
        result["posterior_cov"][1][0][1] += 0.02
        target = "posterior_cov"
    elif mutation == "transition":
        result["parameters"]["B"][0][0][0] += 0.01
        target = "parameter:B"
    elif mutation == "action_time":
        result["transition_actions"] = [0, 1, 0]
    elif mutation == "boolean":
        result["beliefs"][0][0] = True
    else:
        result["beliefs"] = result["beliefs"][:-1]
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] in ("failed", "refused")
    assert receipt["comparisons"][target]["status"] in ("failed", "refused")


def test_coherent_saved_quantity_and_result_tamper_cannot_borrow_success():
    source, result = reported_result("gaussian")
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    changed = deepcopy(receipt)
    result["beliefs"][0][0] += 0.4
    raw = json.dumps(result).encode()
    changed["artifacts"]["result"] = {
        "utf8": raw.decode(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    changed["candidate"]["quantities"]["beliefs"] = result["beliefs"]
    changed["reference"]["quantities"]["beliefs"] = result["beliefs"]
    assert verify_semantic_witness(changed)
    assert build_semantic_witness(source, raw)["status"] == "failed"
    for field in ("formal_evidence", "execution_evidence", "status"):
        altered = deepcopy(receipt)
        altered[field] = {"status": "verified"}
        assert verify_semantic_witness(altered), field


def test_program_digest_binding_does_not_authenticate_execution():
    source, result = reported_result("finite")
    program = b"print('identified program')\n"
    result["generated_script_sha256"] = hashlib.sha256(program).hexdigest()
    receipt = build_semantic_witness(source, json.dumps(result).encode(), program)
    assert receipt["status"] == "passed"
    assert receipt["execution_evidence"]["status"] == "artifact_only"
    receipt["artifacts"]["program"]["utf8"] += "# different bytes\n"
    assert verify_semantic_witness(receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "0" * 64),
        ("source_identity", {"source_sha256": "0" * 64}),
        ("source_identity", []),
        ("state_ids", ["right", "left"]),
        ("observation_ids", ["blue", "green", "red"]),
        ("model_kind", "continuous"),
        ("schema_version", "another_result_schema"),
        ("framework", "unrelated_backend"),
        ("validation", {"all_valid": False}),
        ("validation", {"all_valid": 1}),
        ("validation", {"all_valid": True, "nested": [{"posterior_finite": False}]}),
        ("success", False),
        ("status", "failed"),
        ("status", "running"),
    ],
)
def test_explicit_native_contradictions_cannot_borrow_numeric_agreement(field, value):
    source, result = reported_result("finite")
    result[field] = value
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "refused"
    assert any(row["status"] == "refused" for row in receipt["comparisons"].values())
    assert receipt["comparisons"]["beliefs"]["status"] == "refused"
    assert verify_semantic_witness(receipt) == []


@pytest.mark.parametrize("model", ["finite", "gaussian"])
def test_optional_native_identities_must_agree_when_present(model):
    source, result = reported_result(model)
    contract = read_source(source)
    result.update(
        source_sha256=contract["source_sha256"],
        source_identity={"source_sha256": contract["source_sha256"]},
        state_ids=contract["state_ids"],
        observation_ids=contract["observation_ids"],
        model_kind="discrete" if model == "finite" else "continuous",
        schema_version=result["inference_contract"],
        success=True,
        status="completed",
        validation={"all_valid": True, "posterior_finite": True},
    )
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "passed"
    assert verify_semantic_witness(receipt) == []
    del result["framework"]
    unavailable = build_semantic_witness(source, json.dumps(result).encode())
    assert unavailable["status"] == "unavailable"
    assert unavailable["comparisons"]["identity:framework"]["status"] == "unavailable"


@pytest.mark.parametrize("shape", ["wide", "deep", "long_path"])
def test_native_validation_traversal_refuses_before_unbounded_path_expansion(shape):
    source, result = reported_result("finite")
    if shape == "wide":
        validation = {"checks": [True] * 4097}
    elif shape == "deep":
        validation = {"all_valid": True}
        for _ in range(34):
            validation = {"nested": validation}
    else:
        validation = {"x" * 2049: True}
    result["validation"] = validation
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "refused"
    assert receipt["comparisons"]["identity:validation_budget"]["status"] == "refused"
    assert len(receipt["comparisons"]) < 50
    assert verify_semantic_witness(receipt) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "negative_mass",
        "above_one",
        "zero_variance",
        "negative_variance",
        "indefinite",
        "asymmetric",
    ],
)
def test_invalid_native_distributions_cannot_pass_absolute_tolerance(mutation):
    finite = mutation in {"negative_mass", "above_one"}
    source, result = reported_result("finite" if finite else "gaussian")
    contract = read_source(source)
    if finite:
        matrix = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        source = source.replace(
            ("A = " + json.dumps(contract["parameters"]["A"])).encode(),
            ("A = " + json.dumps(matrix)).encode(),
        )
        result["parameters"]["A"] = matrix
        result["observations"] = [0, 0, 0, 0]
        result["beliefs"] = [[1.0, 0.0] for _ in range(4)]
        assert (
            build_semantic_witness(source, json.dumps(result).encode())["status"]
            == "passed"
        )
        result["beliefs"][0] = [
            1 + 1e-13,
            -1e-13 if mutation == "negative_mass" else 0.0,
        ]
        validity = "validity:beliefs"
    else:
        identity = np.eye(2)
        parameters = {
            "F": (0.5 * identity).tolist(),
            "H": identity.tolist(),
            "Q": (0.75 * identity).tolist(),
            "R": (1e-30 * identity).tolist(),
            "prior_mean": [1.0, 1.0],
            "prior_cov": identity.tolist(),
        }
        for name, value in parameters.items():
            source = source.replace(
                (name + " = " + json.dumps(contract["parameters"][name])).encode(),
                (name + " = " + json.dumps(value)).encode(),
            )
        source = source.replace(b"x[3,1,type=float]", b"x[2,1,type=float]")
        prefix, raw = source.split(b"## NumericalWitness\n")
        declaration = json.loads(raw)
        declaration["state_ids"] = ["x", "z"]
        source = (
            prefix + b"## NumericalWitness\n" + json.dumps(declaration).encode() + b"\n"
        )
        observations = [[0.0, 0.0] for _ in range(4)]
        means, _ = condition_joint_gaussian(read_source(source), observations)
        result.update(
            parameters=parameters,
            observations_continuous=observations,
            beliefs=means.tolist(),
            posterior_cov=[(1e-30 * identity).tolist() for _ in range(4)],
            num_states=2,
            controls=[[0.0, 0.0] for _ in range(4)],
        )
        assert (
            build_semantic_witness(source, json.dumps(result).encode())["status"]
            == "passed"
        )
        covariance = result["posterior_cov"][0]
        if mutation in {"zero_variance", "negative_variance"}:
            covariance[0][0] = 0.0 if mutation == "zero_variance" else -1e-13
        else:
            covariance[0][1] = 1e-13
            covariance[1][0] = 1e-13 if mutation == "indefinite" else 0.0
        validity = "validity:posterior_cov"
    receipt = build_semantic_witness(source, json.dumps(result).encode())
    assert receipt["status"] == "refused"
    assert receipt["comparisons"][validity]["status"] == "refused"
    assert verify_semantic_witness(receipt) == []


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        [],
        {"schema": "gnn-semantic-witness-v2", "artifacts": []},
        {
            "schema": "gnn-semantic-witness-v2",
            "artifacts": {
                "source": {"utf8": 3, "sha256": "0" * 64},
                "result": {},
                "program": None,
            },
        },
    ],
)
def test_malformed_receipt_structures_return_errors_without_attribute_crashes(receipt):
    assert verify_semantic_witness(receipt)


@pytest.mark.parametrize(
    "failure",
    [
        "transition_overflow",
        "innovation_overflow",
        "observation_overflow",
        "unsolvable",
    ],
)
def test_gaussian_reference_numerical_failures_refuse_without_nonfinite_receipts(
    tmp_path, capsys, monkeypatch, failure
):
    source, result = reported_result("gaussian")
    contract = read_source(source)
    if failure == "transition_overflow":
        parameters = {"F": (1e308 * np.eye(3)).tolist()}
    elif failure == "innovation_overflow":
        parameters = {
            "F": np.zeros((3, 3)).tolist(),
            "H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "Q": (1e308 * np.eye(3)).tolist(),
            "R": (1e308 * np.eye(2)).tolist(),
            "prior_mean": [1.0, 1.0, 1.0],
            "prior_cov": (1e308 * np.eye(3)).tolist(),
        }
        source = source.replace(b"num_timesteps: 4", b"num_timesteps: 1")
        result.update(
            num_timesteps=1,
            conditioning_horizon=[1],
            observations_continuous=[[0.0, 0.0]],
            controls=[[0.0, 0.0, 0.0]],
        )
    elif failure == "observation_overflow":
        parameters = {
            "F": np.zeros((3, 3)).tolist(),
            "H": [[1e308, 0.0, 0.0], [0.0, 1e308, 0.0]],
            "Q": np.eye(3).tolist(),
            "R": np.eye(2).tolist(),
            "prior_mean": [1.0, 1.0, 1.0],
            "prior_cov": np.eye(3).tolist(),
        }
        source = source.replace(b"num_timesteps: 4", b"num_timesteps: 1")
        result.update(
            num_timesteps=1,
            conditioning_horizon=[1],
            observations_continuous=[[0.0, 0.0]],
            controls=[[0.0, 0.0, 0.0]],
        )
    else:
        parameters = {}

        def unsolvable(*args, **kwargs):
            raise np.linalg.LinAlgError("injected unsolvable conditioning system")

        monkeypatch.setattr(np.linalg, "solve", unsolvable)
    for name, value in parameters.items():
        source = source.replace(
            (name + " = " + json.dumps(contract["parameters"][name])).encode(),
            (name + " = " + json.dumps(value)).encode(),
        )
        result["parameters"][name] = value
    raw = json.dumps(result).encode()
    with pytest.raises(ValueError, match="Gaussian reference numerical failure"):
        build_semantic_witness(source, raw)
    model, native, output = (
        tmp_path / name for name in ("model.md", "result.json", "receipt.json")
    )
    model.write_bytes(source)
    native.write_bytes(raw)
    assert (
        main(
            [
                "witness",
                str(model),
                "--results",
                str(native),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 1
    )
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["status"] == "error" and envelope["data"] == {}
    assert "Gaussian reference numerical failure" in envelope["error"]
    assert not output.exists()


@pytest.mark.parametrize(
    "heading",
    [
        "## Unsupported",
        "## Unsupported2",
        "## Unsupported-Contract",
        "##NumericalWitness",
        " ## ModelParameters",
        "##",
    ],
)
@pytest.mark.parametrize("reader", [read_source, production_source])
def test_restricted_readers_refuse_unknown_or_malformed_headings(reader, heading):
    source = authored_source()
    for altered in (
        (heading + "\nignored content\n").encode() + source,
        source.replace(
            b"E4 asymmetric finite",
            ("E4 asymmetric finite\n" + heading + "\nignored content").encode(),
        ),
    ):
        with pytest.raises(ValueError, match="heading|source section"):
            reader(altered)
    with pytest.raises(ValueError, match="byte budget"):
        reader(b"#" + b"x" * (1024 * 1024) + b"\n" + source)


def test_native_smoothing_and_unsupported_source_do_not_borrow_filtering():
    _, result = reported_result("finite")
    receipt = build_semantic_witness(
        authored_source(question="smoothing"), json.dumps(result).encode()
    )
    assert receipt["status"] == "refused"
    assert receipt["comparisons"]["identity:question"]["status"] == "refused"
    with pytest.raises(ValueError, match="source section"):
        build_semantic_witness(b"# Unspecified model\n", b"{}")


def test_cli_writes_receipt_and_verifies_without_rewriting(tmp_path, capsys):
    source, result = reported_result("finite")
    model, native, output = (
        tmp_path / name for name in ("model.md", "result.json", "receipt.json")
    )
    model.write_bytes(source)
    native.write_text(json.dumps(result))
    assert (
        main(
            [
                "witness",
                str(model),
                "--results",
                str(native),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["status"] == "passed"
    before = output.read_bytes()
    assert main(["witness", "--verify", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert output.read_bytes() == before


def test_cli_serialized_receipt_limit_is_shared_by_output_and_replay(
    tmp_path, capsys, monkeypatch
):
    from gnn.cli import handlers_ops

    source, result = reported_result("finite")
    raw = json.dumps(result).encode()
    receipt = build_semantic_witness(source, raw)
    size = len((json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
    model, native, output = (
        tmp_path / name for name in ("model.md", "result.json", "receipt.json")
    )
    model.write_bytes(source)
    native.write_bytes(raw)
    arguments = ["witness", str(model), "--results", str(native), "--json"]
    monkeypatch.setattr(handlers_ops, "MAX_WITNESS_RECEIPT_BYTES", size)
    assert main([*arguments, "--output", str(output)]) == 0
    assert output.stat().st_size == size
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert main(["witness", "--verify", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    before = output.read_bytes()
    monkeypatch.setattr(handlers_ops, "MAX_WITNESS_RECEIPT_BYTES", size - 1)
    for command in (
        [*arguments, "--output", str(output)],
        arguments,
        ["witness", "--verify", str(output), "--json"],
    ):
        assert main(command) == 1
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert "byte budget" in envelope["error"]
        assert envelope["data"] == {}
        assert output.read_bytes() == before
    assert main(["witness", "--verify", str(output), "--output", str(output)]) == 1
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "alias", ["source", "results", "program", "symlink", "hardlink"]
)
def test_cli_output_cannot_overwrite_any_input_alias(tmp_path, alias):
    source, result = reported_result("finite")
    model, native, program = (
        tmp_path / name for name in ("model.md", "result.json", "model.py")
    )
    model.write_bytes(source)
    native.write_text(json.dumps(result))
    program.write_bytes(b"# retained program\n")
    inputs = {"source": model, "results": native, "program": program}
    if alias in inputs:
        output = inputs[alias]
    else:
        output = tmp_path / "alias.json"
        try:
            if alias == "symlink":
                output.symlink_to(model)
            else:
                output.hardlink_to(model)
        except OSError as exc:
            pytest.fail(f"Required {alias} creation failed: {exc}")
    before = {key: path.read_bytes() for key, path in inputs.items()}
    assert (
        main(
            [
                "witness",
                str(model),
                "--results",
                str(native),
                "--program",
                str(program),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 1
    )
    assert {key: path.read_bytes() for key, path in inputs.items()} == before
    assert output.read_bytes() in before.values()
