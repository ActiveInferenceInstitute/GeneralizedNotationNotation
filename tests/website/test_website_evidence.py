"""Recorded evidence stays inspectable without becoming a frontend proof claim."""

import hashlib
import json
import re
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

from gnn.website import WebsiteGenerator
from gnn.website.evidence import PROOF_BOUNDARY, render_evidence_section


def _receipt() -> dict:
    """Presentation fixture for the declared v2 artifact-only contract."""
    return {
        "schema": "gnn-semantic-witness-v2",
        "status": "refused",
        "source_contract": {
            "model": "gaussian",
            "source_sha256": "a" * 64,
            "question": "filtering",
            "dtype": "float64",
            "axes": {"H": ["observation", "state"]},
            "state_ids": ["position", "velocity"],
            "observation_ids": ["sensor"],
            "equations": ["o_t = H s_t + noise"],
        },
        "reference": {
            "algorithm": "joint_gaussian_prefix_conditioning",
            "quantities": {"beliefs": [[1.0, 0.0]]},
        },
        "candidate": {"quantities": {"beliefs": [[1.2, 0.0]]}},
        "comparisons": {
            "identity:initial_time": {
                "status": "refused",
                "reason": "incompatible scientific identity",
                "expected": "first_observation_uses_prior",
                "actual": "extra_predict",
            },
            "beliefs": {
                "status": "refused",
                "numeric_status": "failed",
                "reason": "semantic admission did not pass",
                "max_abs_error": 0.2,
                "max_rel_error": 0.2,
                "atol": 1e-12,
                "rtol": 1e-10,
            },
            "posterior_cov": {
                "status": "unavailable",
                "reason": "required native quantity is missing",
            },
            "parameter:H": {
                "status": "passed",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": 0.0,
                "rtol": 0.0,
            },
        },
        "assumptions": {
            "initial_time": "first_observation_uses_prior",
            "conditioning_horizon": [1],
        },
        "artifacts": {
            "source": {"sha256": "a" * 64, "utf8": "<img src=x>"},
            "result": {"sha256": "b" * 64, "utf8": "{}"},
            "program": None,
        },
        "formal_evidence": {"status": "unavailable", "universal_equivalence": False},
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
            "reported_framework": "jax",
            "reported_versions": {"jax_version": "reported-version"},
        },
    }


def _trace_receipt() -> dict:
    """A complete recorded display alignment with one diagnostic mean mismatch."""
    receipt = _receipt()
    receipt["status"] = "failed"
    contract = receipt["source_contract"]
    contract.update(
        model_parameters={"num_timesteps": 2, "random_seed": 0},
        initial_time="first_observation_uses_prior",
    )
    contract["state_ids"] = ["position </script><x>", "velocity"]
    identities = {
        "dtype": "float64",
        "initial_time": contract["initial_time"],
        "axes": contract["axes"],
        "conditioning_horizon": [1, 2],
        "num_timesteps": 2,
        "num_states": 2,
        "num_observations": 1,
        "random_seed": 0,
        "inference_contract": "gnn_lgssm_filter_v1",
        "inference_mode": "gaussian_filtering",
        "control_mode": "passive",
    }
    receipt["comparisons"].update(
        {"identity:" + key: {"status": "passed"} for key in identities}
    )
    receipt["comparisons"]["beliefs"] = {
        "status": "failed",
        "reason": "outside declared tolerance",
        "max_abs_error": 0.25,
        "atol": 1e-12,
        "rtol": 1e-10,
    }
    receipt["comparisons"]["posterior_cov"] = {"status": "passed"}
    receipt["reference"].update(
        conditioning_horizon=[1, 2],
        quantities={
            "beliefs": [[1.0, 2.0], [3.0, 4.0]],
            "posterior_cov": [[[0.8, 0.1], [0.1, 0.6]], [[0.4, 0.05], [0.05, 0.3]]],
        },
    )
    receipt["candidate"] = {
        **identities,
        "quantities": {
            "beliefs": [[1.25, 2.0], [3.0, 4.0]],
            "posterior_cov": deepcopy(
                receipt["reference"]["quantities"]["posterior_cov"]
            ),
        },
    }
    receipt["assumptions"] = {
        key: contract[key]
        for key in ("question", "initial_time", "state_ids", "observation_ids", "axes")
    }
    receipt["assumptions"]["conditioning_horizon"] = [1, 2]
    return receipt


def test_trace_explorer_retains_bounded_arrays_controls_and_full_table(
    tmp_path: Path,
) -> None:
    receipt = _trace_receipt()
    original = deepcopy(receipt)
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path / "site", "analysis": [receipt]}, filesystem=False
    )
    assert result["success"] and result["pages_created"] == 7
    html = (tmp_path / "site/analysis.html").read_text()
    assert receipt == original
    for control in (
        "data-trace-quantity",
        "data-trace-state",
        "data-trace-mode",
        "data-trace-status",
    ):
        assert control in html
    assert (
        'viewBox="0 0 640 250" role="img" aria-labelledby="trace-title-0 trace-desc-0"'
        in html
    )
    assert '<title id="trace-title-0">' in html and '<desc id="trace-desc-0">' in html
    assert "Posterior diagonal variance" in html and "Gaussian posterior mean" in html
    assert "No confidence bands or verification claims are inferred." in html
    assert "Unverified recorded arrays and display-only residuals" in html
    assert "position &lt;/script&gt;&lt;x&gt;" in html
    encoded = re.search(
        r'<script type="application/json" data-trace-data>(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert encoded is not None and "</" not in encoded.group(1)
    payload = json.loads(encoded.group(1))
    assert payload["prefixes"] == [1, 2]
    assert payload["series"]["beliefs"]["reference"] == [[1.0, 2.0], [3.0, 4.0]]
    assert payload["series"]["beliefs"]["candidate"] == [[1.25, 2.0], [3.0, 4.0]]
    assert payload["series"]["beliefs"]["residual"] == [[0.25, 0.0], [0.0, 0.0]]
    assert payload["series"]["beliefs"]["status"] == "failed"
    assert payload["series"]["posterior_variance"]["reference"] == [
        [0.8, 0.6],
        [0.4, 0.3],
    ]
    wrapped = render_evidence_section(
        [
            {
                "schema": "fep-semantic-witness-audit-v1",
                "status": "failed",
                "gnn_witness": receipt,
            }
        ]
    )
    assert (
        '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
        in wrapped
    )


def test_trace_explorer_refuses_unaligned_unavailable_or_malformed_display_data() -> (
    None
):
    for mutation in (
        "identity",
        "horizon",
        "axes",
        "assumptions",
        "boolean",
        "oversize",
        "overflow",
        "unavailable",
        "status",
    ):
        receipt = _trace_receipt()
        receipt["comparisons"]["posterior_cov"]["status"] = "unavailable"
        if mutation == "identity":
            receipt["comparisons"]["identity:dtype"]["status"] = "refused"
        elif mutation == "horizon":
            receipt["candidate"]["conditioning_horizon"] = [2, 2]
        elif mutation == "axes":
            receipt["candidate"]["axes"] = {"H": ["state", "observation"]}
        elif mutation == "assumptions":
            receipt["assumptions"]["initial_time"] = "extra_predict"
        elif mutation == "boolean":
            receipt["candidate"]["quantities"]["beliefs"][0][0] = True
        elif mutation == "oversize":
            receipt["source_contract"]["model_parameters"]["num_timesteps"] = 9
        elif mutation == "overflow":
            receipt["reference"]["quantities"]["beliefs"][0][0] = -1e308
            receipt["candidate"]["quantities"]["beliefs"][0][0] = 1e308
        elif mutation == "unavailable":
            receipt["comparisons"]["beliefs"]["status"] = "unavailable"
        else:
            receipt["status"] = ["passed"]
        html = render_evidence_section([receipt])
        assert (
            '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
            not in html
        ), mutation
        assert '<svg viewBox="0 0 640 250"' not in html, mutation
    assert '<svg viewBox="0 0 640 250"' not in render_evidence_section([_receipt()])


def test_v2_shows_individual_outcomes_assumptions_and_unverified_boundary() -> None:
    receipt = _receipt()
    receipt["artifacts"]["program"] = {
        "sha256": "c" * 64,
        "utf8": "unique retained program: λ <script>source only</script>",
    }
    receipt["comparisons"]["diagnostic"] = {
        "status": "failed",
        "reason": "<script>bad</script>",
    }
    original = deepcopy(receipt)
    html = render_evidence_section([receipt])
    assert receipt == original
    for value in (
        PROOF_BOUNDARY,
        "Reported status: refused",
        "Reported passed",
        "Reported failed",
        "Reported refused",
        "Reported unavailable",
        "Maximum absolute error",
        "Maximum relative error",
        "Absolute tolerance",
        "Relative tolerance",
        "0.2",
        "first_observation_uses_prior",
        "extra_predict",
        "conditioning_horizon",
        "joint_gaussian_prefix_conditioning",
        "artifact_only",
        "reported-version",
        "a" * 64,
    ):
        assert value in html
    assert "&lt;img src=x&gt;" in html and "<img src=x>" not in html
    assert html.count("unique retained program") == 1
    assert html.count("&lt;img src=x&gt;") == 1
    assert '"utf8_bytes": 56' in html
    assert "Complete saved receipt (unverified) below" in html
    assert "&lt;script&gt;source only&lt;/script&gt;" in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert '<label for="evidence-query">' in html
    assert '<label for="evidence-status">' in html
    assert 'role="status" aria-live="polite"' in html
    assert "fetch(" not in html

    # Raw JSON remains complete and safely copyable from a text node. Text-node
    # quotes need no attribute escaping, and metadata summarization cannot
    # mutate the caller's receipt or discard malformed optional records.
    pres = []
    parser = HTMLParser()
    active = False

    def start(tag, attrs):
        nonlocal active
        if tag == "pre":
            pres.append("")
            active = True

    def end(tag):
        nonlocal active
        if tag == "pre":
            active = False

    def content(value):
        if active:
            pres[-1] += value

    parser.handle_starttag = start
    parser.handle_endtag = end
    parser.handle_data = content
    parser.feed(html)
    assert json.loads(pres[-1]) == receipt
    receipt["artifacts"]["source"] = {"utf8": {"malformed": True}}
    assert '"malformed": true' in render_evidence_section([receipt])


def test_fep_audit_keeps_replay_references_compilation_execution_and_pair_separate() -> (
    None
):
    audit = {
        "schema": "fep-semantic-witness-audit-v1",
        "status": "failed",
        "gnn_witness": _receipt(),
        "independent_replay": {
            "algorithm": "gaussian_kalman_joseph",
            "status": "failed",
            "comparisons": {
                "posterior_cov": {
                    "status": "failed",
                    "reason": "covariance mismatch",
                    "max_abs_error": 0.3,
                    "atol": 1e-12,
                    "rtol": 1e-10,
                }
            },
        },
        "formal_references": [
            {
                "declaration": "GaussianFilter.condition",
                "alignment": "structural_analogue",
                "source_sha256": "c" * 64,
                "assumptions": ["positive definite covariance"],
                "unmatched_assumptions": ["native label provenance"],
            }
        ],
        "formal_evidence": {
            "status": "referenced_not_compiled",
            "universal_equivalence": False,
            "native_claim_ready": False,
        },
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
        },
        "pair_evidence": {"status": "unavailable", "current_pair_synchronized": False},
        "live_sources": {"fep": {"revision": "current-fep-source"}},
    }
    html = render_evidence_section([audit])
    assert "FEP recorded replay comparisons" in html
    assert "covariance mismatch" in html
    assert "GaussianFilter.condition" in html
    assert "structural_analogue" in html
    assert "unmatched_assumptions" in html
    assert "referenced_not_compiled" in html
    assert "<dt>Lean compilation</dt><dd>unavailable</dd>" in html
    assert "<dt>Execution</dt><dd>artifact_only</dd>" in html
    assert "<dt>Pair custody</dt><dd>unavailable</dd>" in html
    assert "current-fep-source" in html
    assert PROOF_BOUNDARY in html


def test_legacy_and_cross_framework_admission_do_not_hide_quantity_failures() -> None:
    legacy = {
        "schema": "gnn-independent-numerical-witness-v1",
        "source_contract": {"model": "finite"},
        "native": {
            "script_sha256": "b" * 64,
            "script_utf8": "unique historical script <x>",
            "result_utf8": "unique historical result",
        },
        "comparisons": {
            "beliefs": {
                "status": "passed",
                "max_abs_error": 0,
                "rtol": 1e-10,
                "atol": 1e-12,
            }
        },
    }
    comparison = {
        "comparisons": {
            "metric_agreement": {
                "jax_vs_numpy": {
                    "status": "comparable",
                    "model_id": "source-bound-model",
                    "source_sha256": "a" * 64,
                    "inference_mode": "gaussian_filtering",
                    "tolerance": {"atol": 1e-10, "rtol": 1e-8},
                    "views": {
                        "joint": {
                            "within_tolerance": True,
                            "max_absolute_difference": 0,
                            "covariance_within_tolerance": False,
                            "covariance_max_absolute_difference": 0.25,
                        }
                    },
                    "unavailable_metrics": {
                        "expected_free_energy": "missing declared convention"
                    },
                }
            },
            "scientific_exclusions": {"other[0]": ["no source-bound counterpart"]},
        }
    }
    html = render_evidence_section([legacy, comparison])
    assert 'data-evidence-status="comparable"' in html
    assert "joint / covariance</th><td>Reported failed" in html
    assert "0.25" in html
    assert "missing declared convention" in html
    assert "no source-bound counterpart" in html
    assert "Reported status: comparable" in html
    assert "Admitted for comparison" in html
    assert "4 of 4" not in html  # one legacy receipt, one pair, one exclusion
    assert "3 of 3 recorded evidence entries shown." in html
    assert html.count("unique historical script &lt;x&gt;") == 1
    assert html.count("unique historical result") == 1
    assert "b" * 64 in html


def test_native_formal_receipt_distinguishes_positive_compilation_from_rejected_controls() -> (
    None
):
    receipt = {
        "schema": "fep-gnn-semantic-native-v1",
        "status": "passed",
        "source_path": "formal/GNNSemanticWitness.lean",
        "source_sha256": "c" * 64,
        "toolchain": {
            "lean_toolchain": "leanprover/lean4:v4.24.0",
            "mathlib_tag": "v4.24.0",
            "mathlib_revision": "mathlib-source-revision",
        },
        "results": {
            "positive": {
                "compiles": True,
                "has_sorry": False,
                "accepted": True,
                "stdout": "theorem uses axioms: [propext, Classical.choice]",
                "stderr": "",
                "errors": [],
                "warnings": [],
                "source_sha256": "d" * 64,
            },
            "zero-normalizer": {
                "compiles": False,
                "has_sorry": False,
                "accepted": True,
                "stdout": "",
                "stderr": "false normalization claim rejected",
                "errors": ["type mismatch"],
                "warnings": [],
            },
            "omitted-observation-noise": {
                "compiles": False,
                "has_sorry": False,
                "accepted": True,
                "stdout": "",
                "stderr": "false covariance claim rejected",
                "errors": ["unsolved goals"],
                "warnings": [],
            },
        },
        "execution_source_verified": False,
        "universal_equivalence": False,
    }
    html = render_evidence_section([receipt])
    assert "Retained native Lean compilation" in html
    assert (
        "Global formal artifact; no source-model numerical binding is claimed." in html
    )
    assert "Reported positive compilation: compiled" in html
    assert html.count("Reported negative control: false claim rejected") == 2
    assert "Reported failed" not in html
    assert "Recorded numerical comparisons" not in html
    assert "theorem uses axioms: [propext, Classical.choice]" in html
    assert "mathlib-source-revision" in html
    assert (
        "Finite-weight normalization and scalar arbitrary-H prior measurement algebra only"
        in html
    )
    assert (
        "no GNN parser/runner, matrix Gaussian conditioning or universal translation proof."
        in html
    )
    assert PROOF_BOUNDARY in html
    receipt["results"]["zero-normalizer"].update(compiles=True, accepted=False)
    unexpected = render_evidence_section([receipt])
    assert "false claim compiled unexpectedly" in unexpected


def test_pure_dict_site_renders_evidence_without_collectors(
    tmp_path: Path, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem collector invoked")

    monkeypatch.setattr(WebsiteGenerator, "_collect_all_data", forbidden)
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path / "site", "analysis": [_receipt()]}, filesystem=False
    )
    assert result["success"], result["errors"]
    assert result["pages_created"] == 7
    html = (tmp_path / "site/analysis.html").read_text()
    assert "Source-bound numerical evidence" in html
    scripts = []
    parser = HTMLParser()
    parser.handle_starttag = lambda tag, attrs: (
        scripts.append(dict(attrs)) if tag == "script" else None
    )
    parser.feed(html)
    assert scripts and all("src" not in script for script in scripts)


def test_bound_collection_excludes_unowned_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    from gnn.pipeline.run_context import build_run_context
    from gnn.website.collection import collect_website_data

    source = tmp_path / "input"
    (source / "selected").mkdir(parents=True)
    (source / "other").mkdir()
    source_bytes = b"# Selected source model\r\n"
    other_bytes = b"# Other source model\n"
    (source / "selected/source.md").write_bytes(source_bytes)
    (source / "other/source.md").write_bytes(other_bytes)
    output = tmp_path / "output"
    evidence = output / "16_analysis_output/semantic_witnesses"
    evidence.mkdir(parents=True)
    current = _receipt()
    current["source_contract"]["model"] = "current-evidence"
    current["source_contract"]["source_sha256"] = hashlib.sha256(
        source_bytes
    ).hexdigest()
    current["artifacts"]["source"] = {
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "utf8": source_bytes.decode("utf-8"),
    }
    old = _receipt()
    old["source_contract"]["model"] = "stale-evidence"
    (evidence / "stale.json").write_text(json.dumps(old))
    unselected = deepcopy(current)
    unselected["source_contract"]["model"] = "owned-unselected-evidence"
    unselected["source_contract"]["source_sha256"] = hashlib.sha256(
        other_bytes
    ).hexdigest()
    unselected["artifacts"]["source"] = {
        "sha256": hashlib.sha256(other_bytes).hexdigest(),
        "utf8": other_bytes.decode("utf-8"),
    }
    tampered = deepcopy(current)
    tampered["source_contract"]["model"] = "tampered-embedded-source"
    tampered["artifacts"]["source"]["utf8"] = "changed source bytes"
    missing = deepcopy(current)
    missing["source_contract"]["model"] = "missing-embedded-source"
    missing["artifacts"].pop("source")
    invalid = deepcopy(current)
    invalid["source_contract"]["model"] = "invalid-source-identity"
    invalid["source_contract"]["source_sha256"] = "not-a-digest"
    legacy = {
        "schema": "gnn-independent-numerical-witness-v1",
        "source_contract": {
            "model": "current-legacy-evidence",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "comparisons": {},
    }
    legacy_other = deepcopy(legacy)
    legacy_other["source_contract"] = {
        "model": "unselected-legacy-evidence",
        "source_sha256": hashlib.sha256(other_bytes).hexdigest(),
    }
    records = {
        "current.json": current,
        "other.json": unselected,
        "tampered.json": tampered,
        "missing.json": missing,
        "invalid.json": invalid,
        "legacy.json": legacy,
        "legacy-other.json": legacy_other,
        "fep.json": {"schema": "fep-semantic-witness-audit-v1", "gnn_witness": current},
        "fep-other.json": {
            "schema": "fep-semantic-witness-audit-v1",
            "gnn_witness": unselected,
        },
        "generic.json": {"file_name": "generic analysis remains available"},
        "native-proof.json": {
            "schema": "fep-gnn-semantic-native-v1",
            "source_sha256": "c" * 64,
            "status": "passed",
            "results": {},
        },
    }
    for name, record in records.items():
        (evidence / name).write_text(json.dumps(record))
    context = build_run_context(
        source,
        output,
        "evidence-view",
        [16, 20],
        {
            "testing_matrix": {
                "enabled": True,
                "folders": {"selected": [16, 20], "other": [16]},
            }
        },
    )
    assert len(context.selected_models(20)) == 1
    context_file = context.write(tmp_path / "context.json")
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(context_file))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)
    summaries = output / "00_pipeline_summary"
    summaries.mkdir()
    (summaries / "current_summary.json").write_text(
        json.dumps(
            {
                "run_id": context.run_id,
                "steps": [
                    {
                        "artifacts": [
                            {"path": f"16_analysis_output/semantic_witnesses/{name}"}
                            for name in records
                        ]
                    }
                ],
            }
        )
    )
    data = collect_website_data(output, source, output / "20_website_output/assets")
    html = render_evidence_section(data["analysis"])
    assert "current-evidence" in html
    assert "current-legacy-evidence" in html
    assert "stale-evidence" not in html
    for excluded in (
        "owned-unselected-evidence",
        "tampered-embedded-source",
        "missing-embedded-source",
        "invalid-source-identity",
        "unselected-legacy-evidence",
    ):
        assert excluded not in html
    assert len(data["analysis"]) == 5
    assert (
        "Global formal artifact; no source-model numerical binding is claimed." in html
    )
    assert any(
        row.get("file_name") == "generic analysis remains available"
        for row in data["analysis"]
    )


def test_malformed_rows_remain_visible_without_page_failure(tmp_path: Path) -> None:
    malformed = {
        "schema": "gnn-semantic-witness-v2",
        "status": True,
        "source_contract": [],
        "comparisons": {"beliefs": ["bad row"]},
        "formal_evidence": "not a mapping",
    }
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path / "site", "analysis": [None, "bad", malformed]},
        filesystem=False,
    )
    assert result["success"], result["errors"]
    html = (tmp_path / "site/analysis.html").read_text()
    assert "Reported status: unavailable" in html
    assert "Unusable comparison record; no outcome established." in html
    assert "Reported passed" not in html
    assert "No recognized numerical evidence receipts" in render_evidence_section(
        [None, {"schema": "unsupported"}]
    )
