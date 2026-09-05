#!/usr/bin/env python3
"""Tests for the refactored security policy and scanning layer.

Pins real behavior of the composability surface added in the 1.7.0 refactor:

- ``resolve_security_policy``: pure resolution and validation of the policy
  request (levels, explicit block_on, scan override, fail-closed invalid
  policies) — including parity with the policy block that
  ``process_security`` writes into ``security_results.json``.
- ``ResolvedSecurityPolicy.to_receipt``: the static receipt keys.
- ``findings_at_or_above`` / ``count_by_severity``: threshold filtering with
  fail-closed unknown severities, and severity histograms.
- ``scan_source``: in-memory Python source scanning (findings-only mode and
  verdict mode), including the memory-label receipt.
- ``SecurityScanError``: typed failure of ``perform_security_check`` that
  remains catchable as ``Exception`` for callers with broad exception handling.
- Real permission reporting in ``perform_security_check`` and the
  mode-driven world-writable detection in ``check_vulnerabilities``.

All tests are deterministic, isolated (tmp_path), and network-free.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from security import (
    ResolvedSecurityPolicy,
    SecurityScanError,
    count_by_severity,
    findings_at_or_above,
    resolve_security_policy,
    scan_script_for_execution,
    scan_source,
)
from security.processor import check_vulnerabilities, perform_security_check


class TestResolveSecurityPolicy:
    """Policy resolution: pure, total, and receipt-compatible."""

    def test_strict_level_defaults_to_high_gate(self) -> None:
        policy = resolve_security_policy("strict")
        assert policy.is_valid is True
        assert policy.security_level == "strict"
        assert policy.scan_vulnerabilities is True
        assert policy.block_on == "high"
        assert policy.enforced is True
        assert policy.error is None

    def test_standard_scans_but_does_not_enforce(self) -> None:
        policy = resolve_security_policy("standard")
        assert policy.is_valid
        assert policy.scan_vulnerabilities is True
        assert policy.block_on is None
        assert policy.enforced is False

    def test_basic_neither_scans_nor_enforces(self) -> None:
        policy = resolve_security_policy("basic")
        assert policy.is_valid is True
        assert policy.scan_vulnerabilities is False
        assert policy.block_on is None
        assert policy.enforced is False

    def test_explicit_block_on_enables_enforcement_on_any_level(self) -> None:
        policy = resolve_security_policy("basic", block_on="medium")
        assert policy.is_valid is True
        assert policy.enforced is True
        assert policy.block_on == "medium"
        # An explicit block_on forces scanning on even for "basic".
        assert policy.scan_vulnerabilities is True

    def test_invalid_level_fails_closed_with_error(self) -> None:
        policy = resolve_security_policy("paranoid")
        assert policy.is_valid is False
        assert policy.error is not None
        assert "Invalid security policy" in policy.error

    def test_invalid_block_on_fails_closed_and_is_echoed(self) -> None:
        policy = resolve_security_policy("standard", block_on="critical")
        assert policy.is_valid is False
        assert policy.requested_block_on == "critical"
        # An unresolvable threshold must not leak a default into the receipt.
        assert policy.block_on is None

    def test_non_bool_scan_override_is_invalid(self) -> None:
        policy = resolve_security_policy("standard", check_vulnerabilities="yes")
        assert policy.is_valid is False
        # Non-bool overrides are stringified into the receipt for forensics.
        assert policy.requested_scan_vulnerabilities == "yes"

    @pytest.mark.parametrize(
        ("level", "scan", "expected_valid"),
        [
            ("strict", False, False),
            ("strict", True, True),
            ("standard", False, True),
            ("basic", False, True),
        ],
    )
    def test_scan_disable_matrix(
        self, level: str, scan: bool, expected_valid: bool
    ) -> None:
        assert (
            resolve_security_policy(level, check_vulnerabilities=scan).is_valid
            is expected_valid
        )

    def test_enforced_policy_cannot_disable_scanning(self) -> None:
        policy = resolve_security_policy(
            "standard", block_on="low", check_vulnerabilities=False
        )
        assert policy.is_valid is False

    def test_to_receipt_matches_process_security_policy_keys(self) -> None:
        policy = resolve_security_policy("strict")
        receipt = policy.to_receipt()
        assert receipt == {
            "security_level": "strict",
            "enforced": True,
            "scan_vulnerabilities": True,
            "requested_scan_vulnerabilities": None,
            "requested_block_on": None,
            "block_on": "high",
        }

    def test_resolution_is_case_and_whitespace_insensitive(self) -> None:
        policy = resolve_security_policy("  STRICT ", block_on=" HIGH ")
        assert policy.is_valid is True
        assert policy.security_level == "strict"
        assert policy.block_on == "high"

    def test_resolution_is_pure_never_raises(self) -> None:
        # None is not a valid bool or str for any slot; resolution must still
        # return a (invalid) policy rather than raising.
        policy = resolve_security_policy(None, None, None)
        assert policy.is_valid is False


class TestSeverityHelpers:
    """findings_at_or_above and count_by_severity semantics."""

    def test_threshold_filters_in_input_order(self) -> None:
        findings = [
            {"severity": "low", "id": 1},
            {"severity": "high", "id": 2},
            {"severity": "medium", "id": 3},
        ]
        blocked = findings_at_or_above(findings, "medium")
        assert [f["id"] for f in blocked] == [2, 3]

    def test_unknown_severity_fails_closed_to_high(self) -> None:
        findings = [{"severity": "catastrophic"}, {"severity": "low"}]
        blocked = findings_at_or_above(findings, "high")
        assert blocked == [{"severity": "catastrophic"}]

    def test_missing_severity_fails_closed_to_high(self) -> None:
        blocked = findings_at_or_above([{"id": "no-severity"}], "high")
        assert blocked == [{"id": "no-severity"}]

    def test_case_insensitive_threshold(self) -> None:
        assert findings_at_or_above([{"severity": "high"}], "HIGH") == [
            {"severity": "high"}
        ]

    def test_count_by_severity_omits_absent_labels(self) -> None:
        findings = [{"severity": "high"}, {"severity": "high"}, {"severity": "low"}]
        assert count_by_severity(findings) == {"high": 2, "low": 1}

    def test_count_by_severity_empty_input(self) -> None:
        assert count_by_severity([]) == {}

    def test_count_by_severity_defaults_unlabeled_to_medium(self) -> None:
        assert count_by_severity([{}]) == {"medium": 1}


class TestScanSource:
    """In-memory source scanning (pre-write validation for the render step)."""

    def test_clean_source_has_no_findings(self) -> None:
        result = scan_source("x = [1, 2, 3]\nprint(sum(x))\n")
        assert result["findings"] == []
        assert result["file_name"] == "<memory>.py"
        # No threshold given -> no verdict fields.
        assert "ok" not in result and "blocked" not in result

    def test_detects_eval_in_memory(self) -> None:
        result = scan_source("payload = '1+1'\nresult = eval(payload)\n")
        types = [f["vulnerability_type"] for f in result["findings"]]
        assert "Code injection via eval()" in types

    def test_block_on_produces_verdict_fields(self) -> None:
        result = scan_source(
            "import subprocess\nsubprocess.call(['ls'], shell=True)\n",
            block_on="high",
        )
        assert result["ok"] is False
        assert result["decision"] == "deny"
        assert result["block_on"] == "high"
        assert any("shell=True" in f["vulnerability_type"] for f in result["blocked"])

    def test_allow_verdict_when_below_threshold(self) -> None:
        # subprocess.run is low severity by AST classification.
        result = scan_source(
            "import subprocess\nsubprocess.run(['echo'])\n", block_on="high"
        )
        assert result["ok"] is True
        assert result["blocked"] == []
        assert result["decision"] == "allow"

    def test_custom_file_name_labels_findings(self) -> None:
        result = scan_source("eval(x)\n", file_name="rendered_model.py")
        assert result["file_name"] == "rendered_model.py"
        assert result["findings"][0]["file_name"] == "rendered_model.py"

    def test_invalid_threshold_fails_closed(self) -> None:
        result = scan_source("eval(x)\n", block_on="critical")
        assert result["ok"] is False
        assert result["decision"] == "deny_invalid_policy"

    def test_syntax_error_is_high_severity_finding(self) -> None:
        result = scan_source("def broken(:\n", block_on="high")
        assert result["ok"] is False
        assert result["findings"][0]["detection_method"] == "ast_parse"


class TestTypedScanError:
    """perform_security_check fails with the typed SecurityScanError."""

    def test_missing_file_raises_security_scan_error(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityScanError):
            perform_security_check(tmp_path / "absent.md")

    def test_error_is_catchable_as_exception(self, tmp_path: Path) -> None:
        # MCP wrappers and the orchestrator catch bare Exception; the typed
        # error must not escape that contract.
        try:
            perform_security_check(tmp_path / "absent.md")
        except Exception:  # noqa: B014 - the contract under test
            pass
        else:
            pytest.fail("expected SecurityScanError")

    def test_error_names_the_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.md"
        with pytest.raises(SecurityScanError) as excinfo:
            perform_security_check(missing)
        assert str(missing) in str(excinfo.value)


class TestPermissionSemantics:
    """Real permission reporting and mode-driven world-writable detection."""

    def test_perform_security_check_reports_octal_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "model.md"
        target.write_text("# clean\n")
        target.chmod(0o644)
        result = perform_security_check(target)
        assert result["file_permissions"] == "0o644"

    def test_perform_security_check_reports_octal_mode_on_readable_file(
        self, tmp_path: Path
    ) -> None:
        # The stat branch succeeds for a normal readable file and reports the
        # real POSIX mode; 0o600 exercises a non-default mode.
        target = tmp_path / "private.md"
        target.write_text("secret = 'x'\n")
        target.chmod(0o600)
        try:
            result = perform_security_check(target)
            assert result["file_permissions"] == "0o600"
        finally:
            target.chmod(0o644)

    def test_perform_security_check_unreadable_file_raises_typed_error(
        self, tmp_path: Path
    ) -> None:
        # The read failure is wrapped in the typed error — pin ONLY
        # SecurityScanError so a silently un-wrapped PermissionError escape
        # would fail this test.
        target = tmp_path / "locked.md"
        target.write_text("secret = 'x'\n")
        target.chmod(0o000)
        try:
            with pytest.raises(SecurityScanError):
                perform_security_check(target)
        finally:
            target.chmod(0o644)

    def test_stat_failure_reports_unknown_permissions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When stat itself fails (OSError), the receipt must degrade to
        # "unknown" instead of crashing — deterministic via monkeypatch, so
        # the branch is covered identically on every platform.
        target = tmp_path / "model.md"
        target.write_text("# clean\n")

        def boom(self: Path) -> os.stat_result:
            raise OSError("stat unavailable")

        monkeypatch.setattr(Path, "stat", boom)
        result = perform_security_check(target)
        assert result["file_permissions"] == "unknown"

    def test_world_writable_python_file_is_flagged(self, tmp_path: Path) -> None:
        target = tmp_path / "writable.py"
        target.write_text("print('hi')\n")
        target.chmod(0o666)
        try:
            vulns = check_vulnerabilities(target)
            world_writable = [
                v
                for v in vulns
                if v["vulnerability_type"] == "World-writable file permissions"
            ]
            assert len(world_writable) == 1
            assert world_writable[0]["severity"] == "low"
            assert world_writable[0]["context"].startswith("Mode: 0o")
        finally:
            target.chmod(0o644)

    def test_strictly_read_only_python_file_is_not_flagged(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "readonly.py"
        target.write_text("print('hi')\n")
        target.chmod(0o444)
        try:
            vulns = check_vulnerabilities(target)
            assert not any(
                v["vulnerability_type"] == "World-writable file permissions"
                for v in vulns
            )
        finally:
            target.chmod(0o644)

    def test_world_writable_markdown_is_not_flagged(self, tmp_path: Path) -> None:
        # Only .py files are permission-checked (generated scripts).
        target = tmp_path / "note.md"
        target.write_text("# md\n")
        target.chmod(0o666)
        try:
            vulns = check_vulnerabilities(target)
            assert not any(v["detection_method"] == "permission_check" for v in vulns)
        finally:
            target.chmod(0o644)


class TestGateVerdictContract:
    """scan_script_for_execution receipt keys stay stable for Step 12."""

    def test_verdict_keys_complete(self, tmp_path: Path) -> None:
        script = tmp_path / "clean.py"
        script.write_text("x = 1\n")
        verdict: dict[str, Any] = scan_script_for_execution(script)
        assert set(verdict) == {
            "ok",
            "blocked",
            "findings",
            "scanned",
            "block_on",
            "decision",
        }
        assert verdict["block_on"] == "high"
        assert verdict["decision"] == "allow"

    def test_invalid_threshold_receipt_keeps_block_on_key(self, tmp_path: Path) -> None:
        script = tmp_path / "any.py"
        verdict = scan_script_for_execution(script, block_on="critical")
        assert verdict["decision"] == "deny_invalid_policy"
        assert verdict["block_on"] == "critical"
        assert verdict["scanned"] is False

    def test_deny_receipts_have_single_finding(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.py"
        verdict = scan_script_for_execution(missing)
        assert verdict["decision"] == "deny_unreadable"
        assert verdict["findings"] == verdict["blocked"]
        assert len(verdict["findings"]) == 1


class TestPolicyRoundTrip:
    """resolve_security_policy output flows into process_security receipts."""

    def test_process_security_receipt_uses_resolver(self, tmp_path: Path) -> None:
        from security import process_security

        target = tmp_path / "in"
        target.mkdir()
        (target / "m.md").write_text("# model\nA = [[0.5]]\n")
        out = tmp_path / "out"

        # The resolver says this is invalid; the receipt must agree.
        requested = resolve_security_policy("standard", "not-a-threshold")
        assert requested.is_valid is False

        success = process_security(
            target, out, security_level="standard", block_on="not-a-threshold"
        )
        assert success is False
        receipt = (out / "security_results.json").read_text()
        import json

        data = json.loads(receipt)
        assert data["policy"]["decision"] == "deny_invalid_policy"
        assert data["policy"]["requested_block_on"] == "not-a-threshold"

    def test_resolved_policy_feeds_gate_threshold(self, tmp_path: Path) -> None:
        policy = resolve_security_policy("strict")
        script = tmp_path / "s.py"
        script.write_text("import subprocess\nsubprocess.run(['echo'])\n")
        verdict = scan_script_for_execution(script, block_on=policy.block_on or "high")
        # policy.block_on == "high": low-severity subprocess.run passes.
        assert verdict["ok"] is True
