#!/usr/bin/env python3
"""Tests for the pre-execution security gate (scan_script_for_execution).

Covers the RED_TEAM_REVIEW V-01/V-06 remediation: rendered scripts are AST
scanned *before* Step 12 executes them, and high-severity findings block
execution.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from security.processor import scan_script_for_execution


class TestPreExecGate:
    """Pre-execution gate behavior on rendered scripts."""

    @pytest.mark.parametrize(
        "source,substring",
        [
            ("import os\nos.system('echo pwned')\n", "OS command injection"),
            ("x = eval('1+1')\n", "Code injection via eval"),
            ("exec('import os')\n", "Code injection via exec"),
            ("import pickle\npickle.loads(b'x')\n", "pickle.loads"),
            ("import pickle\ndata = pickle.load(open('f', 'rb'))\n", "pickle.load"),
        ],
    )
    def test_blocks_high_severity_calls(
        self, tmp_path: Path, source: str, substring: str
    ) -> None:
        script = tmp_path / "malicious.py"
        script.write_text(source)
        verdict = scan_script_for_execution(script)
        assert verdict["ok"] is False
        assert verdict["scanned"] is True
        assert any(substring in b["vulnerability_type"] for b in verdict["blocked"])

    def test_allows_clean_script(self, tmp_path: Path) -> None:
        script = tmp_path / "clean.py"
        script.write_text("x = [1, 2, 3]\nprint(sum(x))\n")
        verdict = scan_script_for_execution(script)
        assert verdict["ok"] is True
        assert verdict["blocked"] == []

    def test_blocks_unreadable_script(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.py"
        verdict = scan_script_for_execution(missing)
        assert verdict["ok"] is False
        assert verdict["scanned"] is False
        assert verdict["findings"] == verdict["blocked"]
        assert verdict["decision"] == "deny_unreadable"

    def test_block_on_threshold(self, tmp_path: Path) -> None:
        # `subprocess.run` is classified "low" severity by the AST scanner, so
        # blocking on "high" must let it through, while "low" must block it.
        script = tmp_path / "subproc.py"
        script.write_text("import subprocess\nsubprocess.run(['echo', 'hi'])\n")
        assert scan_script_for_execution(script, block_on="high")["ok"] is True
        assert scan_script_for_execution(script, block_on="low")["ok"] is False

    def test_blocks_shell_true_through_import_alias(self, tmp_path: Path) -> None:
        script = tmp_path / "shell.py"
        script.write_text(
            "import subprocess as sp\nsp.run('echo unsafe', shell=True)\n"
        )

        verdict = scan_script_for_execution(script)

        assert verdict["ok"] is False
        assert verdict["blocked"][0]["vulnerability_type"] == (
            "Subprocess execution with shell=True"
        )

    def test_blocks_direct_import_alias(self, tmp_path: Path) -> None:
        script = tmp_path / "direct.py"
        script.write_text("from os import system as invoke\ninvoke('echo unsafe')\n")

        assert scan_script_for_execution(script)["ok"] is False

    @pytest.mark.parametrize(
        "source",
        [
            "import subprocess\nsubprocess.check_call(['echo', 'unsafe'], shell=flag)\n",
            "import subprocess as sp\ninvoke = sp.check_output\ninvoke('echo unsafe', shell=1)\n",
        ],
    )
    def test_blocks_subprocess_aliases_with_potential_shell_execution(
        self, tmp_path: Path, source: str
    ) -> None:
        script = tmp_path / "subprocess_alias.py"
        script.write_text(source)

        verdict = scan_script_for_execution(script)

        assert verdict["ok"] is False
        assert verdict["decision"] == "deny"
        assert any(
            finding["vulnerability_type"] == "Subprocess execution with shell=True"
            for finding in verdict["blocked"]
        )

    def test_unparseable_python_fails_closed_with_consistent_receipt(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "malformed.py"
        script.write_text("if True print('not valid')\n")

        verdict = scan_script_for_execution(script)

        assert verdict["ok"] is False
        assert verdict["scanned"] is True
        assert verdict["decision"] == "deny"
        assert verdict["findings"] == verdict["blocked"]
        assert verdict["blocked"][0]["detection_method"] == "ast_parse"

    def test_rejects_invalid_threshold_and_unknown_script_type(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "model.sh"
        script.write_text("echo hello\n")

        assert scan_script_for_execution(script, block_on="critical")["ok"] is False
        verdict = scan_script_for_execution(script)
        assert verdict["ok"] is False
        assert verdict["scanned"] is False
        assert verdict["blocked"][0]["detection_method"] == "file_type_policy"

    def test_julia_script_blocked_when_malformed(self, tmp_path: Path) -> None:
        # Unbalanced parenthesis reliably triggers an :incomplete node in
        # Julia's Meta.parseall on Julia 1.12+.
        script = tmp_path / "model.jl"
        script.write_text("x = (1 + 2\n")
        verdict = scan_script_for_execution(script)
        if any(
            f["detection_method"] == "julia_meta_parseall" for f in verdict["findings"]
        ):
            # Julia available: malformed code is a hard high-severity block.
            assert verdict["scanned"] is True
            assert verdict["ok"] is False
            assert any(
                b["severity"] == "high" and "Malformed Julia" in b["vulnerability_type"]
                for b in verdict["blocked"]
            )
        else:
            # Julia unavailable: degraded to advisory sweep.
            assert verdict["scanned"] is False

    def test_julia_suspicious_but_valid_code_is_medium(self, tmp_path: Path) -> None:
        # Valid Julia that constructs a Cmd — suspicious (medium) but the
        # default high gate does not block it.
        script = tmp_path / "model.jl"
        script.write_text("cmd = Cmd([`echo hi`])\nrun(cmd)\n")
        verdict = scan_script_for_execution(script)
        assert any(f["severity"] == "medium" for f in verdict["findings"])
        assert verdict["ok"] is True

    def test_non_python_script_is_textual_only(self, tmp_path: Path) -> None:
        script = tmp_path / "model.jl"
        script.write_text("run(`echo hi`)\n")
        verdict = scan_script_for_execution(script)
        # Julia is parse-scanned when Julia is available; the backtick run()
        # pattern is flagged as a medium advisory finding and does not block
        # the default high-severity gate.
        assert verdict["ok"] is True
        assert any(
            f["vulnerability_type"].startswith("Julia") for f in verdict["findings"]
        )
