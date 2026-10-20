"""Real subprocess/file controls for the workflow's runtime custody wrapper.

The tiny fixture CLI exercises custody, not the scientific bridge contract;
the workflow separately executes the actual pinned FEP bridge.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path

HELPER = Path(__file__).with_name("fep_paired_runtime.py")
CLI = """import json, sys
from pathlib import Path
def main(argv):
    root = Path(argv[argv.index("--project-root") + 1])
    if (root / "fail").exists():
        print('{"status": "error"}')
        return 7
    if (root / "mutate").exists():
        (root / "uv.lock").write_text("changed during command")
    if "status" in argv:
        result = {"status": "ok", "native_claim_ready": False}
    else:
        result = {"status": "ok", "read_only": True, "model": argv[argv.index("--model") + 1]}
    print(json.dumps(result))
    return 0
"""


class RuntimeCustodyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fep-runtime-control-"))
        print(f"retained control: {self.root}", flush=True)
        self.environment = self.root / "environment"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.environment)
        self.python = self.environment / "bin/python"
        self.version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.gnn = self.root / "gnn"
        self.fep = self.root / "fep"
        for root in (self.gnn, self.fep):
            root.mkdir()
            self.git(root, "init", "-q")
            self.git(root, "config", "core.fsmonitor", "false")
            self.git(root, "config", "user.name", "Runtime test")
            self.git(root, "config", "user.email", "runtime@example.invalid")
            (root / "uv.lock").write_text("fixture lock\n")
        package = self.fep / "src/fep_lean"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text(CLI)
        self.fep_revision = self.commit(self.fep)
        pin = self.gnn / ".github/fep-lean-pair.json"
        pin.parent.mkdir()
        pin.write_text(
            json.dumps(
                {
                    "repository": "ActiveInferenceInstitute/fep_formal",
                    "revision": self.fep_revision,
                }
            )
        )
        self.gnn_revision = self.commit(self.gnn)
        site = subprocess.check_output(
            [str(self.python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
        Path(site, "fixture.pth").write_text(str(self.fep / "src"))
        self.receipts = self.root / "receipts"
        self.receipts.mkdir()

    def git(self, root, *args):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args], text=True
        ).strip()

    def commit(self, root):
        self.git(root, "add", ".")
        self.git(root, "commit", "-qm", "fixture")
        return self.git(root, "rev-parse", "HEAD")

    def run_stage(
        self, stage, *, version=None, python=None, environment=None, success=True
    ):
        result = subprocess.run(
            [
                str(python or self.python),
                str(HELPER),
                stage,
                "--expected-python",
                version or self.version,
                "--environment",
                str(environment or self.environment),
                "--gnn-root",
                str(self.gnn),
                "--gnn-revision",
                self.gnn_revision,
                "--fep-root",
                str(self.fep),
                "--fep-revision",
                self.fep_revision,
                "--receipts",
                str(self.receipts),
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode == 0, success, result.stderr)
        if not success:
            self.assertTrue((self.receipts / f"runtime-{stage}-failed.json").is_file())
        return result

    def test_actual_cli_execution_and_unchanged_environment(self):
        for stage in ("installed", "status", "finite", "continuous", "final"):
            self.run_stage(stage)
        receipt = json.loads((self.receipts / "runtime-final.json").read_text())
        self.assertEqual(receipt["before"], receipt["after"])
        self.assertEqual(receipt["before"]["runtime"]["actual_python"], self.version)
        self.assertEqual(receipt["before"]["runtime"]["prefix"], str(self.environment))
        self.assertFalse(receipt["native_claim_ready"])
        # Successful bridge commands alone cannot establish current Q7 freshness.
        self.run_stage("q7-source", success=False)
        rejected = json.loads(
            (self.receipts / "runtime-q7-source-failed.json").read_text()
        )
        self.assertIn(
            "verify_native.py" if self.version == "3.14" else "CPython 3.14",
            rejected["error"],
        )

    def test_wrong_requested_interpreter_rejected(self):
        self.run_stage(
            "installed",
            version="3.12" if self.version == "3.11" else "3.11",
            success=False,
        )

    def test_wrong_environment_rejected(self):
        self.run_stage("installed", environment=self.root / "wrong", success=False)

    def test_changed_environment_configuration_rejected(self):
        self.run_stage("installed")
        cfg = self.environment / "pyvenv.cfg"
        cfg.write_text(cfg.read_text() + "\nfixture = changed\n")
        self.run_stage("status", success=False)

    def test_changed_environment_inode_rejected(self):
        self.run_stage("installed")
        retained = self.root / "retained_environment"
        self.environment.rename(retained)
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.environment)
        self.run_stage("status", success=False)

    def test_source_revision_drift_rejected(self):
        self.run_stage("installed")
        (self.gnn / "new").write_text("different source")
        self.commit(self.gnn)
        self.run_stage("status", success=False)

    def test_source_byte_drift_rejected(self):
        self.run_stage("installed")
        (self.fep / "uv.lock").write_text("uncommitted drift")
        self.run_stage("status", success=False)

    def test_mutation_during_cli_rejected(self):
        self.run_stage("installed")
        (self.fep / "mutate").touch()
        self.run_stage("status", success=False)

    def test_failed_cli_retains_output_and_exit_code(self):
        self.run_stage("installed")
        (self.fep / "fail").touch()
        self.run_stage("status", success=False)
        receipt = json.loads((self.receipts / "runtime-status-failed.json").read_text())
        self.assertEqual(receipt["exit_code"], 7)
        self.assertEqual(
            json.loads((self.receipts / "bridge-status.json").read_text())["status"],
            "error",
        )

    def test_missing_receipt_rejected(self):
        self.run_stage("status", success=False)

    def test_wrong_receipt_stage_rejected(self):
        self.run_stage("installed")
        path = self.receipts / "runtime-installed.json"
        receipt = json.loads(path.read_text())
        receipt["stage"] = "final"
        path.write_text(json.dumps(receipt))
        self.run_stage("status", success=False)

    def test_changed_output_receipt_rejected(self):
        self.run_stage("installed")
        self.run_stage("status")
        (self.receipts / "bridge-status.json").write_text(
            '{"status": "ok", "native_claim_ready": false}'
        )
        self.run_stage("finite", success=False)

    def test_wrong_command_receipt_rejected(self):
        self.run_stage("installed")
        self.run_stage("status")
        path = self.receipts / "runtime-status.json"
        receipt = json.loads(path.read_text())
        receipt["command"] = ["catalogue"]
        path.write_text(json.dumps(receipt))
        self.run_stage("finite", success=False)

    def test_final_requires_every_stage(self):
        self.run_stage("installed")
        self.run_stage("final", success=False)

    def test_existing_receipt_not_overwritten(self):
        self.run_stage("installed")
        path = self.receipts / "runtime-installed.json"
        original = path.read_bytes()
        self.run_stage("installed", success=False)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
