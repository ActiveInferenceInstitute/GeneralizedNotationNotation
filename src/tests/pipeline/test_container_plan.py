#!/usr/bin/env python3
"""
Tests for pipeline/container_plan.py — auditable container plans.

Real objects only: all tests use real Pydantic models, real serialization, and
real hashing. Includes positive (clean hardened plan) and negative-control
(deliberately insecure plan) security-review tests, deterministic-hash checks,
rollback semantics, and a source-level safety assertion proving the module has
no container-execution path.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _benign_config() -> list[dict[str, Any]]:
    """A spec config that, after hardening, should review completely clean."""
    return [
        {
            "name": "web",
            "image": "registry.example.com/web@sha256:" + ("a" * 64),
            "env": {"DB_PASSWORD": "${DB_PASSWORD}"},  # indirect reference, safe
        }
    ]


class TestGenerateAndReviewClean:
    def test_hardened_default_is_clean(self) -> Any:
        from pipeline.container_plan import generate_container_plan, security_review

        plan = generate_container_plan("svc", _benign_config())
        findings = security_review(plan)
        assert findings == [], f"expected clean plan, got: {findings}"

    def test_hardened_defaults_applied(self) -> Any:
        from pipeline.container_plan import generate_container_plan

        plan = generate_container_plan("svc", _benign_config())
        spec = plan.specs[0]
        assert spec.user == "nonroot"
        assert spec.read_only_rootfs is True
        assert spec.privileged is False
        assert spec.cap_drop == ["ALL"]
        assert spec.resources.cpu == "1.0"
        assert spec.resources.memory == "512Mi"

    def test_plan_hash_is_set(self) -> Any:
        from pipeline.container_plan import generate_container_plan

        plan = generate_container_plan("svc", _benign_config())
        assert len(plan.plan_hash) == 64
        assert plan.version == 1
        assert plan.rollback is None


class TestSecurityReviewNegativeControl:
    """NEGATIVE CONTROL — a deliberately insecure plan must trip every check."""

    def _insecure_plan(self) -> Any:
        from pipeline.container_plan import (
            ContainerPlan,
            ContainerSpec,
            ResourceLimits,
        )

        spec = ContainerSpec(
            name="bad",
            image="myimage:latest",  # no @sha256 digest
            user="root",
            read_only_rootfs=False,
            privileged=True,
            env={"DB_PASSWORD": "hunter2"},  # plaintext secret
            cap_drop=[],
            resources=ResourceLimits(cpu="", memory=""),
        )
        return ContainerPlan(plan_id="bad-plan", specs=[spec])

    def test_each_issue_is_reported(self) -> Any:
        from pipeline.container_plan import security_review

        findings = security_review(self._insecure_plan())
        by_code = {f.code: f for f in findings}

        assert by_code["PRIVILEGED_CONTAINER"].severity == "CRITICAL"
        assert by_code["ROOT_USER"].severity == "HIGH"
        assert by_code["UNPINNED_IMAGE"].severity == "HIGH"
        assert by_code["PLAINTEXT_SECRET"].severity == "HIGH"
        assert by_code["UNBOUNDED_RESOURCES"].severity == "MEDIUM"
        assert by_code["WRITABLE_ROOTFS"].severity == "LOW"
        assert by_code["CAPS_NOT_DROPPED"].severity == "LOW"

        severities = sorted({f.severity for f in findings})
        assert severities == ["CRITICAL", "HIGH", "LOW", "MEDIUM"]
        # All findings attribute to the offending spec.
        assert all(f.spec_name == "bad" for f in findings)

    def test_empty_user_also_flags_root(self) -> Any:
        from pipeline.container_plan import (
            ContainerPlan,
            ContainerSpec,
            security_review,
        )

        spec = ContainerSpec(
            name="x",
            image="img@sha256:" + ("b" * 64),
            user="",  # empty == insecure
            read_only_rootfs=True,
            cap_drop=["ALL"],
        )
        spec.resources.cpu = "1.0"
        spec.resources.memory = "256Mi"
        findings = security_review(ContainerPlan(plan_id="p", specs=[spec]))
        codes = {f.code for f in findings}
        assert codes == {"ROOT_USER"}

    def test_reference_env_value_is_not_a_secret(self) -> Any:
        from pipeline.container_plan import (
            ContainerPlan,
            ContainerSpec,
            security_review,
        )

        spec = ContainerSpec(
            name="x",
            image="img@sha256:" + ("c" * 64),
            user="nonroot",
            read_only_rootfs=True,
            cap_drop=["ALL"],
            env={"API_TOKEN": "${API_TOKEN}"},  # indirect, not plaintext
        )
        spec.resources.cpu = "1.0"
        spec.resources.memory = "256Mi"
        findings = security_review(ContainerPlan(plan_id="p", specs=[spec]))
        assert findings == []


class TestRollback:
    def test_v2_attaches_rollback_to_v1(self) -> Any:
        from pipeline.container_plan import generate_container_plan

        v1 = generate_container_plan("svc", _benign_config())
        v2 = generate_container_plan("svc", _benign_config(), previous=v1)

        assert v1.version == 1
        assert v2.version == 2
        assert v2.rollback is not None
        assert v2.rollback.previous_version == 1
        assert v2.rollback.previous_plan_hash == v1.plan_hash
        assert v2.rollback.strategy == "redeploy-previous"


class TestHashDeterminism:
    def test_same_specs_same_hash(self) -> Any:
        from pipeline.container_plan import (
            compute_plan_hash,
            generate_container_plan,
        )

        a = generate_container_plan("svc", _benign_config())
        b = generate_container_plan("svc", _benign_config())
        assert compute_plan_hash(a) == compute_plan_hash(b)
        assert a.plan_hash == b.plan_hash

    def test_changed_spec_changes_hash(self) -> Any:
        from pipeline.container_plan import (
            compute_plan_hash,
            generate_container_plan,
        )

        a = generate_container_plan("svc", _benign_config())
        changed = _benign_config()
        changed[0]["command"] = ["./serve", "--port", "8443"]
        b = generate_container_plan("svc", changed)
        assert compute_plan_hash(a) != compute_plan_hash(b)

    def test_hash_excludes_plan_hash_field(self) -> Any:
        from pipeline.container_plan import (
            compute_plan_hash,
            generate_container_plan,
        )

        plan = generate_container_plan("svc", _benign_config())
        before = compute_plan_hash(plan)
        plan.plan_hash = "tampered-value"
        # Hash recomputes the same value because plan_hash is excluded.
        assert compute_plan_hash(plan) == before


class TestSerialize:
    def test_serialize_is_sorted_and_roundtrips(self) -> Any:
        from pipeline.container_plan import generate_container_plan, serialize_plan

        plan = generate_container_plan("svc", _benign_config())
        text = serialize_plan(plan)
        data = json.loads(text)
        assert data["plan_id"] == "svc"
        assert data["plan_hash"] == plan.plan_hash
        # Deterministic: serializing twice yields byte-identical output.
        assert serialize_plan(plan) == text
        # Top-level keys are sorted.
        keys = list(data.keys())
        assert keys == sorted(keys)

    def test_to_compose_is_pure_data(self) -> Any:
        from pipeline.container_plan import generate_container_plan, plan_to_compose

        plan = generate_container_plan("svc", _benign_config())
        compose = plan_to_compose(plan)
        assert "web" in compose["services"]
        assert compose["services"]["web"]["read_only"] is True
        assert compose["x-plan-hash"] == plan.plan_hash


class TestSafetyContract:
    def test_module_has_no_execution_path(self) -> Any:
        """SAFETY: the module must not *import or call* any execution primitive.

        An AST check (not a substring grep) is used deliberately: a static
        security reviewer must be free to *name* dangerous things like
        ``/var/run/docker.sock`` in its detection logic without that being
        mistaken for an execution path. The real safety property is that the
        module imports no runtime/network library and makes no exec-style call.
        """
        from pipeline import container_plan

        tree = ast.parse(Path(container_plan.__file__).read_text())
        forbidden_imports = {
            "subprocess", "docker", "kubernetes", "socket", "requests",
            "urllib", "http", "asyncio", "paramiko", "pty",
        }
        forbidden_calls = {"system", "popen", "exec", "eval", "spawn", "fork"}
        bad_imports: list[str] = []
        bad_calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_imports:
                        bad_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in forbidden_imports:
                    bad_imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in forbidden_calls:
                    bad_calls.append(name)
        assert bad_imports == [], f"module imports execution primitives: {bad_imports}"
        assert bad_calls == [], f"module makes execution calls: {bad_calls}"


def test_unpinned_image_strict_digest_rejects_fake_digests() -> None:
    """NEGATIVE: malformed/short/non-hex digests and :latest are NOT valid pins."""
    from pipeline import container_plan as cp

    for bad_image in (
        "evil.com/x:latest@sha256:zzzz",            # non-hex digest
        "ubuntu:@sha256:nonsense",                  # malformed
        "x@sha256:" + ("z" * 64),                   # 64 non-hex chars
        "x@sha256:" + ("a" * 63),                   # too short
        "registry/app:latest",                      # mutable tag, no digest
    ):
        plan = cp.ContainerPlan(plan_id="p", specs=[cp.ContainerSpec(name="s", image=bad_image)])
        codes = {f.code for f in cp.security_review(plan)}
        assert "UNPINNED_IMAGE" in codes, f"{bad_image!r} should be flagged UNPINNED_IMAGE"
    # A real 64-hex digest is accepted (no UNPINNED finding).
    good = cp.ContainerPlan(
        plan_id="p",
        specs=[cp.ContainerSpec(
            name="s", image="registry/app@sha256:" + ("a" * 64), user="nonroot",
            read_only_rootfs=True, cap_drop=["ALL"], resources=cp.ResourceLimits(cpu="1", memory="1Mi"),
        )],
    )
    assert "UNPINNED_IMAGE" not in {f.code for f in cp.security_review(good)}


def test_sensitive_host_mount_is_critical() -> None:
    """NEGATIVE: mounting the host root or docker socket is a CRITICAL escape risk."""
    from pipeline import container_plan as cp

    for host_mount in ("/:/host", "/var/run/docker.sock:/var/run/docker.sock", "/proc:/proc", "/etc:/etc"):
        plan = cp.ContainerPlan(
            plan_id="p",
            specs=[cp.ContainerSpec(name="s", image="i@sha256:" + ("a" * 64), mounts=[host_mount])],
        )
        findings = cp.security_review(plan)
        crit = [f for f in findings if f.code == "SENSITIVE_HOST_MOUNT" and f.severity == "CRITICAL"]
        assert crit, f"{host_mount!r} should yield a CRITICAL SENSITIVE_HOST_MOUNT"


def test_host_namespace_and_dangerous_caps_flagged() -> None:
    """NEGATIVE: host network/pid namespaces and dangerous cap_add are HIGH findings."""
    from pipeline import container_plan as cp

    plan = cp.ContainerPlan(
        plan_id="p",
        specs=[cp.ContainerSpec(
            name="s", image="i@sha256:" + ("a" * 64),
            network="host", pid="host", cap_add=["SYS_ADMIN", "NET_ADMIN"],
        )],
    )
    codes = {f.code: f.severity for f in cp.security_review(plan)}
    assert codes.get("HOST_NETWORK") == "HIGH"
    assert codes.get("HOST_PID") == "HIGH"
    assert codes.get("DANGEROUS_CAP_ADD") == "HIGH"


def test_new_spec_fields_change_plan_hash() -> None:
    """The new fields participate in the deterministic plan hash."""
    from pipeline import container_plan as cp

    base = cp.ContainerSpec(name="s", image="i@sha256:" + ("a" * 64))
    hostile = base.model_copy(update={"network": "host"})
    p1 = cp.ContainerPlan(plan_id="p", specs=[base])
    p2 = cp.ContainerPlan(plan_id="p", specs=[hostile])
    assert cp.compute_plan_hash(p1) != cp.compute_plan_hash(p2)
