#!/usr/bin/env python3
"""
Auditable Container Plans — generate validated, hardened container plans with
static security review and rollback semantics.

This module is PURE: it builds, hashes, reviews, and serializes container-plan
DATA. It NEVER executes containers, invokes any runtime CLI, opens network
connections, or mutates any cluster. Security review is a static inspection of
the plan data only.

Public API:
  - ResourceLimits, ContainerSpec, RollbackDescriptor, ContainerPlan, Finding
  - compute_plan_hash(plan): deterministic SHA256 hex digest of the specs
  - generate_container_plan(plan_id, specs_config, previous=None): hardened plan
  - security_review(plan): list[Finding] (empty == clean)
  - serialize_plan(plan): deterministic sorted-key JSON
  - plan_to_compose(plan): pure compose-shaped dict (no I/O)
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

# Keys whose plaintext values are treated as candidate secrets.
_SECRET_KEY_RE = re.compile(r"PASSWORD|SECRET|TOKEN|KEY", re.IGNORECASE)
# A "${...}" style indirect reference is NOT a plaintext secret.
_REFERENCE_RE = re.compile(r"^\$\{[^}]+\}$")
# A properly pinned image ends with a full 64-hex-char sha256 digest. A mere
# "@sha256:" substring (with a fake/short/non-hex digest, or a trailing mutable
# tag) is NOT a valid pin.
_DIGEST_PIN_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
# Host paths whose mount into a container enables trivial host escape.
_SENSITIVE_HOST_MOUNTS = (
    "/",
    "/var/run/docker.sock",
    "/var/run",
    "/proc",
    "/sys",
    "/etc",
    "/dev",
    "/root",
    "/var/lib/kubelet",
)
# Linux capabilities that materially weaken container isolation when added.
_DANGEROUS_CAPS = frozenset(
    {
        "SYS_ADMIN",
        "NET_ADMIN",
        "SYS_PTRACE",
        "SYS_MODULE",
        "SYS_RAWIO",
        "DAC_READ_SEARCH",
        "BPF",
        "ALL",
    }
)


def _mount_host_path(mount: str) -> str:
    """Return the host-side path of a ``host:container[:mode]`` mount string."""
    return mount.split(":", 1)[0].strip()


class ResourceLimits(BaseModel):
    """CPU / memory resource limits for a container.

    Empty strings mean "unset" and are flagged MEDIUM by security review.
    """

    cpu: str = ""
    memory: str = ""


class ContainerSpec(BaseModel):
    """Specification for a single container in a plan."""

    name: str
    image: str
    command: List[str] = Field(default_factory=list)
    user: str = ""
    read_only_rootfs: bool = False
    privileged: bool = False
    env: Dict[str, str] = Field(default_factory=dict)
    mounts: List[str] = Field(default_factory=list)
    cap_drop: List[str] = Field(default_factory=list)
    cap_add: List[str] = Field(default_factory=list)
    # Namespace sharing: "host" shares the host's namespace (insecure), "" is isolated.
    network: str = ""
    pid: str = ""
    ipc: str = ""
    resources: ResourceLimits = Field(default_factory=ResourceLimits)


class RollbackDescriptor(BaseModel):
    """Describes how to roll a plan back to its predecessor."""

    previous_version: int
    previous_plan_hash: str
    strategy: str = "redeploy-previous"


class ContainerPlan(BaseModel):
    """A versioned, auditable container deployment plan."""

    plan_id: str
    version: int = 1
    specs: List[ContainerSpec] = Field(default_factory=list)
    rollback: Optional[RollbackDescriptor] = None
    plan_hash: str = ""
    schema_version: str = SCHEMA_VERSION
    created_by: str = "container_plan"


class Finding(BaseModel):
    """A single static security-review finding."""

    severity: str
    code: str
    message: str
    spec_name: str


def _spec_hash_payload(spec: ContainerSpec) -> Dict[str, Any]:
    """Return a deterministic, plan_hash-independent payload for one spec."""
    return {
        "name": spec.name,
        "image": spec.image,
        "command": list(spec.command),
        "user": spec.user,
        "read_only_rootfs": spec.read_only_rootfs,
        "privileged": spec.privileged,
        "env": dict(spec.env),
        "mounts": list(spec.mounts),
        "cap_drop": list(spec.cap_drop),
        "cap_add": list(spec.cap_add),
        "network": spec.network,
        "pid": spec.pid,
        "ipc": spec.ipc,
        "resources": {
            "cpu": spec.resources.cpu,
            "memory": spec.resources.memory,
        },
    }


def compute_plan_hash(plan: ContainerPlan) -> str:
    """Deterministically hash a plan's specs (excludes the plan_hash field).

    The hash covers plan_id, version, and the ordered list of spec payloads.
    No wall-clock or environment-dependent values are included, so identical
    specs always yield an identical hash.

    Args:
        plan: The container plan to hash.

    Returns:
        A 64-char SHA256 hex digest.
    """
    payload = {
        "plan_id": plan.plan_id,
        "version": plan.version,
        "specs": [_spec_hash_payload(s) for s in plan.specs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_container_plan(
    plan_id: str,
    specs_config: List[Dict[str, Any]],
    previous: Optional[ContainerPlan] = None,
) -> ContainerPlan:
    """Build a HARDENED container plan from a list of spec configs.

    Hardening defaults applied when a field is not explicitly supplied:
      - user defaults to "nonroot" (never "" or "root")
      - read_only_rootfs defaults to True
      - privileged defaults to False
      - cap_drop defaults to ["ALL"]
      - resources default to cpu="1.0", memory="512Mi"

    If ``previous`` is given, the new plan's version is bumped to
    ``previous.version + 1`` and a RollbackDescriptor pointing at the previous
    version/plan_hash is attached.

    Args:
        plan_id: Stable identifier for the plan.
        specs_config: List of per-spec config dicts (must include "name" and
            "image"; other fields override the hardened defaults).
        previous: Optional prior plan to roll back to.

    Returns:
        A fully populated ContainerPlan with plan_hash set.
    """
    specs: List[ContainerSpec] = []
    for cfg in specs_config:
        merged: Dict[str, Any] = {
            "user": "nonroot",
            "read_only_rootfs": True,
            "privileged": False,
            "cap_drop": ["ALL"],
            "resources": ResourceLimits(cpu="1.0", memory="512Mi"),
        }
        merged.update(cfg)
        specs.append(ContainerSpec(**merged))

    version = 1
    rollback: Optional[RollbackDescriptor] = None
    if previous is not None:
        version = previous.version + 1
        rollback = RollbackDescriptor(
            previous_version=previous.version,
            previous_plan_hash=previous.plan_hash,
        )

    plan = ContainerPlan(
        plan_id=plan_id,
        version=version,
        specs=specs,
        rollback=rollback,
    )
    plan.plan_hash = compute_plan_hash(plan)
    return plan


def security_review(plan: ContainerPlan) -> List[Finding]:
    """Statically review a plan and return all security findings.

    Per spec, the following issues are flagged (an empty result == clean):
      - privileged True                              -> CRITICAL
      - user in ("", "root")                         -> HIGH
      - image lacking "@sha256:" digest pin          -> HIGH
      - plaintext secret env value                   -> HIGH
      - resources.cpu or .memory empty               -> MEDIUM
      - read_only_rootfs False                       -> LOW
      - cap_drop not containing "ALL"                -> LOW

    A secret env value is one whose key matches PASSWORD|SECRET|TOKEN|KEY, has a
    non-empty value, and is not a "${...}" indirect reference.

    Args:
        plan: The plan to inspect.

    Returns:
        A list of Finding objects (possibly empty).
    """
    findings: List[Finding] = []
    for spec in plan.specs:
        if spec.privileged:
            findings.append(
                Finding(
                    severity="CRITICAL",
                    code="PRIVILEGED_CONTAINER",
                    message="Container runs in privileged mode.",
                    spec_name=spec.name,
                )
            )
        if spec.user in ("", "root"):
            findings.append(
                Finding(
                    severity="HIGH",
                    code="ROOT_USER",
                    message="Container runs as root (or unspecified user).",
                    spec_name=spec.name,
                )
            )
        if not _DIGEST_PIN_RE.search(spec.image):
            findings.append(
                Finding(
                    severity="HIGH",
                    code="UNPINNED_IMAGE",
                    message="Image is not pinned by a full @sha256:<64-hex> digest "
                    "(a mutable tag such as :latest or a malformed digest is not a pin).",
                    spec_name=spec.name,
                )
            )
        for key, value in spec.env.items():
            if (
                _SECRET_KEY_RE.search(key)
                and value
                and not _REFERENCE_RE.match(value)
            ):
                findings.append(
                    Finding(
                        severity="HIGH",
                        code="PLAINTEXT_SECRET",
                        message=f"Env var '{key}' looks like a plaintext secret.",
                        spec_name=spec.name,
                    )
                )
        if not spec.resources.cpu or not spec.resources.memory:
            findings.append(
                Finding(
                    severity="MEDIUM",
                    code="UNBOUNDED_RESOURCES",
                    message="Resource limits (cpu/memory) are not fully set.",
                    spec_name=spec.name,
                )
            )
        if not spec.read_only_rootfs:
            findings.append(
                Finding(
                    severity="LOW",
                    code="WRITABLE_ROOTFS",
                    message="Root filesystem is not read-only.",
                    spec_name=spec.name,
                )
            )
        if "ALL" not in spec.cap_drop:
            findings.append(
                Finding(
                    severity="LOW",
                    code="CAPS_NOT_DROPPED",
                    message="cap_drop does not drop ALL capabilities.",
                    spec_name=spec.name,
                )
            )
        # Host-path mounts that enable trivial container escape.
        for mount in spec.mounts:
            host_path = _mount_host_path(mount)
            if host_path in _SENSITIVE_HOST_MOUNTS:
                findings.append(
                    Finding(
                        severity="CRITICAL",
                        code="SENSITIVE_HOST_MOUNT",
                        message=f"Mounts sensitive host path '{host_path}' "
                        "(enables host escape).",
                        spec_name=spec.name,
                    )
                )
        # Shared host namespaces break container isolation.
        for ns_field, ns_value in (("network", spec.network), ("pid", spec.pid), ("ipc", spec.ipc)):
            if ns_value == "host":
                findings.append(
                    Finding(
                        severity="HIGH",
                        code=f"HOST_{ns_field.upper()}",
                        message=f"Container shares the host {ns_field} namespace.",
                        spec_name=spec.name,
                    )
                )
        # Added capabilities that materially weaken isolation.
        dangerous_added = sorted(c for c in spec.cap_add if c.upper() in _DANGEROUS_CAPS)
        if dangerous_added:
            findings.append(
                Finding(
                    severity="HIGH",
                    code="DANGEROUS_CAP_ADD",
                    message=f"Adds dangerous capabilities: {', '.join(dangerous_added)}.",
                    spec_name=spec.name,
                )
            )
    return findings


def serialize_plan(plan: ContainerPlan) -> str:
    """Serialize a plan to deterministic, sorted-key JSON.

    Args:
        plan: The plan to serialize.

    Returns:
        A canonical JSON string (sorted keys, 2-space indent).
    """
    return json.dumps(plan.model_dump(), sort_keys=True, indent=2)


def plan_to_compose(plan: ContainerPlan) -> Dict[str, Any]:
    """Render a plan as a pure compose-shaped dict (no file I/O).

    This produces DATA only — it never writes, deploys, or executes anything.

    Args:
        plan: The plan to render.

    Returns:
        A dict with a "services" mapping keyed by spec name.
    """
    services: Dict[str, Any] = {}
    for spec in plan.specs:
        service: Dict[str, Any] = {
            "image": spec.image,
            "read_only": spec.read_only_rootfs,
            "privileged": spec.privileged,
        }
        if spec.command:
            service["command"] = list(spec.command)
        if spec.user:
            service["user"] = spec.user
        if spec.env:
            service["environment"] = dict(spec.env)
        if spec.mounts:
            service["volumes"] = list(spec.mounts)
        if spec.cap_drop:
            service["cap_drop"] = list(spec.cap_drop)
        limits: Dict[str, str] = {}
        if spec.resources.cpu:
            limits["cpus"] = spec.resources.cpu
        if spec.resources.memory:
            limits["memory"] = spec.resources.memory
        if limits:
            service["deploy"] = {"resources": {"limits": limits}}
        services[spec.name] = service
    return {
        "version": "3.8",
        "x-plan-id": plan.plan_id,
        "x-plan-version": plan.version,
        "x-plan-hash": plan.plan_hash,
        "services": services,
    }


def plan_provenance(plan: ContainerPlan) -> Dict[str, Any]:
    """Return non-hashed provenance metadata for a plan.

    The wall-clock timestamp here is provenance only and never feeds any hash.

    Args:
        plan: The plan to describe.

    Returns:
        A provenance metadata dict.
    """
    return {
        "plan_id": plan.plan_id,
        "version": plan.version,
        "plan_hash": plan.plan_hash,
        "schema_version": plan.schema_version,
        "created_by": plan.created_by,
        "described_at": datetime.now().isoformat(),
    }
