#!/usr/bin/env python3
"""Durable observation streams — replayable file/array stream manifests + traces.

This is the v3.0.0 foundation laid down BEFORE any live sensor/device stream.
Every public function here is PURE: it generates, validates, serializes, and
replays DATA. Nothing in this module executes containers, opens sockets, or
touches real devices. All file IO is local and atomic (tmp file + os.replace).

Public surface:
  - StreamKind (Enum)
  - StreamManifest (BaseModel) with from_array / from_file classmethods
  - TraceEvent (BaseModel)
  - ExecutionTrace (BaseModel) with append_event
  - trace_integrity(trace) -> list[str]
  - write_stream_manifest / read_stream_manifest
  - write_trace / read_trace
  - validate_stream_manifest(manifest, base_dir) -> list[str]
  - replay_trace(trace) -> str / verify_replay(trace, expected_digest) -> bool
"""

import hashlib
import os
import re
import tempfile
from enum import Enum
from math import prod
from pathlib import Path
from typing import List, Union

import numpy as np
from pydantic import BaseModel, Field

# A well-formed sha256 digest is exactly 64 lowercase hex characters.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class StreamKind(str, Enum):
    """Kind of durable observation stream."""

    FILE = "FILE"
    ARRAY = "ARRAY"


def _sha256_hex(data: bytes) -> str:
    """Return the lowercase hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` (tmp file in same dir + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, str(path))
    except BaseException:
        # Clean up the temp file if anything went wrong before the replace.
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class StreamManifest(BaseModel):
    """Manifest describing one durable observation stream.

    A manifest is a content-addressed description of a stream's payload. For
    FILE streams ``source`` is a path relative to a base directory; for ARRAY
    streams ``source`` is a logical name. ``checksum`` is the sha256 hex digest
    of the canonical content bytes.
    """

    stream_id: str
    kind: StreamKind
    dtype: str
    shape: List[int] = Field(default_factory=list)
    source: str = ""
    chunk_size: int = 1024
    n_elements: int = 0
    checksum: str = ""
    schema_version: str = "1.0"
    created_by: str = ""

    @classmethod
    def from_array(
        cls,
        stream_id: str,
        array: np.ndarray,
        source: str = "",
        chunk_size: int = 1024,
        created_by: str = "",
    ) -> "StreamManifest":
        """Build an ARRAY manifest from a real numpy array.

        shape / dtype / n_elements / checksum are computed deterministically
        from the array's C-contiguous byte representation.

        Args:
            stream_id: Stable identifier for this stream.
            array: A numpy array supplying the content bytes and metadata.
            source: Logical name for the array stream.
            chunk_size: Logical chunk size recorded in the manifest.
            created_by: Optional provenance label.

        Returns:
            A populated :class:`StreamManifest`.
        """
        arr = np.ascontiguousarray(array)
        # Canonicalize to little-endian before hashing so the checksum is identical
        # across architectures (a big-endian host would otherwise hash the same
        # logical array to a different digest). dtype.name is byte-order independent.
        canonical = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
        return cls(
            stream_id=stream_id,
            kind=StreamKind.ARRAY,
            dtype=arr.dtype.name,
            shape=list(arr.shape),
            source=source,
            chunk_size=chunk_size,
            n_elements=int(arr.size),
            checksum=_sha256_hex(canonical.tobytes()),
            created_by=created_by,
        )

    @classmethod
    def from_file(
        cls,
        stream_id: str,
        path: Union[str, Path],
        source: str = "",
        chunk_size: int = 1024,
        dtype: str = "uint8",
        created_by: str = "",
    ) -> "StreamManifest":
        """Build a FILE manifest by checksumming the file's bytes.

        Args:
            stream_id: Stable identifier for this stream.
            path: Path to the file whose bytes are checksummed.
            source: Relative path recorded in the manifest; defaults to the
                file name when empty.
            chunk_size: Logical chunk size recorded in the manifest.
            dtype: Logical dtype label for the byte stream.
            created_by: Optional provenance label.

        Returns:
            A populated :class:`StreamManifest`.
        """
        p = Path(path)
        content = p.read_bytes()
        return cls(
            stream_id=stream_id,
            kind=StreamKind.FILE,
            dtype=dtype,
            shape=[len(content)],
            source=source or p.name,
            chunk_size=chunk_size,
            n_elements=len(content),
            checksum=_sha256_hex(content),
            created_by=created_by,
        )


class TraceEvent(BaseModel):
    """A single ordered event within an execution trace."""

    seq: int
    step: str
    action: str
    payload_ref: str = ""
    payload_checksum: str = ""


class ExecutionTrace(BaseModel):
    """An ordered, replayable sequence of trace events."""

    trace_id: str
    events: List[TraceEvent] = Field(default_factory=list)
    schema_version: str = "1.0"
    created_by: str = ""

    def append_event(
        self,
        step: str,
        action: str,
        payload_bytes: bytes,
        payload_ref: str = "",
    ) -> "ExecutionTrace":
        """Return a NEW trace with one event appended.

        The new event's ``seq`` is the next monotonic integer (starting at 0)
        and its ``payload_checksum`` is the sha256 of ``payload_bytes``. The
        receiver is not mutated.

        Args:
            step: The pipeline step name producing the event.
            action: The action performed.
            payload_bytes: Raw bytes whose checksum is recorded.
            payload_ref: Optional reference (e.g. relative path) to the payload.

        Returns:
            A new :class:`ExecutionTrace` with the appended event.
        """
        next_seq = len(self.events)
        new_event = TraceEvent(
            seq=next_seq,
            step=step,
            action=action,
            payload_ref=payload_ref,
            payload_checksum=_sha256_hex(payload_bytes),
        )
        return ExecutionTrace(
            trace_id=self.trace_id,
            events=[*self.events, new_event],
            schema_version=self.schema_version,
            created_by=self.created_by,
        )


def trace_integrity(trace: ExecutionTrace) -> List[str]:
    """Check structural integrity of a trace's event sequence.

    Verifies that ``seq`` starts at 0, is contiguous and strictly increasing,
    has no duplicates, and that every event carries a checksum.

    Args:
        trace: The execution trace to inspect.

    Returns:
        A list of human-readable problems. Empty means the trace is sound.
    """
    problems: List[str] = []
    events = trace.events

    if not events:
        return problems

    seqs = [e.seq for e in events]

    # Duplicate detection.
    seen: set[int] = set()
    for s in seqs:
        if s in seen:
            problems.append(f"duplicate seq {s}")
        seen.add(s)

    # Must start at 0.
    if seqs[0] != 0:
        problems.append(f"seq must start at 0, got {seqs[0]}")

    # Contiguous and strictly increasing.
    for prev, curr in zip(seqs, seqs[1:]):
        if curr <= prev:
            problems.append(f"seq not strictly increasing: {prev} -> {curr}")
        elif curr != prev + 1:
            problems.append(f"seq gap: {prev} -> {curr}")

    # Every event must have a checksum.
    for e in events:
        if not e.payload_checksum:
            problems.append(f"event seq {e.seq} missing payload_checksum")

    return problems


def write_stream_manifest(manifest: StreamManifest, path: Union[str, Path]) -> Path:
    """Atomically serialize a stream manifest to JSON.

    Args:
        manifest: The manifest to write.
        path: Destination path.

    Returns:
        The destination :class:`Path`.
    """
    out = Path(path)
    _atomic_write_text(out, manifest.model_dump_json(indent=2))
    return out


def read_stream_manifest(path: Union[str, Path]) -> StreamManifest:
    """Read and validate a stream manifest from JSON.

    Args:
        path: Source path.

    Returns:
        The parsed :class:`StreamManifest`.
    """
    text = Path(path).read_text(encoding="utf-8")
    return StreamManifest.model_validate_json(text)


def write_trace(trace: ExecutionTrace, path: Union[str, Path]) -> Path:
    """Atomically serialize an execution trace to JSON.

    Args:
        trace: The trace to write.
        path: Destination path.

    Returns:
        The destination :class:`Path`.
    """
    out = Path(path)
    _atomic_write_text(out, trace.model_dump_json(indent=2))
    return out


def read_trace(path: Union[str, Path]) -> ExecutionTrace:
    """Read and validate an execution trace from JSON.

    Args:
        path: Source path.

    Returns:
        The parsed :class:`ExecutionTrace`.
    """
    text = Path(path).read_text(encoding="utf-8")
    return ExecutionTrace.model_validate_json(text)


def validate_stream_manifest(
    manifest: StreamManifest, base_dir: Union[str, Path]
) -> List[str]:
    """Validate a stream manifest against its backing content.

    For FILE streams the checksum is recomputed from ``base_dir/source`` and
    compared. For ARRAY streams the shape/element-count and dtype are checked
    for internal consistency.

    Args:
        manifest: The manifest to validate.
        base_dir: Directory that FILE sources are resolved against.

    Returns:
        A list of problems. Empty means the manifest is valid.
    """
    problems: List[str] = []
    base = Path(base_dir)

    if manifest.kind == StreamKind.FILE:
        target = base / manifest.source
        if not target.exists():
            problems.append(f"source file does not exist: {target}")
            return problems
        actual = _sha256_hex(target.read_bytes())
        if actual != manifest.checksum:
            problems.append(
                f"checksum mismatch for {manifest.source}: "
                f"expected {manifest.checksum}, got {actual}"
            )
    elif manifest.kind == StreamKind.ARRAY:
        expected_elements = prod(manifest.shape) if manifest.shape else 0
        if manifest.n_elements != expected_elements:
            problems.append(
                f"n_elements {manifest.n_elements} != prod(shape) "
                f"{expected_elements}"
            )
        # ARRAY streams carry no payload to re-hash, but the manifest's own fields
        # must still be well-formed: a 64-hex checksum and a parseable dtype. Without
        # these the validator would pass a manifest with a bogus checksum/dtype.
        if not _SHA256_HEX_RE.match(manifest.checksum):
            problems.append(f"checksum is not a 64-char sha256 hex digest: {manifest.checksum!r}")
        if not manifest.dtype:
            problems.append("dtype is empty")
        else:
            try:
                np.dtype(manifest.dtype)
            except TypeError:
                problems.append(f"dtype is not a valid numpy dtype: {manifest.dtype!r}")
    else:  # pragma: no cover - exhaustive guard
        problems.append(f"unknown stream kind: {manifest.kind}")

    return problems


def replay_trace(trace: ExecutionTrace) -> str:
    """Deterministically fold a trace's ordered events into one digest.

    The events are processed in their stored order and a sha256 is accumulated
    over each field with explicit length-prefix framing, so no delimiter that
    appears inside a field value can make two distinct traces collide (a naive
    ``step|action`` join would let ``"a|b","c"`` and ``"a","b|c"`` share a digest).

    Args:
        trace: The trace to digest.

    Returns:
        The hex sha256 digest of the ordered event stream.
    """
    hasher = hashlib.sha256()

    def _feed(field: bytes) -> None:
        # length-prefix every field so its bytes can never be confused with a
        # field boundary, regardless of what characters the value contains.
        hasher.update(len(field).to_bytes(8, "big"))
        hasher.update(field)

    for e in trace.events:
        _feed(str(e.seq).encode("utf-8"))
        _feed(e.step.encode("utf-8"))
        _feed(e.action.encode("utf-8"))
        _feed(e.payload_checksum.encode("utf-8"))
    return hasher.hexdigest()


def verify_replay(trace: ExecutionTrace, expected_digest: str) -> bool:
    """Return True iff replaying ``trace`` reproduces ``expected_digest``.

    Args:
        trace: The trace to replay.
        expected_digest: The digest to compare against.

    Returns:
        Whether the recomputed digest matches.
    """
    return replay_trace(trace) == expected_digest
