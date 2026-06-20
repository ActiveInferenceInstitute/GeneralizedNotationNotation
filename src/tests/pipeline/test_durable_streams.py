#!/usr/bin/env python3
"""Tests for pipeline/durable_streams.py — no mocks, real numpy + tmp_path."""

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.durable_streams import (  # noqa: E402
    ExecutionTrace,
    StreamKind,
    StreamManifest,
    read_stream_manifest,
    read_trace,
    replay_trace,
    trace_integrity,
    validate_stream_manifest,
    verify_replay,
    write_stream_manifest,
    write_trace,
)


class TestStreamManifestRoundTrip:
    def test_from_array_roundtrip(self, tmp_path: Path) -> Any:
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        manifest = StreamManifest.from_array("arr-stream", arr, source="logical")
        assert manifest.kind == StreamKind.ARRAY
        assert manifest.shape == [3, 4]
        assert manifest.n_elements == 12
        assert manifest.dtype == "float64"
        assert len(manifest.checksum) == 64

        path = tmp_path / "manifest.json"
        write_stream_manifest(manifest, path)
        loaded = read_stream_manifest(path)
        assert loaded == manifest

    def test_from_file_roundtrip(self, tmp_path: Path) -> Any:
        data_file = tmp_path / "obs.bin"
        data_file.write_bytes(b"durable observation payload")
        manifest = StreamManifest.from_file("file-stream", data_file, source="obs.bin")
        assert manifest.kind == StreamKind.FILE
        assert manifest.source == "obs.bin"
        assert manifest.n_elements == len(b"durable observation payload")

        path = tmp_path / "file_manifest.json"
        write_stream_manifest(manifest, path)
        loaded = read_stream_manifest(path)
        assert loaded == manifest

    def test_validate_file_manifest_ok(self, tmp_path: Path) -> Any:
        data_file = tmp_path / "obs.bin"
        data_file.write_bytes(b"\x00\x01\x02\x03 stream bytes")
        manifest = StreamManifest.from_file("s", data_file, source="obs.bin")
        assert validate_stream_manifest(manifest, tmp_path) == []

    def test_validate_array_manifest_ok(self, tmp_path: Path) -> Any:
        arr = np.ones((2, 5), dtype=np.int32)
        manifest = StreamManifest.from_array("s", arr)
        assert validate_stream_manifest(manifest, tmp_path) == []


class TestStreamManifestNegativeControls:
    def test_corrupted_file_detected(self, tmp_path: Path) -> Any:
        data_file = tmp_path / "obs.bin"
        data_file.write_bytes(b"original bytes")
        manifest = StreamManifest.from_file("s", data_file, source="obs.bin")
        assert validate_stream_manifest(manifest, tmp_path) == []

        # Tamper with the bytes AFTER the manifest was created.
        data_file.write_bytes(b"tampered bytes!")
        problems = validate_stream_manifest(manifest, tmp_path)
        assert problems
        assert any("checksum mismatch" in p for p in problems)

    def test_missing_file_detected(self, tmp_path: Path) -> Any:
        manifest = StreamManifest(
            stream_id="s",
            kind=StreamKind.FILE,
            dtype="uint8",
            source="absent.bin",
            checksum="0" * 64,
        )
        problems = validate_stream_manifest(manifest, tmp_path)
        assert any("does not exist" in p for p in problems)

    def test_array_element_count_mismatch_detected(self, tmp_path: Path) -> Any:
        manifest = StreamManifest(
            stream_id="s",
            kind=StreamKind.ARRAY,
            dtype="float64",
            shape=[3, 4],
            n_elements=99,  # wrong: prod(shape) == 12
        )
        problems = validate_stream_manifest(manifest, tmp_path)
        assert any("n_elements" in p for p in problems)

    def test_empty_dtype_detected(self, tmp_path: Path) -> Any:
        manifest = StreamManifest(
            stream_id="s",
            kind=StreamKind.ARRAY,
            dtype="",
            shape=[2],
            n_elements=2,
        )
        problems = validate_stream_manifest(manifest, tmp_path)
        assert any("dtype" in p for p in problems)


def _build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(trace_id="t-1")
    trace = trace.append_event("step_a", "observe", b"payload-0", payload_ref="p0")
    trace = trace.append_event("step_b", "validate", b"payload-1", payload_ref="p1")
    trace = trace.append_event("step_c", "emit", b"payload-2", payload_ref="p2")
    return trace


class TestExecutionTrace:
    def test_append_event_is_immutable_and_monotonic(self) -> Any:
        empty = ExecutionTrace(trace_id="t")
        one = empty.append_event("s", "a", b"x")
        # Original is unchanged.
        assert empty.events == []
        assert len(one.events) == 1
        assert one.events[0].seq == 0
        two = one.append_event("s", "a", b"y")
        assert [e.seq for e in two.events] == [0, 1]
        assert two.events[0].payload_checksum != two.events[1].payload_checksum

    def test_trace_integrity_clean(self) -> Any:
        trace = _build_trace()
        assert trace_integrity(trace) == []

    def test_replay_stable_across_two_builds(self, tmp_path: Path) -> Any:
        d1 = replay_trace(_build_trace())
        d2 = replay_trace(_build_trace())
        assert d1 == d2
        assert len(d1) == 64

        # Round-trip the trace through disk and confirm digest stability.
        trace = _build_trace()
        path = tmp_path / "trace.json"
        write_trace(trace, path)
        loaded = read_trace(path)
        assert loaded == trace
        assert replay_trace(loaded) == d1

    def test_verify_replay_true_and_false(self) -> Any:
        trace = _build_trace()
        digest = replay_trace(trace)
        assert verify_replay(trace, digest) is True
        assert verify_replay(trace, "deadbeef" * 8) is False


class TestTraceIntegrityNegativeControls:
    def test_seq_gap_detected(self) -> Any:
        from pipeline.durable_streams import TraceEvent

        trace = ExecutionTrace(
            trace_id="t",
            events=[
                TraceEvent(seq=0, step="a", action="x", payload_checksum="aa"),
                TraceEvent(seq=2, step="b", action="y", payload_checksum="bb"),
            ],
        )
        problems = trace_integrity(trace)
        assert any("gap" in p for p in problems)

    def test_duplicate_seq_detected(self) -> Any:
        from pipeline.durable_streams import TraceEvent

        trace = ExecutionTrace(
            trace_id="t",
            events=[
                TraceEvent(seq=0, step="a", action="x", payload_checksum="aa"),
                TraceEvent(seq=0, step="b", action="y", payload_checksum="bb"),
            ],
        )
        problems = trace_integrity(trace)
        assert any("duplicate seq" in p for p in problems)

    def test_does_not_start_at_zero_detected(self) -> Any:
        from pipeline.durable_streams import TraceEvent

        trace = ExecutionTrace(
            trace_id="t",
            events=[
                TraceEvent(seq=1, step="a", action="x", payload_checksum="aa"),
            ],
        )
        problems = trace_integrity(trace)
        assert any("start at 0" in p for p in problems)

    def test_missing_checksum_detected(self) -> Any:
        from pipeline.durable_streams import TraceEvent

        trace = ExecutionTrace(
            trace_id="t",
            events=[
                TraceEvent(seq=0, step="a", action="x", payload_checksum=""),
            ],
        )
        problems = trace_integrity(trace)
        assert any("missing payload_checksum" in p for p in problems)


def test_replay_no_field_framing_collision() -> None:
    """NEGATIVE: distinct field splits must NOT share a replay digest (length-prefix framing)."""
    from pipeline.durable_streams import ExecutionTrace, TraceEvent, replay_trace

    a = ExecutionTrace(trace_id="a", events=[TraceEvent(seq=0, step="a|b", action="c", payload_checksum="h")])
    b = ExecutionTrace(trace_id="b", events=[TraceEvent(seq=0, step="a", action="b|c", payload_checksum="h")])
    assert replay_trace(a) != replay_trace(b)

    # A separator-bearing payload_checksum cannot fake a second event either.
    one = ExecutionTrace(trace_id="o", events=[TraceEvent(seq=0, step="s1", action="a1", payload_checksum="h1\x1e1|s2|a2|h2")])
    two = ExecutionTrace(
        trace_id="t",
        events=[
            TraceEvent(seq=0, step="s1", action="a1", payload_checksum="h1"),
            TraceEvent(seq=1, step="s2", action="a2", payload_checksum="h2"),
        ],
    )
    assert replay_trace(one) != replay_trace(two)


def test_array_checksum_is_byte_order_canonical() -> None:
    """Cross-endian determinism: the same logical array hashes identically regardless of byte order."""
    import numpy as np

    from pipeline.durable_streams import StreamManifest

    native = np.array([1, 2, 3, 4], dtype="<i4")
    swapped = native.astype(">i4")  # same logical values, big-endian bytes
    m_native = StreamManifest.from_array("s", native)
    m_swapped = StreamManifest.from_array("s", swapped)
    assert m_native.checksum == m_swapped.checksum
    assert m_native.dtype == m_swapped.dtype == "int32"


def test_array_manifest_validation_has_teeth(tmp_path) -> None:
    """NEGATIVE: a bogus checksum or unparseable dtype on an ARRAY manifest is rejected."""
    from pipeline.durable_streams import (
        StreamKind,
        StreamManifest,
        validate_stream_manifest,
    )

    bad = StreamManifest(
        stream_id="s", kind=StreamKind.ARRAY, dtype="not-a-dtype", shape=[2, 2],
        source="x", chunk_size=1, n_elements=4, checksum="totally-wrong-checksum",
    )
    problems = validate_stream_manifest(bad, tmp_path)
    assert any("sha256 hex" in p for p in problems)
    assert any("valid numpy dtype" in p for p in problems)
