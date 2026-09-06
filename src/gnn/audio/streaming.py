"""Device-free audio telemetry contracts for Step 12 stream artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class AudioTelemetryFrame:
    """One normalized Step 12 telemetry sample available to audio generation."""

    t: int
    free_energy: float | None = None
    belief: List[float] = field(default_factory=list)
    action: int | None = None


@dataclass(frozen=True)
class AudioStreamChunk:
    """A deterministic audio-control chunk derived from telemetry frames."""

    index: int
    frame_start: int
    frame_count: int
    amplitude: float
    frequency_hz: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def frames_from_execution_trace(trace: Dict[str, Any]) -> List[AudioTelemetryFrame]:
    """Normalize Step 12 simulation outputs into telemetry frames."""
    trace = _unwrap_trace_payload(trace)
    free_energy = _first_sequence(
        trace,
        "free_energy",
        "expected_free_energy",
        "free_energy_trace",
        "expected_free_energy_trace",
        "free_energy_history",
    )
    beliefs = _first_sequence(trace, "beliefs", "belief_trace", "belief_history")
    actions = _first_sequence(
        trace, "actions", "action_trace", "action_history", "selected_actions"
    )
    n = max(len(free_energy), len(beliefs), len(actions))
    frames: List[AudioTelemetryFrame] = []
    for index in range(n):
        belief = _coerce_belief(beliefs[index]) if index < len(beliefs) else []
        action = _coerce_action(actions[index]) if index < len(actions) else None
        frames.append(
            AudioTelemetryFrame(
                t=index,
                free_energy=_coerce_scalar(free_energy[index])
                if index < len(free_energy)
                else None,
                belief=[float(value) for value in belief],
                action=int(action) if action is not None else None,
            )
        )
    return frames


def _unwrap_trace_payload(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return the telemetry-bearing object from common Step 12 result envelopes."""
    for key in ("simulation_data", "telemetry", "execution_trace", "trace"):
        nested = trace.get(key)
        if isinstance(nested, dict):
            return nested
    traces = trace.get("traces")
    if isinstance(traces, list) and traces and isinstance(traces[0], dict):
        return traces[0]
    return trace


def _first_sequence(trace: Dict[str, Any], *keys: str) -> List[Any]:
    """Return the first list value under ``keys``."""
    for key in keys:
        value = trace.get(key)
        if isinstance(value, list):
            return value
    return []


def _coerce_belief(value: Any) -> List[float]:
    """Coerce Step 12 belief shapes into a numeric vector."""
    if isinstance(value, dict):
        value = value.get("belief") or value.get("state_beliefs") or value.get("values")
    if isinstance(value, list):
        return [float(item) for item in value if isinstance(item, (int, float))]
    return []


def _coerce_action(value: Any) -> int | None:
    """Coerce Step 12 action shapes into an integer action id."""
    if isinstance(value, dict):
        for key in ("action", "selected_action", "id"):
            if key in value:
                value = value[key]
                break
        else:
            value = None
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, (int, float))), None)
    if value is None:
        return None
    return int(value)


def _coerce_scalar(value: Any) -> float | None:
    """Coerce scalar or vector telemetry values into one numeric sample."""
    if isinstance(value, dict):
        for key in ("value", "free_energy", "expected_free_energy"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, list):
        numeric = [float(item) for item in value if isinstance(item, (int, float))]
        return sum(numeric) / len(numeric) if numeric else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def chunks_from_frames(
    frames: Iterable[AudioTelemetryFrame], *, chunk_size: int = 32
) -> List[AudioStreamChunk]:
    """Create deterministic chunks without requiring live audio devices."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    frame_list = list(frames)
    chunks: List[AudioStreamChunk] = []
    for chunk_index, start in enumerate(range(0, len(frame_list), chunk_size)):
        group = frame_list[start : start + chunk_size]
        free_energy_values = [
            frame.free_energy for frame in group if frame.free_energy is not None
        ]
        mean_fe = (
            sum(free_energy_values) / len(free_energy_values)
            if free_energy_values
            else 0.0
        )
        last_action = next(
            (frame.action for frame in reversed(group) if frame.action is not None),
            None,
        )
        belief_confidence = max(
            (max(frame.belief) for frame in group if frame.belief), default=0.0
        )
        chunks.append(
            AudioStreamChunk(
                index=chunk_index,
                frame_start=start,
                frame_count=len(group),
                amplitude=max(0.0, min(1.0, belief_confidence)),
                frequency_hz=220.0
                + abs(mean_fe) * 30.0
                + float(last_action or 0) * 15.0,
                metadata={"mean_free_energy": mean_fe, "last_action": last_action},
            )
        )
    return chunks


def write_stream_summary(
    chunks: Iterable[AudioStreamChunk], output_path: Path
) -> Dict[str, Any]:
    """Persist stream chunk metadata for downstream SAPF/Pedalboard consumers."""
    chunk_list = list(chunks)
    if not chunk_list:
        payload = {
            "schema": "gnn_audio_stream_chunks_v1",
            "status": "no_frames",
            "streaming_safe": False,
            "frame_count": 0,
            "duration_frames": 0,
            "chunk_count": 0,
            "chunks": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    payload = {
        "schema": "gnn_audio_stream_chunks_v1",
        "status": "ready",
        "streaming_safe": True,
        "frame_count": sum(chunk.frame_count for chunk in chunk_list),
        "duration_frames": sum(chunk.frame_count for chunk in chunk_list),
        "chunk_count": len(chunk_list),
        "chunks": [asdict(chunk) for chunk in chunk_list],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
