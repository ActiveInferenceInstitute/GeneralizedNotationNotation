from __future__ import annotations

import json
from pathlib import Path

from audio.processor import (
    _process_audio_streaming,
    _resolve_execution_summary_artifact,
)
from audio.streaming import (
    chunks_from_frames,
    frames_from_execution_trace,
    write_stream_summary,
)


def test_audio_telemetry_frames_from_execution_trace() -> None:
    frames = frames_from_execution_trace(
        {
            "free_energy": [1.0, 0.5],
            "beliefs": [[0.2, 0.8], [0.7, 0.3]],
            "actions": [1, 0],
        }
    )
    assert len(frames) == 2
    assert frames[0].belief == [0.2, 0.8]
    assert frames[1].action == 0


def test_audio_stream_chunks_are_device_free(tmp_path: Path) -> None:
    frames = frames_from_execution_trace(
        {
            "expected_free_energy": [1.0, 2.0, 3.0],
            "beliefs": [[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]],
            "actions": [0, 1, 1],
        }
    )
    chunks = chunks_from_frames(frames, chunk_size=2)
    assert [chunk.frame_count for chunk in chunks] == [2, 1]
    payload = write_stream_summary(chunks, tmp_path / "chunks.json")
    assert payload["schema"] == "gnn_audio_stream_chunks_v1"
    assert payload["streaming_safe"] is True
    assert payload["frame_count"] == 3
    assert (tmp_path / "chunks.json").exists()


def test_audio_stream_summary_empty_chunks_is_not_success_like(tmp_path: Path) -> None:
    payload = write_stream_summary([], tmp_path / "empty_chunks.json")

    assert payload["status"] == "no_frames"
    assert payload["streaming_safe"] is False
    assert payload["chunk_count"] == 0
    assert (tmp_path / "empty_chunks.json").exists()


def test_audio_frames_accept_step12_simulation_data_aliases() -> None:
    frames = frames_from_execution_trace(
        {
            "simulation_data": {
                "free_energy_trace": [2.0, 1.0],
                "belief_history": [
                    {"state_beliefs": [0.4, 0.6]},
                    {"state_beliefs": [0.8, 0.2]},
                ],
                "action_history": [{"selected_action": 1}, {"selected_action": 0}],
            }
        }
    )

    assert [frame.free_energy for frame in frames] == [2.0, 1.0]
    assert frames[0].belief == [0.4, 0.6]
    assert frames[1].action == 0


def test_audio_frames_accept_vector_free_energy_and_action_samples() -> None:
    frames = frames_from_execution_trace(
        {
            "expected_free_energy": [[1.0, 3.0], [2.0, 4.0]],
            "beliefs": [[0.2, 0.8], [0.6, 0.4]],
            "actions": [[1, 0], [0, 1]],
        }
    )

    assert [frame.free_energy for frame in frames] == [2.0, 3.0]
    assert [frame.action for frame in frames] == [1, 0]


def test_audio_streaming_loads_execution_output_dir(tmp_path: Path) -> None:
    execution_dir = tmp_path / "12_execute_output"
    execution_dir.mkdir()
    (execution_dir / "demo_results.json").write_text(
        """
        {
          "simulation_data": {
            "free_energy_trace": [1.0],
            "belief_history": [[0.7, 0.3]],
            "action_history": [1]
          }
        }
        """,
        encoding="utf-8",
    )
    summary = _process_audio_streaming(
        {"execution_output_dir": execution_dir, "audio_chunk_size": 1},
        tmp_path / "audio",
        __import__("logging").getLogger("test"),
    )

    assert summary["telemetry_source_count"] == 1
    assert summary["chunk_count"] == 1
    assert (tmp_path / "audio" / "audio_stream_chunks.json").exists()


def test_audio_streaming_follows_step12_summary_structured_result(
    tmp_path: Path,
) -> None:
    execution_dir = tmp_path / "12_execute_output"
    summaries_dir = execution_dir / "summaries"
    structured_dir = execution_dir / "demo" / "pymdp" / "execution_logs"
    structured_dir.mkdir(parents=True)
    summaries_dir.mkdir(parents=True)
    structured_result = structured_dir / "demo_results.json"
    structured_result.write_text(
        """
        {
          "simulation_data": {
            "free_energy_trace": [3.0, 1.0],
            "belief_history": [[0.2, 0.8], [0.9, 0.1]],
            "action_history": [1, 0]
          }
        }
        """,
        encoding="utf-8",
    )
    (summaries_dir / "execution_summary.json").write_text(
        f"""
        {{
          "execution_details": [
            {{"structured_result_file": "{structured_result}"}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    summary = _process_audio_streaming(
        {"execution_output_dir": execution_dir, "audio_chunk_size": 2},
        tmp_path / "audio",
        __import__("logging").getLogger("test"),
    )

    assert summary["telemetry_source_count"] == 1
    assert summary["frame_count"] == 2
    assert summary["telemetry_provenance"] == [str(structured_result)]
    assert summary["chunks"][0]["metadata"]["last_action"] == 0


def test_audio_summary_artifact_resolver_handles_execution_rooted_relative_paths(
    tmp_path: Path,
) -> None:
    execution_dir = tmp_path / "output" / "12_execute_output"
    summary_dir = execution_dir / "summaries"
    structured_result = (
        execution_dir
        / "pomdp_gridworld_3x3"
        / "pymdp"
        / "execution_logs"
        / "pomdp_gridworld_3x3_pymdp_results.json"
    )
    structured_result.parent.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    structured_result.write_text('{"simulation_data": {}}', encoding="utf-8")

    resolved = _resolve_execution_summary_artifact(
        "output/12_execute_output/pomdp_gridworld_3x3/pymdp/execution_logs/"
        "pomdp_gridworld_3x3_pymdp_results.json",
        summary_dir,
        execution_dir,
    )

    assert resolved == structured_result.resolve()
    assert "summaries/output/12_execute_output" not in str(resolved)


def test_audio_streaming_rejects_summary_pointer_outside_execution_dir(
    tmp_path: Path,
) -> None:
    execution_dir = tmp_path / "12_execute_output"
    summaries_dir = execution_dir / "summaries"
    summaries_dir.mkdir(parents=True)
    outside_result = tmp_path / "outside_results.json"
    outside_result.write_text(
        """
        {
          "simulation_data": {
            "free_energy_trace": [3.0],
            "belief_history": [[0.2, 0.8]],
            "action_history": [1]
          }
        }
        """,
        encoding="utf-8",
    )
    (summaries_dir / "execution_summary.json").write_text(
        f"""
        {{
          "execution_details": [
            {{"structured_result_file": "{outside_result}"}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    summary = _process_audio_streaming(
        {"execution_output_dir": execution_dir, "audio_chunk_size": 1},
        tmp_path / "audio",
        __import__("logging").getLogger("test"),
    )

    assert summary == {}
    assert not (tmp_path / "audio" / "audio_stream_chunks.json").exists()


def test_audio_streaming_discovers_sibling_step12_output(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    audio_dir = output_root / "15_audio_output"
    execution_dir = output_root / "12_execute_output"
    execution_dir.mkdir(parents=True)
    (execution_dir / "demo_results.json").write_text(
        """
        {
          "simulation_data": {
            "free_energy_trace": [1.0],
            "belief_history": [[0.6, 0.4]],
            "action_history": [1]
          }
        }
        """,
        encoding="utf-8",
    )

    summary = _process_audio_streaming(
        {"audio_chunk_size": 1},
        audio_dir,
        __import__("logging").getLogger("test"),
    )

    assert summary["telemetry_source_count"] == 1
    assert (audio_dir / "audio_stream_chunks.json").exists()


def test_audio_streaming_empty_telemetry_reports_no_frames(tmp_path: Path) -> None:
    summary = _process_audio_streaming(
        {"telemetry": {"simulation_data": {}}},
        tmp_path / "audio",
        __import__("logging").getLogger("test"),
    )

    assert summary["status"] == "no_frames"
    assert summary["frame_count"] == 0
    assert summary["streaming_safe"] is False
    artifact = tmp_path / "audio" / "audio_stream_chunks.json"
    assert artifact.exists()
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["status"] == "no_frames"
    assert persisted["streaming_safe"] is False
