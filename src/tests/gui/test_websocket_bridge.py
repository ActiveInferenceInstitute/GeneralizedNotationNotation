from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, cast

import pytest
import websockets

from gui.websocket_bridge import (
    GUI_WEBSOCKET_MESSAGE_TYPES,
    GUIBridgeState,
    GUIWebSocketMessage,
    build_initial_messages,
    run_local_gui_bridge,
)


def test_gui_websocket_message_types_are_explicit() -> None:
    assert GUI_WEBSOCKET_MESSAGE_TYPES == {
        "model.load",
        "matrix.patch",
        "validation.result",
        "model.export",
        "error",
    }


def test_gui_websocket_message_round_trips() -> None:
    message = GUIWebSocketMessage(
        type="matrix.patch", payload={"matrix": "A", "path": [0, 1], "value": 0.5}
    )
    parsed = GUIWebSocketMessage.from_json(message.to_json())
    assert parsed == message


def test_gui_websocket_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        GUIWebSocketMessage.from_json(json.dumps({"type": "unknown", "payload": {}}))


def test_build_initial_messages_uses_model_load() -> None:
    messages = build_initial_messages([{"model_name": "demo"}])
    assert messages[0].type == "model.load"
    assert messages[0].payload["model_name"] == "demo"


def test_gui_bridge_state_load_patch_and_export_flow() -> None:
    state = GUIBridgeState()
    load_response = state.apply_message(
        GUIWebSocketMessage(
            type="model.load",
            payload={"model_id": "demo", "matrices": {"A": [[1.0, 0.0]]}},
        )
    )
    assert load_response is not None
    assert load_response.type == "validation.result"

    patch_response = state.apply_message(
        GUIWebSocketMessage(
            type="matrix.patch",
            payload={
                "model_id": "demo",
                "matrix": "A",
                "path": [0, 1],
                "value": 0.25,
            },
        )
    )
    assert patch_response is not None
    assert patch_response.payload["valid"] is True
    assert state.models["demo"]["matrices"]["A"][0][1] == 0.25

    export_response = state.apply_message(
        GUIWebSocketMessage(type="model.export", payload={"model_id": "demo"})
    )
    assert export_response is not None
    assert export_response.type == "model.export"
    assert export_response.payload["models"]["demo"]["matrices"]["A"][0][1] == 0.25


@pytest.mark.asyncio
async def test_gui_bridge_rejects_nonlocal_host() -> None:
    with pytest.raises(ValueError, match="local-only"):
        await run_local_gui_bridge("0.0.0.0", 8765, [])


@pytest.mark.asyncio
async def test_gui_bridge_runs_local_websocket_exchange() -> None:
    port = _free_local_port()
    initial = [
        GUIWebSocketMessage(
            type="model.load",
            payload={"model_id": "demo", "matrices": {"A": [[1.0, 0.0]]}},
        )
    ]
    task = asyncio.create_task(run_local_gui_bridge("127.0.0.1", port, initial))
    try:
        websocket = cast(
            Any, await _connect_with_retry(websockets, f"ws://127.0.0.1:{port}")
        )
        async with websocket:
            loaded = GUIWebSocketMessage.from_json(await websocket.recv())
            assert loaded.type == "model.load"
            await websocket.send(
                GUIWebSocketMessage(
                    type="matrix.patch",
                    payload={
                        "model_id": "demo",
                        "matrix": "A",
                        "path": [0, 1],
                        "value": 0.25,
                    },
                    request_id="patch-1",
                ).to_json()
            )
            validation = GUIWebSocketMessage.from_json(await websocket.recv())
            assert validation.type == "validation.result"
            assert validation.request_id == "patch-1"
            assert validation.payload["valid"] is True
            await websocket.send(
                GUIWebSocketMessage(
                    type="model.export",
                    payload={"model_id": "demo"},
                    request_id="export-1",
                ).to_json()
            )
            exported = GUIWebSocketMessage.from_json(await websocket.recv())
            assert exported.type == "model.export"
            assert exported.payload["models"]["demo"]["matrices"]["A"][0][1] == 0.25
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _connect_with_retry(websockets: Any, uri: str) -> Any:
    last_error: Exception | None = None
    for _ in range(20):
        try:
            return await websockets.connect(uri)
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.05)
    raise AssertionError(f"WebSocket bridge did not start: {last_error}")
