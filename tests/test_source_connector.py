"""Unit tests for SourceHandler and SourceConnector."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiosendspin.models.core import ServerCommandPayload, SourceCommandServerPayload
from aiosendspin.models.types import AudioCodec

from sendspin.audio_devices import AudioDevice
from sendspin.source_connector import SourceHandler


class FakeClient:
    """Fake SendspinClient for testing source handler attachments."""

    def __init__(self, connected: bool = True) -> None:
        self.connected = connected
        self.server_command_listeners: list[object] = []
        self.disconnect_listeners: list[object] = []
        self.source_capture_mock = MagicMock()

    def add_server_command_listener(self, callback: object) -> object:
        self.server_command_listeners.append(callback)
        return lambda: (
            self.server_command_listeners.remove(callback)
            if callback in self.server_command_listeners
            else None
        )

    def add_disconnect_listener(self, callback: object) -> object:
        self.disconnect_listeners.append(callback)
        return lambda: (
            self.disconnect_listeners.remove(callback)
            if callback in self.disconnect_listeners
            else None
        )

    def create_source_capture(self, audio_format: object) -> object:
        return self.source_capture_mock


@pytest.fixture
def mock_device() -> AudioDevice:
    return AudioDevice(
        index=2,
        name="Mock Line-In",
        input_channels=2,
        output_channels=0,
        sample_rate=48000.0,
        is_default_input=True,
    )


def test_source_handler_initialization(mock_device: AudioDevice) -> None:
    handler = SourceHandler(
        audio_device=mock_device,
        codec="flac",
        sample_rate=44100,
        channels=2,
        bit_depth=24,
    )
    assert handler.audio_device == mock_device
    assert handler.codec == AudioCodec.FLAC
    assert handler.sample_rate == 44100
    assert handler.channels == 2
    assert handler.bit_depth == 24
    assert not handler.is_streaming
    assert handler.audio_capture is None
    assert handler.source_capture is None


def test_attach_and_detach_client() -> None:
    handler = SourceHandler()
    client = FakeClient()

    handler.attach_client(client)
    assert len(client.server_command_listeners) == 1
    assert len(client.disconnect_listeners) == 1

    handler.detach_client()
    assert len(client.server_command_listeners) == 0
    assert len(client.disconnect_listeners) == 0


@pytest.mark.asyncio
async def test_start_streaming_client_not_connected() -> None:
    client = FakeClient(connected=False)
    handler = SourceHandler()
    handler.attach_client(client)

    await handler.start_streaming()
    assert not handler.is_streaming


@pytest.mark.asyncio
async def test_start_and_stop_streaming_lifecycle(mock_device: AudioDevice) -> None:
    client = FakeClient(connected=True)
    events: list[str] = []
    state_changes: list[bool] = []

    handler = SourceHandler(
        audio_device=mock_device,
        codec=AudioCodec.FLAC,
        sample_rate=48000,
        channels=2,
        bit_depth=16,
        on_state_change=state_changes.append,
        on_event=events.append,
    )
    handler.attach_client(client)

    mock_source_cap = AsyncMock()
    client.source_capture_mock = mock_source_cap

    async def fake_stream():
        yield b"\x01\x02\x03\x04", 1000
        yield b"\x05\x06\x07\x08", 2000

    with patch("sendspin.source_connector.AudioCapture") as mock_audio_cap_cls:
        mock_audio_cap = MagicMock()
        mock_audio_cap.start = AsyncMock()
        mock_audio_cap.stop = AsyncMock()
        mock_audio_cap.stream = fake_stream
        mock_audio_cap_cls.return_value = mock_audio_cap

        await handler.start_streaming()

        assert handler.is_streaming
        assert state_changes == [True]
        assert events == ["start"]
        mock_source_cap.start.assert_awaited_once()
        mock_audio_cap.start.assert_awaited_once()

        # Let stream loop execute
        await asyncio.sleep(0.01)

        assert mock_source_cap.feed.await_count == 2
        mock_source_cap.feed.assert_any_await(b"\x01\x02\x03\x04", 1000)
        mock_source_cap.feed.assert_any_await(b"\x05\x06\x07\x08", 2000)

        # Stop streaming
        await handler.stop_streaming()

        assert not handler.is_streaming
        assert state_changes == [True, False]
        assert events == ["start", "stop"]
        mock_audio_cap.stop.assert_awaited()
        mock_source_cap.stop.assert_awaited()


@pytest.mark.asyncio
async def test_start_streaming_passes_clock_to_audio_capture(mock_device: AudioDevice) -> None:
    client = FakeClient(connected=True)
    fake_clock = MagicMock()
    client._clock = fake_clock  # type: ignore[attr-defined]
    client.source_capture_mock = AsyncMock()

    handler = SourceHandler(
        audio_device=mock_device,
        codec=AudioCodec.FLAC,
        sample_rate=48000,
        channels=2,
        bit_depth=16,
    )
    handler.attach_client(client)

    with patch("sendspin.source_connector.AudioCapture") as mock_audio_cap_cls:

        async def fake_stream():
            if False:
                yield b"", 0

        mock_audio_cap = MagicMock()
        mock_audio_cap.start = AsyncMock()
        mock_audio_cap.stop = AsyncMock()
        mock_audio_cap.stream = fake_stream
        mock_audio_cap_cls.return_value = mock_audio_cap

        await handler.start_streaming()
        mock_audio_cap_cls.assert_called_once()
        assert mock_audio_cap_cls.call_args.kwargs["clock"] is fake_clock
        await handler.stop_streaming()


@pytest.mark.asyncio
async def test_server_command_routing() -> None:
    client = FakeClient(connected=True)
    handler = SourceHandler()
    handler.attach_client(client)

    with (
        patch.object(handler, "start_streaming", new_callable=AsyncMock) as mock_start,
        patch.object(handler, "stop_streaming", new_callable=AsyncMock) as mock_stop,
    ):
        on_cmd = client.server_command_listeners[0]

        # Ignore unrelated command
        on_cmd(ServerCommandPayload(player=None, source=None))
        await asyncio.sleep(0.001)
        mock_start.assert_not_called()
        mock_stop.assert_not_called()

        # Start command
        on_cmd(ServerCommandPayload(source=SourceCommandServerPayload(command="start")))
        await asyncio.sleep(0.001)
        mock_start.assert_called_once()

        # Stop command
        on_cmd(ServerCommandPayload(source=SourceCommandServerPayload(command="stop")))
        await asyncio.sleep(0.001)
        mock_stop.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_and_shutdown() -> None:
    client = FakeClient(connected=True)
    handler = SourceHandler()
    handler.attach_client(client)

    with patch.object(handler, "stop_streaming", new_callable=AsyncMock) as mock_stop:
        on_disconnect = client.disconnect_listeners[0]
        on_disconnect()
        await asyncio.sleep(0.001)
        mock_stop.assert_called_once()

    with patch.object(handler, "stop_streaming", new_callable=AsyncMock) as mock_stop:
        await handler.shutdown()
        mock_stop.assert_called_once()
        assert handler._client is None


def test_source_handler_overflow_count(mock_device: AudioDevice) -> None:
    handler = SourceHandler(audio_device=mock_device)
    assert handler.overflow_count == 0

    mock_cap = MagicMock()
    mock_cap.overflow_count = 3
    handler._audio_capture = mock_cap
    assert handler.overflow_count == 3
