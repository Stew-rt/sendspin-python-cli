from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from aiosendspin.models.types import Roles
from aiosendspin.noise import Identity, InMemoryClientPairingStore
from rich.console import Console

from sendspin.audio_devices import AudioDevice
from sendspin.settings import ClientSettings
from sendspin.tui.app import AppArgs, SendspinApp
from sendspin.tui.ui import SendspinUI


class _FakeAudioHandler:
    def __init__(self, *, volume: int = 50, muted: bool = False) -> None:
        self.volume = volume
        self.muted = muted

    def attach_client(self, client: object) -> None:
        pass

    def detach_client(self) -> None:
        pass

    async def handle_disconnect(self) -> None:
        pass


class _FakeSourceHandler:
    def __init__(self) -> None:
        self.attached_client: object | None = None
        self.detached = False
        self.disconnected = False
        self.shutdown_called = False

    def attach_client(self, client: object) -> None:
        self.attached_client = client

    def detach_client(self) -> None:
        self.detached = True
        self.attached_client = None

    async def handle_disconnect(self) -> None:
        self.disconnected = True

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeUI:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.source_streaming: bool | None = None
        self.state = SimpleNamespace(artwork_image=None, artwork_generation=0)

    def add_event(self, event: str) -> None:
        self.events.append(event)

    def set_source_streaming(self, streaming: bool) -> None:
        self.source_streaming = streaming

    def set_visualizer_enabled(self, enabled: bool) -> None:
        pass

    def set_visualizer_types(self, types: object) -> None:
        pass

    def set_server_clock(self, clock: object) -> None:
        pass

    def reset_palette(self) -> None:
        pass

    def clear_beats(self) -> None:
        pass

    def clear_peaks(self) -> None:
        pass

    def set_disconnected(self, message: str) -> None:
        pass


@pytest.mark.asyncio
async def test_tui_create_client_roles_and_source_support(tmp_path: Path) -> None:
    settings = ClientSettings(_settings_file=tmp_path / "settings.json")
    audio_device = AudioDevice(index=0, name="Fake Device", output_channels=2)

    # Disabled source
    args_no_source = AppArgs(
        audio_device=audio_device,
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test App",
        settings=settings,
        use_mpris=False,
        source_enabled=False,
    )
    app_no_source = SendspinApp(args_no_source)
    app_no_source._audio_handler = _FakeAudioHandler()
    client_no_source = app_no_source._create_client()

    assert Roles.SOURCE not in client_no_source.roles
    assert client_no_source.source_support is None

    # Enabled source
    args_with_source = AppArgs(
        audio_device=audio_device,
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test App Source",
        settings=settings,
        use_mpris=False,
        source_enabled=True,
    )
    app_with_source = SendspinApp(args_with_source)
    app_with_source._audio_handler = _FakeAudioHandler()
    client_with_source = app_with_source._create_client()

    assert Roles.SOURCE in client_with_source.roles
    assert client_with_source.source_support is not None


@pytest.mark.asyncio
async def test_tui_source_handler_lifecycle(tmp_path: Path) -> None:
    settings = ClientSettings(_settings_file=tmp_path / "settings.json")
    audio_device = AudioDevice(index=0, name="Fake Device", output_channels=2)
    args = AppArgs(
        audio_device=audio_device,
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test App Source",
        settings=settings,
        use_mpris=False,
        source_enabled=True,
    )
    app = SendspinApp(args)
    fake_audio = _FakeAudioHandler()
    fake_source = _FakeSourceHandler()
    app._audio_handler = fake_audio
    app._source_handler = fake_source  # type: ignore[assignment]
    app._ui = _FakeUI()  # type: ignore[assignment]

    fake_client = SimpleNamespace(
        add_metadata_listener=lambda cb: (lambda: None),
        add_group_update_listener=lambda cb: (lambda: None),
        add_controller_state_listener=lambda cb: (lambda: None),
        add_server_command_listener=lambda cb: (lambda: None),
        add_color_listener=lambda cb: (lambda: None),
        add_pairing_abort_listener=lambda cb: (lambda: None),
    )
    app._client = fake_client  # type: ignore[assignment]

    # Attach
    app._attach_client()
    assert fake_source.attached_client is fake_client

    # Disconnect
    await app._handle_disconnect("Disconnected")
    assert fake_source.disconnected is True

    # Detach
    app._detach_client()
    assert fake_source.detached is True


def test_tui_source_callbacks(tmp_path: Path) -> None:
    settings = ClientSettings(_settings_file=tmp_path / "settings.json")
    audio_device = AudioDevice(index=0, name="Fake Device", output_channels=2)
    args = AppArgs(
        audio_device=audio_device,
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test App Source",
        settings=settings,
        use_mpris=False,
        source_enabled=True,
    )
    app = SendspinApp(args)
    ui = _FakeUI()
    app._ui = ui  # type: ignore[assignment]

    assert app._state.source_streaming is False
    app._on_source_state_change(True)
    assert app._state.source_streaming is True
    assert ui.source_streaming is True

    app._on_source_event("start")
    assert any("streaming started" in ev for ev in ui.events)

    app._on_source_error(RuntimeError("Audio test error"))
    assert any("Audio test error" in ev for ev in ui.events)


def test_sendspin_ui_source_panel_rendering() -> None:
    # Player mode (default): Stream quality panel has no Source row
    ui_player = SendspinUI(delay_ms=0.0, player_enabled=True, source_enabled=False)
    panel_player = ui_player._build_stream_quality_panel()
    console = Console(record=True, width=80)
    console.print(panel_player)
    text_player = console.export_text()
    assert "Source:" not in text_player
    assert "Delay:" in text_player

    # Source streaming UI (Idle)
    ui_with_source = SendspinUI(
        delay_ms=0.0,
        player_enabled=False,
        source_enabled=True,
        source_device_name="USB Mic",
        source_codec="flac",
        source_sample_rate=48000,
        source_bit_depth=16,
        source_channels=2,
    )
    panel_idle = ui_with_source._build_source_stream_panel()
    console = Console(record=True, width=80)
    console.print(panel_idle)
    text_idle = console.export_text()
    assert "Status: Idle" in text_idle or "Idle" in text_idle
    assert "Device: USB Mic" in text_idle or "USB Mic" in text_idle
    assert "Codec: FLAC" in text_idle or "FLAC" in text_idle
    assert "48.0kHz" in text_idle
    assert "16bit" in text_idle
    assert "Stereo" in text_idle

    # Source streaming (Active)
    ui_with_source.set_source_streaming(True)
    panel_streaming = ui_with_source._build_source_stream_panel()
    console = Console(record=True, width=80)
    console.print(panel_streaming)
    text_streaming = console.export_text()
    assert "Streaming" in text_streaming


def test_sendspin_ui_layout_input_mode_vs_player_mode() -> None:
    # Input streaming mode: only streaming relevant UI elements
    ui_streaming = SendspinUI(
        delay_ms=0.0,
        player_enabled=False,
        source_enabled=True,
        source_device_name="USB Mic",
        source_codec="flac",
        source_sample_rate=48000,
        source_bit_depth=16,
        source_channels=2,
    )
    layout_streaming = ui_streaming._build_layout()
    console = Console(record=True, width=80)
    console.print(layout_streaming)
    text_streaming = console.export_text()

    # Stream & Server panels present, VU Meter absent
    assert "VU Meter" not in text_streaming
    assert "Stream" in text_streaming
    assert "Server" in text_streaming
    assert "USB Mic" in text_streaming
    # Player-only panels absent
    assert "Playback" not in text_streaming
    assert "Volume" not in text_streaming
    assert "visualizer" not in text_streaming
    assert "repeat" not in text_streaming
    assert "shuffle" not in text_streaming
    assert "change group" not in text_streaming

    # Player mode: player UI elements present, streaming elements absent
    ui_player = SendspinUI(
        delay_ms=0.0,
        player_enabled=True,
        source_enabled=False,
    )
    layout_player = ui_player._build_layout()
    console = Console(record=True, width=80)
    console.print(layout_player)
    text_player = console.export_text()

    assert "VU Meter" not in text_player
    assert "Volume" in text_player
    assert "Playback" in text_player
    assert "Stream" in text_player
    assert "Server" in text_player
    assert "visualizer" in text_player
    assert "USB Mic" not in text_player


@pytest.mark.asyncio
async def test_tui_create_client_input_streaming_mode(tmp_path: Path) -> None:
    settings = ClientSettings(_settings_file=tmp_path / "settings.json")
    args = AppArgs(
        audio_device=None,
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test Input Streamer",
        settings=settings,
        use_mpris=False,
        player_enabled=False,
        source_enabled=True,
    )
    app = SendspinApp(args)
    client = app._create_client()

    assert Roles.SOURCE in client.roles
    assert Roles.PLAYER not in client.roles
    assert client.source_support is not None
    assert client.player_support is None
