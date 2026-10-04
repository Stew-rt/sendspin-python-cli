from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sendspin.hardware_volume import _get_sink


class _FakePulseClient:
    def __init__(self, sinks: list[object], default_sink_name: str) -> None:
        self._sinks = sinks
        self._server_info = SimpleNamespace(default_sink_name=default_sink_name)

    async def sink_list(self) -> list[object]:
        return self._sinks

    async def server_info(self) -> SimpleNamespace:
        return self._server_info


def test_get_sink_matches_selected_alsa_device() -> None:
    sinks = [
        SimpleNamespace(
            name="alsa_output.usb-dac",
            description="USB DAC Analog Stereo",
            proplist={"alsa.card_name": "USB DAC", "alsa.name": "Analog Stereo"},
        ),
        SimpleNamespace(
            name="alsa_output.hdmi",
            description="HDMI Output",
            proplist={"alsa.card_name": "HDMI", "alsa.name": "HDMI 0"},
        ),
    ]
    client = _FakePulseClient(sinks, default_sink_name="alsa_output.hdmi")
    audio_device = SimpleNamespace(
        is_default=False,
        is_default_output=False,
        name="USB DAC: Analog Stereo (hw:2,0)",
    )

    sink = asyncio.run(_get_sink(audio_device, client))

    assert sink is sinks[0]


def test_get_sink_uses_default_sink_for_pipewire_backend_device() -> None:
    sinks = [
        SimpleNamespace(
            name="alsa_output.usb-dac",
            description="USB DAC",
            proplist={"alsa.card_name": "USB DAC", "alsa.name": "Analog Stereo"},
        ),
        SimpleNamespace(
            name="alsa_output.hdmi",
            description="HDMI Output",
            proplist={"alsa.card_name": "HDMI", "alsa.name": "HDMI 0"},
        ),
    ]
    client = _FakePulseClient(sinks, default_sink_name="alsa_output.hdmi")
    audio_device = SimpleNamespace(is_default=False, name="pipewire")

    sink = asyncio.run(_get_sink(audio_device, client))

    assert sink is sinks[1]


def test_get_sink_returns_none_for_unmatched_specific_device() -> None:
    sinks = [
        SimpleNamespace(
            name="alsa_output.hdmi",
            description="HDMI Output",
            proplist={"alsa.card_name": "HDMI", "alsa.name": "HDMI 0"},
        )
    ]
    client = _FakePulseClient(sinks, default_sink_name="alsa_output.hdmi")
    audio_device = SimpleNamespace(is_default=False, name="USB DAC: Analog Stereo (hw:2,0)")

    sink = asyncio.run(_get_sink(audio_device, client))

    assert sink is None
