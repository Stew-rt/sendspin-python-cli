"""Tests for ALSA and audio device resolution in audio_devices.py."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import sounddevice
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec

import sendspin.audio_devices as _mod
from sendspin.audio_devices import (
    AudioDevice,
    DeviceDirection,
    _check_format,
    _try_alsa_device,
    detect_supported_audio_formats,
    list_alsa_devices,
    query_devices,
    resolve_audio_device,
    validate_audio_format,
)


def test_audio_device_properties():
    """AudioDevice properties for input, output, duplex, and default."""
    out_dev = AudioDevice(
        index=0, name="Output Only", output_channels=2, input_channels=0, is_default_output=True
    )
    assert out_dev.is_output
    assert not out_dev.is_input
    assert not out_dev.is_duplex
    assert out_dev.is_default_output
    assert out_dev.device_id == 0

    in_dev = AudioDevice(
        index=1, name="Input Only", output_channels=0, input_channels=1, is_default_input=True
    )
    assert not in_dev.is_output
    assert in_dev.is_input
    assert not in_dev.is_duplex
    assert not in_dev.is_default_output
    assert in_dev.device_id == 1

    duplex_dev = AudioDevice(index=2, name="Duplex", output_channels=2, input_channels=2)
    assert duplex_dev.is_output
    assert duplex_dev.is_input
    assert duplex_dev.is_duplex
    assert duplex_dev.device_id == 2

    alsa_dev = AudioDevice(index=None, name="dmix", alsa_device_name="dmix")
    assert alsa_dev.device_id == "dmix"


def test_audio_device_post_init_validation():
    """AudioDevice requires index or alsa_device_name."""
    with pytest.raises(ValueError, match="AudioDevice must have an index or alsa_device_name"):
        AudioDevice(index=None, name="Invalid", alsa_device_name=None)


def test_query_devices_filtering():
    """query_devices filters devices by direction."""
    mock_devices = [
        {
            "name": "Output Dev",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Input Dev",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
        {
            "name": "Duplex Dev",
            "max_input_channels": 1,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Zero Dev",
            "max_input_channels": 0,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
    ]

    with (
        patch.object(sounddevice, "query_devices", return_value=mock_devices),
        patch.object(sounddevice.default, "device", (1, 0)),
    ):
        out_list = query_devices(direction=DeviceDirection.OUTPUT)
        assert len(out_list) == 2
        assert [d.name for d in out_list] == ["Output Dev", "Duplex Dev"]
        assert out_list[0].is_default_output
        assert not out_list[1].is_default_output

        in_list = query_devices(direction=DeviceDirection.INPUT)
        assert len(in_list) == 2
        assert [d.name for d in in_list] == ["Input Dev", "Duplex Dev"]
        assert in_list[0].is_default_input
        assert not in_list[1].is_default_input


def test_check_format_direction():
    """_check_format routes to check_input_settings or check_output_settings."""
    dev = AudioDevice(index=0, name="Dev", output_channels=2, input_channels=2)

    with patch.object(sounddevice, "check_output_settings") as mock_out:
        assert _check_format(dev, 48000, 2, "int16", direction=DeviceDirection.OUTPUT)
        mock_out.assert_called_once_with(device=0, samplerate=48000, channels=2, dtype="int16")

    with patch.object(sounddevice, "check_input_settings") as mock_in:
        assert _check_format(dev, 44100, 1, "int24", direction=DeviceDirection.INPUT)
        mock_in.assert_called_once_with(device=0, samplerate=44100, channels=1, dtype="int24")


def test_validate_audio_format_direction():
    """validate_audio_format checks supported format dimensions with direction."""
    dev = AudioDevice(index=0, name="Dev", output_channels=2, input_channels=2)
    fmt = SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16)

    with patch.object(sounddevice, "check_input_settings") as mock_in:
        assert validate_audio_format(fmt, dev, direction=DeviceDirection.INPUT)
        mock_in.assert_called_once_with(device=0, samplerate=48000, channels=2, dtype="int16")


def test_detect_supported_audio_formats_direction():
    """detect_supported_audio_formats respects direction."""
    dev = AudioDevice(index=0, name="Dev", output_channels=2, input_channels=2)

    with patch.object(sounddevice, "check_input_settings") as mock_in:
        formats = detect_supported_audio_formats(dev, direction=DeviceDirection.INPUT)
        assert len(formats) > 0
        assert mock_in.called


def test_list_alsa_devices_directions():
    """list_alsa_devices invokes aplay for OUTPUT and arecord for INPUT."""
    mock_stdout_play = "hw:0,0\n    Playback Device\n"
    mock_stdout_rec = "hw:1,0\n    Capture Device\n"

    def fake_run(cmd, capture_output, text, timeout):
        if "arecord" in cmd:
            return MagicMock(returncode=0, stdout=mock_stdout_rec)
        return MagicMock(returncode=0, stdout=mock_stdout_play)

    with patch.object(subprocess, "run", side_effect=fake_run):
        out_devs = list_alsa_devices(direction=DeviceDirection.OUTPUT)
        assert out_devs == [("hw:0,0", "Playback Device")]

        in_devs = list_alsa_devices(direction=DeviceDirection.INPUT)
        assert in_devs == [("hw:1,0", "Capture Device")]


def test_resolve_audio_device_input_and_output():
    """resolve_audio_device resolves default, index, and name for given direction."""
    dev_out = AudioDevice(index=0, name="USB Speaker", output_channels=2, is_default_output=True)
    dev_in = AudioDevice(index=1, name="USB Mic", input_channels=1, is_default_input=True)

    with patch.object(_mod, "query_devices") as mock_query:
        # Default output
        mock_query.return_value = [dev_out]
        resolved = resolve_audio_device(None, direction=DeviceDirection.OUTPUT)
        assert resolved.name == "USB Speaker"

        # Default input
        mock_query.return_value = [dev_in]
        resolved_in = resolve_audio_device(None, direction=DeviceDirection.INPUT)
        assert resolved_in.name == "USB Mic"

        # By index
        mock_query.return_value = [dev_out, dev_in]
        resolved_idx = resolve_audio_device("1", direction=DeviceDirection.INPUT)
        assert resolved_idx.name == "USB Mic"

        # By name prefix
        resolved_prefix = resolve_audio_device("USB Mic", direction=DeviceDirection.INPUT)
        assert resolved_prefix.name == "USB Mic"


def test_try_alsa_device_input_direction():
    """_try_alsa_device opens input ALSA devices."""
    info = {
        "name": "dsnoop_mic",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    }
    with (
        patch.object(sounddevice, "check_input_settings"),
        patch.object(sounddevice, "query_devices", return_value=info),
    ):
        result = _try_alsa_device("dsnoop_mic", direction=DeviceDirection.INPUT)

    assert result is not None
    assert result.alsa_device_name == "dsnoop_mic"
    assert result.input_channels == 2
    assert result.is_input


def test_try_alsa_device_returns_device_when_portaudio_accepts():
    """Device recognized by PortAudio is returned with queried info."""
    info = {
        "name": "jsm_speaker_left",
        "max_output_channels": 128,
        "default_samplerate": 48000.0,
    }
    with (
        patch.object(sounddevice, "check_output_settings"),
        patch.object(sounddevice, "query_devices", return_value=info),
    ):
        result = _try_alsa_device("jsm_speaker_left")

    assert result is not None
    assert result.alsa_device_name == "jsm_speaker_left"
    assert result.output_channels == 128
    assert result.sample_rate == 48000.0


def test_try_alsa_device_returns_none_on_portaudio_error():
    """PortAudioError from check_output_settings means device can't be opened."""
    with patch.object(
        sounddevice,
        "check_output_settings",
        side_effect=sounddevice.PortAudioError("err", -1, ""),
    ):
        assert _try_alsa_device("nonexistent") is None


def test_try_alsa_device_accepts_hw_card_format_known_to_alsa():
    """hw:CARD=...,DEV=... that PortAudio doesn't know but ALSA lists should work."""
    alsa_list = [
        (
            "hw:CARD=sndrpihifiberry,DEV=0",
            "snd_rpi_hifiberry_dacplus, HiFiBerry DAC+ HiFi",
        ),
    ]
    with (
        patch.object(
            sounddevice,
            "check_output_settings",
            side_effect=ValueError("not found"),
        ),
        patch.object(
            sounddevice,
            "query_devices",
            side_effect=ValueError("not found"),
        ),
        patch.object(_mod, "list_alsa_devices", return_value=alsa_list),
    ):
        result = _try_alsa_device("hw:CARD=sndrpihifiberry,DEV=0")

    assert result is not None
    assert result.alsa_device_name == "hw:CARD=sndrpihifiberry,DEV=0"
    assert result.output_channels == 2
    assert result.sample_rate == 48000.0


def test_try_alsa_device_returns_none_for_unknown_hw_card():
    """hw:CARD=...,DEV=... not in ALSA list should return None."""
    with (
        patch.object(
            sounddevice,
            "check_output_settings",
            side_effect=ValueError("not found"),
        ),
        patch.object(_mod, "list_alsa_devices", return_value=[]),
    ):
        assert _try_alsa_device("hw:CARD=nonexistent,DEV=0") is None


def test_try_alsa_device_uses_safe_defaults_when_query_fails():
    """PortAudio accepts the device but can't query info → safe defaults."""
    with (
        patch.object(sounddevice, "check_output_settings"),
        patch.object(
            sounddevice,
            "query_devices",
            side_effect=sounddevice.PortAudioError("err", -1, ""),
        ),
    ):
        result = _try_alsa_device("jsm_speaker_left")

    assert result is not None
    assert result.output_channels == 2
    assert result.sample_rate == 48000.0


def test_try_alsa_device_accepts_alsa_only_device():
    """ALSA-only device (e.g. bluealsa) not known to PortAudio should work."""
    alsa_list = [("bluealsa", "Bluetooth Audio")]
    with (
        patch.object(
            sounddevice,
            "check_output_settings",
            side_effect=ValueError("not found"),
        ),
        patch.object(
            sounddevice,
            "query_devices",
            side_effect=ValueError("not found"),
        ),
        patch.object(_mod, "list_alsa_devices", return_value=alsa_list),
    ):
        result = _try_alsa_device("bluealsa")

    assert result is not None
    assert result.alsa_device_name == "bluealsa"


def test_cli_audio_devices_list_flags():
    """CLI parser handles --input and --output for audio-devices list."""
    from sendspin.cli import parse_args

    args_default = parse_args(["audio-devices", "list"])
    assert args_default.command == "audio-devices"
    assert args_default.audio_devices_command == "list"
    assert args_default.direction == "output"

    args_in = parse_args(["audio-devices", "list", "--input"])
    assert args_in.direction == "input"

    args_out = parse_args(["audio-devices", "list", "--output"])
    assert args_out.direction == "output"


def test_list_audio_devices_output(capsys):
    """list_audio_devices prints devices formatted with channel info."""
    from sendspin.cli import list_audio_devices

    mock_devices = [
        AudioDevice(
            index=0,
            name="Built-in Output",
            output_channels=2,
            input_channels=0,
            is_default_output=True,
        ),
    ]

    with patch.object(_mod, "query_devices", return_value=mock_devices):
        list_audio_devices(direction=DeviceDirection.OUTPUT)
        captured = capsys.readouterr()
        assert "Available audio output devices:" in captured.out
        assert "Built-in Output (default)" in captured.out
        assert "Channels: 2" in captured.out
