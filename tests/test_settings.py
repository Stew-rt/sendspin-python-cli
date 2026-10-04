from __future__ import annotations

import json
from pathlib import Path

import pytest

from sendspin.settings import ClientSettings


@pytest.mark.asyncio
async def test_client_settings_source_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings = ClientSettings(_settings_file=settings_file)
    assert settings.source_enabled is False
    assert settings.source_audio_device is None
    assert settings.source_codec == "flac"
    assert settings.source_sample_rate == 48000
    assert settings.source_channels == 2
    assert settings.source_bit_depth == 16
    assert settings.player_enabled is True


@pytest.mark.asyncio
async def test_client_settings_source_update_and_save(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings = ClientSettings(_settings_file=settings_file)

    settings.update(
        source_enabled=True,
        source_audio_device="hw:1,0",
        source_codec="opus",
        source_sample_rate=44100,
        source_channels=1,
        source_bit_depth=24,
        player_enabled=False,
    )
    assert settings.source_enabled is True
    assert settings.source_audio_device == "hw:1,0"
    assert settings.source_codec == "opus"
    assert settings.source_sample_rate == 44100
    assert settings.source_channels == 1
    assert settings.source_bit_depth == 24
    assert settings.player_enabled is False

    await settings.flush()

    assert settings_file.exists()
    data = json.loads(settings_file.read_text())
    assert data["source_enabled"] is True
    assert data["source_audio_device"] == "hw:1,0"
    assert data["source_codec"] == "opus"
    assert data["source_sample_rate"] == 44100
    assert data["source_channels"] == 1
    assert data["source_bit_depth"] == 24
    assert data["player_enabled"] is False

    loaded_settings = ClientSettings(_settings_file=settings_file)
    await loaded_settings.load()
    assert loaded_settings.source_enabled is True
    assert loaded_settings.source_audio_device == "hw:1,0"
    assert loaded_settings.source_codec == "opus"
    assert loaded_settings.source_sample_rate == 44100
    assert loaded_settings.source_channels == 1
    assert loaded_settings.source_bit_depth == 24
    assert loaded_settings.player_enabled is False
