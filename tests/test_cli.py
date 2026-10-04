from __future__ import annotations

from sendspin.cli import parse_args


def test_cli_player_source_flags_default() -> None:
    args = parse_args([])
    assert getattr(args, "source_enabled", False) is False
    assert getattr(args, "source_device", None) is None
    assert getattr(args, "source_codec", None) is None
    assert getattr(args, "source_sample_rate", None) is None
    assert getattr(args, "source_channels", None) is None
    assert getattr(args, "source_bit_depth", None) is None


def test_cli_player_source_flags_provided() -> None:
    args = parse_args(
        [
            "--source",
            "--source-device",
            "1",
            "--source-codec",
            "opus",
            "--source-rate",
            "44100",
            "--source-channels",
            "1",
            "--source-bit-depth",
            "24",
        ]
    )
    assert args.source_enabled is True
    assert args.source_device == "1"
    assert args.source_codec == "opus"
    assert args.source_sample_rate == 44100
    assert args.source_channels == 1
    assert args.source_bit_depth == 24


def test_cli_player_enable_source_alias() -> None:
    args = parse_args(["--enable-source"])
    assert args.source_enabled is True


def test_cli_daemon_source_flags_default() -> None:
    args = parse_args(["daemon"])
    assert getattr(args, "source_enabled", False) is False
    assert getattr(args, "source_device", None) is None
    assert getattr(args, "source_codec", None) is None
    assert getattr(args, "source_sample_rate", None) is None
    assert getattr(args, "source_channels", None) is None
    assert getattr(args, "source_bit_depth", None) is None


def test_cli_daemon_source_flags_provided() -> None:
    args = parse_args(
        [
            "daemon",
            "--source",
            "--source-device",
            "hw:0,0",
            "--source-codec",
            "pcm",
            "--source-rate",
            "48000",
            "--source-channels",
            "2",
            "--source-bit-depth",
            "16",
        ]
    )
    assert args.source_enabled is True
    assert args.source_device == "hw:0,0"
    assert args.source_codec == "pcm"
    assert args.source_sample_rate == 48000
    assert args.source_channels == 2
    assert args.source_bit_depth == 16


def test_cli_daemon_source_flag() -> None:
    args = parse_args(["daemon", "--source"])
    assert args.source_enabled is True


def test_cli_source_input_device_flag() -> None:
    args = parse_args(["daemon", "--source", "--source-device", "hw:1,0"])
    assert args.source_enabled is True
    assert args.source_device == "hw:1,0"
