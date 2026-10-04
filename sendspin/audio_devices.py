"""Audio device resolution, format detection, and ALSA device listing."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

import sounddevice
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec

logger = logging.getLogger(__name__)

SOUNDDEVICE_DTYPE_MAP: dict[int, str] = {
    16: "int16",
    24: "int24",
    32: "int32",
}


class DeviceDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(slots=True)
class AudioDevice:
    """Represents an audio input, output, or duplex device.

    Attributes:
        index: PortAudio device index, or None for string-named devices.
        name: Human-readable device name.
        output_channels: Number of output channels supported.
        input_channels: Number of input channels supported.
        sample_rate: Default sample rate in Hz.
        is_default_output: Whether this is the system default output device.
        is_default_input: Whether this is the system default input device.
        alsa_device_name: Raw ALSA device name for direct access (e.g. dmix
            plugin devices). When set, this is used instead of index to open
            the device via sounddevice.
    """

    index: int | None
    name: str
    output_channels: int = 0
    input_channels: int = 0
    sample_rate: float = 44100.0
    is_default_output: bool = False
    is_default_input: bool = False
    alsa_device_name: str | None = None

    def __post_init__(self) -> None:
        if self.index is None and self.alsa_device_name is None:
            raise ValueError("AudioDevice must have an index or alsa_device_name")

    @property
    def is_output(self) -> bool:
        """Whether this device is an output device."""
        return self.output_channels > 0

    @property
    def is_input(self) -> bool:
        """Whether this device is an input device."""
        return self.input_channels > 0

    @property
    def is_duplex(self) -> bool:
        """Whether this device is both an input and output device."""
        return self.is_input and self.is_output

    @property
    def device_id(self) -> int | str:
        """Return the identifier to pass to sounddevice APIs."""
        if self.alsa_device_name is not None:
            return self.alsa_device_name
        assert self.index is not None  # guaranteed by __post_init__
        return self.index


def query_devices(direction: DeviceDirection = DeviceDirection.OUTPUT) -> list[AudioDevice]:
    """Query available audio devices matching the specified direction.

    Args:
        direction: Direction to filter by (INPUT or OUTPUT).

    Returns:
        List of AudioDevice objects.
    """
    devices = sounddevice.query_devices()
    default_in, default_out = sounddevice.default.device

    result: list[AudioDevice] = []
    for i, dev in enumerate(devices):
        max_input_channels = int(dev["max_input_channels"])
        max_output_channels = int(dev["max_output_channels"])

        if direction == DeviceDirection.OUTPUT and max_output_channels <= 0:
            continue
        if direction == DeviceDirection.INPUT and max_input_channels <= 0:
            continue

        result.append(
            AudioDevice(
                index=i,
                name=str(dev["name"]),
                input_channels=max_input_channels,
                output_channels=max_output_channels,
                sample_rate=float(dev["default_samplerate"]),
                is_default_output=(i == default_out),
                is_default_input=(i == default_in),
            )
        )
    return result


def _check_format(
    device: AudioDevice,
    rate: int,
    channels: int,
    dtype: str,
    direction: DeviceDirection = DeviceDirection.OUTPUT,
) -> bool:
    """Check if a specific audio format is supported by the device."""
    try:
        if direction == DeviceDirection.INPUT:
            sounddevice.check_input_settings(
                device=device.device_id,
                samplerate=rate,
                channels=channels,
                dtype=dtype,
            )
        else:
            sounddevice.check_output_settings(
                device=device.device_id,
                samplerate=rate,
                channels=channels,
                dtype=dtype,
            )
        return True
    except sounddevice.PortAudioError:
        return False


def detect_supported_audio_formats(
    device: AudioDevice,
    direction: DeviceDirection = DeviceDirection.OUTPUT,
) -> list[SupportedAudioFormat]:
    """Detect supported audio formats by testing dimensions independently.

    Tests sample rates, bit depths, and channels separately then creates the
    cartesian product. This assumes that if individual dimensions work, their
    combinations will too (valid for PulseAudio/PipeWire which handle conversion).

    Returns formats for both PCM (raw) and FLAC (compressed) codecs. FLAC formats
    are preferred and listed first since they reduce bandwidth. FLAC decoding is
    done client-side before playback.

    Args:
        device: Audio device to check formats against.
        direction: Device direction to check formats for (INPUT or OUTPUT).

    Returns:
        List of supported audio formats, with FLAC formats first (preferred).
    """
    sample_rates = [48000, 44100, 96000, 192000]
    bit_depths = [24, 16]
    channel_counts = [2, 1]

    # Test each dimension independently
    supported_rates = [
        r for r in sample_rates if _check_format(device, r, 2, "int16", direction=direction)
    ]
    supported_depths = [
        d
        for d in bit_depths
        if _check_format(device, 48000, 2, SOUNDDEVICE_DTYPE_MAP[d], direction=direction)
    ]
    supported_channels = [
        c for c in channel_counts if _check_format(device, 48000, c, "int16", direction=direction)
    ]

    # Build formats for both FLAC (preferred) and PCM
    # FLAC is preferred as it reduces bandwidth while maintaining lossless quality
    supported: list[SupportedAudioFormat] = []

    # Add FLAC formats first (preferred)
    for depth in supported_depths:
        for rate in supported_rates:
            for ch in supported_channels:
                supported.append(
                    SupportedAudioFormat(
                        codec=AudioCodec.FLAC, channels=ch, sample_rate=rate, bit_depth=depth
                    )
                )

    # Add PCM formats as fallback
    for depth in supported_depths:
        for rate in supported_rates:
            for ch in supported_channels:
                supported.append(
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM, channels=ch, sample_rate=rate, bit_depth=depth
                    )
                )

    if not supported:
        logger.warning("Could not detect supported formats, using safe defaults")
        supported = [
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, channels=2, sample_rate=44100, bit_depth=16
            ),
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=44100, bit_depth=16),
        ]

    logger.info("Detected %d supported audio formats (FLAC + PCM)", len(supported))
    return supported


def parse_audio_format(format_str: str) -> SupportedAudioFormat:
    """Parse an audio format string into a SupportedAudioFormat.

    Format: ``codec:sample_rate:bit_depth:channels``
    Example: ``flac:48000:24:2``

    Args:
        format_str: The format string to parse.

    Returns:
        A SupportedAudioFormat matching the specification.

    Raises:
        ValueError: If the format string is invalid.
    """
    parts = format_str.lower().split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid audio format '{format_str}'. "
            "Expected format: codec:sample_rate:bit_depth:channels (e.g., flac:48000:24:2)"
        )

    codec_str, rate_str, depth_str, channels_str = parts

    if codec_str == "flac":
        codec = AudioCodec.FLAC
    elif codec_str == "pcm":
        codec = AudioCodec.PCM
    else:
        raise ValueError(f"Unknown codec '{codec_str}'. Supported codecs: flac, pcm")

    try:
        sample_rate = int(rate_str)
    except ValueError:
        raise ValueError(f"Invalid sample rate '{rate_str}'. Expected an integer.") from None

    try:
        bit_depth = int(depth_str)
    except ValueError:
        raise ValueError(f"Invalid bit depth '{depth_str}'. Expected an integer.") from None

    try:
        channels = int(channels_str)
    except ValueError:
        raise ValueError(f"Invalid channel count '{channels_str}'. Expected an integer.") from None

    return SupportedAudioFormat(
        codec=codec, channels=channels, sample_rate=sample_rate, bit_depth=bit_depth
    )


def validate_audio_format(
    fmt: SupportedAudioFormat,
    device: AudioDevice,
    direction: DeviceDirection = DeviceDirection.OUTPUT,
) -> bool:
    """Validate that an audio format's PCM dimensions are supported by the device.

    Checks sample rate, bit depth, and channel count independently against the
    audio device (matching the approach used by detect_supported_audio_formats).

    Args:
        fmt: The audio format to validate.
        device: Audio device to check formats against.
        direction: Device direction to check formats for (INPUT or OUTPUT).

    Returns:
        True if the device supports the format dimensions.
    """
    dtype = SOUNDDEVICE_DTYPE_MAP.get(fmt.bit_depth)
    if dtype is None:
        return False

    return _check_format(device, fmt.sample_rate, fmt.channels, dtype, direction=direction)


def list_alsa_devices(direction: DeviceDirection = DeviceDirection.OUTPUT) -> list[tuple[str, str]]:
    """List ALSA PCM devices from ``aplay -L`` or ``arecord -L``.

    Returns a list of (device_name, description) tuples for the specified direction.
    Returns an empty list if the ALSA command is not available.
    """
    cmd = ["arecord", "-L"] if direction == DeviceDirection.INPUT else ["aplay", "-L"]
    try:
        result = subprocess.run(
            cmd,  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    devices: list[tuple[str, str]] = []
    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Device names start at column 0, descriptions are indented
        if line and not line[0].isspace():
            name = line.strip()
            description = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("    "):
                description = lines[i + 1].strip()
            devices.append((name, description))
        i += 1

    return devices


def resolve_audio_device(
    device_arg: str | None, direction: DeviceDirection = DeviceDirection.OUTPUT
) -> AudioDevice:
    """Resolve audio device from a CLI argument.

    Args:
        device_arg: Device specifier (index number, name prefix, raw ALSA device
            name, or None for default).
        direction: Device direction (INPUT or OUTPUT).

    Returns:
        The resolved AudioDevice.

    Raises:
        ValueError: If the device cannot be found.
    """
    devices = query_devices(direction=direction)

    # Find device by: default, index, or name prefix
    if device_arg is None:
        if direction == DeviceDirection.INPUT:
            device = next((d for d in devices if d.is_default_input), None)
        else:
            device = next((d for d in devices if d.is_default_output), None)
    elif device_arg.isnumeric():
        device_id = int(device_arg)
        device = next((d for d in devices if d.index == device_id), None)
    else:
        device = next((d for d in devices if d.name.startswith(device_arg)), None)

    if device is None:
        if device_arg is None:
            dir_str = "input" if direction == DeviceDirection.INPUT else "output"
            raise ValueError(f"Default audio {dir_str} device not found.")

        # On Linux, try opening as a raw ALSA device name (e.g. dmix plugin)
        if sys.platform.startswith("linux"):
            device = _try_alsa_device(device_arg, direction=direction)

        if device is None:
            raise ValueError(f"Audio device '{device_arg}' not found.")

    logger.info("Using audio device %s: %s", device.device_id, device.name)
    return device


def _try_alsa_device(
    name: str, direction: DeviceDirection = DeviceDirection.OUTPUT
) -> AudioDevice | None:
    """Try to open a raw ALSA device by name.

    This allows using ALSA plugin devices (dmix, dsnoop, plug, etc.) that are not
    enumerated by PortAudio but can be opened by name. This is needed for
    setups like dual mono where multiple clients share hardware via dmix.

    Returns:
        An AudioDevice if the ALSA device could be opened, None otherwise.
    """
    try:
        if direction == DeviceDirection.INPUT:
            sounddevice.check_input_settings(device=name)
        else:
            sounddevice.check_output_settings(device=name)
    except sounddevice.PortAudioError:
        return None
    except ValueError:
        # PortAudio doesn't recognize this device name (e.g. hw:CARD=...,DEV=...).
        # Validate it exists in ALSA before accepting it with safe defaults.
        alsa_names = {dev_name for dev_name, _ in list_alsa_devices(direction=direction)}
        if name not in alsa_names:
            return None

    # Try to query device info from PortAudio
    kind = "input" if direction == DeviceDirection.INPUT else "output"
    try:
        # FIXME: sounddevice.query_devices() throws exception if more than one device has the same name
        info = sounddevice.query_devices(name, kind)
        in_channels = int(info.get("max_input_channels", 0))
        out_channels = int(info.get("max_output_channels", 0))
        sample_rate = float(info.get("default_samplerate", 48000.0))
    except (sounddevice.PortAudioError, ValueError):
        # PortAudio can't enumerate this device — use safe defaults.
        # The actual format is negotiated with the server later.
        in_channels = 2 if direction == DeviceDirection.INPUT else 0
        out_channels = 2 if direction == DeviceDirection.OUTPUT else 0
        sample_rate = 48000.0

    return AudioDevice(
        index=None,
        name=name,
        input_channels=in_channels,
        output_channels=out_channels,
        sample_rate=sample_rate,
        is_default_output=False,
        is_default_input=False,
        alsa_device_name=name,
    )
