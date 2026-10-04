from __future__ import annotations

import numpy as np
import pytest

from aiosendspin.audio.codecs import create_decoder
from sendspin.decoder import FlacDecoder
from sendspin.encoder import (
    FlacEncoder,
    OpusEncoder,
    PcmPassthrough,
    create_encoder,
)


def _make_audio_format(*, sample_rate: int, channels: int, bit_depth: int, codec_header: bytes):
    from types import SimpleNamespace

    return SimpleNamespace(
        pcm_format=SimpleNamespace(
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
        ),
        codec_header=codec_header,
    )


def test_create_encoder_factory() -> None:
    enc_pcm = create_encoder("pcm", sample_rate=48000, bit_depth=16, channels=2)
    assert isinstance(enc_pcm, PcmPassthrough)

    enc_flac = create_encoder("flac", sample_rate=48000, bit_depth=16, channels=2)
    assert isinstance(enc_flac, FlacEncoder)

    enc_opus = create_encoder("opus", sample_rate=48000, bit_depth=16, channels=2)
    assert isinstance(enc_opus, OpusEncoder)

    with pytest.raises(ValueError, match="Unsupported source codec"):
        create_encoder("mp3", sample_rate=48000, bit_depth=16, channels=2)


def test_flac_encoder_validation() -> None:
    # 16 and 24 bit are valid
    enc16 = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
    assert enc16.frame_samples > 0

    enc24 = FlacEncoder(sample_rate=48000, bit_depth=24, channels=2)
    assert enc24.frame_samples > 0

    with pytest.raises(ValueError, match="FLAC only supports bit depths"):
        FlacEncoder(sample_rate=48000, bit_depth=32, channels=2)


def test_flac_encoder_header() -> None:
    enc = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
    header = enc.get_header()
    assert header is not None
    assert len(header) == 42
    assert header[:4] == b"fLaC"

    codec_header = enc.get_codec_header()
    assert codec_header == header


def test_flac_encoder_roundtrip_16bit() -> None:
    sample_rate = 48000
    channels = 2
    bit_depth = 16

    encoder = FlacEncoder(sample_rate=sample_rate, bit_depth=bit_depth, channels=channels)
    header = encoder.get_codec_header()
    assert header is not None

    decoder = FlacDecoder(
        _make_audio_format(
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            codec_header=header,
        )
    )

    # Generate 100ms sine wave (exact multiple of 25ms chunk size)
    chunk_samples = int(sample_rate * 0.025)
    total_samples = chunk_samples * 4  # 100ms
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    freq_left = 440.0
    freq_right = 880.0
    sig_left = (np.sin(2 * np.pi * freq_left * t) * 30000).astype(np.int16)
    sig_right = (np.sin(2 * np.pi * freq_right * t) * 30000).astype(np.int16)
    pcm_in = np.column_stack([sig_left, sig_right]).tobytes()

    frames = encoder.process(pcm_in, timestamp_us=1000000, duration_us=100000)
    flushed = encoder.flush()
    all_packets = [f[0] for f in frames] + [f[0] for f in flushed]
    assert len(all_packets) > 0

    decoded_pcm = bytearray()
    for pkt in all_packets:
        decoded_pcm.extend(decoder.decode(pkt))

    assert bytes(decoded_pcm)[: len(pcm_in)] == pcm_in


def test_flac_encoder_roundtrip_with_aiosendspin_decoder() -> None:
    sample_rate = 48000
    channels = 2
    bit_depth = 16

    encoder = FlacEncoder(sample_rate=sample_rate, bit_depth=bit_depth, channels=channels)
    header = encoder.get_header()
    assert header is not None

    decoder = create_decoder(
        "flac",
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=channels,
        codec_header=header,
    )

    chunk_samples = int(sample_rate * 0.025)
    total_samples = chunk_samples * 4
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    sig = (np.sin(2 * np.pi * 440.0 * t) * 30000).astype(np.int16)
    pcm_in = np.column_stack([sig, sig]).tobytes()

    frames = encoder.process(pcm_in, timestamp_us=0, duration_us=100000)
    flushed = encoder.flush()
    all_packets = [f[0] for f in frames] + [f[0] for f in flushed]

    decoded_pcm = bytearray()
    for pkt in all_packets:
        decoded_pcm.extend(decoder.decode(pkt))

    assert bytes(decoded_pcm)[: len(pcm_in)] == pcm_in


def test_opus_encoder_validation() -> None:
    enc = OpusEncoder(sample_rate=48000, bit_depth=16, channels=2)
    assert enc.frame_samples > 0
    assert enc.get_header() is None

    with pytest.raises(ValueError, match="Opus only supports sample rates"):
        OpusEncoder(sample_rate=44100, bit_depth=16, channels=2)


def test_opus_encoder_process() -> None:
    encoder = OpusEncoder(sample_rate=48000, bit_depth=16, channels=2)
    t = np.linspace(0, 0.1, int(48000 * 0.1), endpoint=False)
    sig = (np.sin(2 * np.pi * 440.0 * t) * 30000).astype(np.int16)
    pcm_in = np.column_stack([sig, sig]).tobytes()

    frames = encoder.process(pcm_in, timestamp_us=500000, duration_us=100000)
    assert len(frames) > 0
    for pkt, dur in frames:
        assert len(pkt) > 0
        assert dur > 0


def test_pcm_passthrough_process() -> None:
    encoder = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2, chunk_duration_us=25000)
    assert encoder.get_header() is None
    assert encoder.get_codec_header() is None

    # 100ms of data (4 chunks of 25ms)
    pcm_in = b"\x01\x00\x02\x00" * (48000 * 25 // 1000 * 4)
    frames = encoder.process(pcm_in, timestamp_us=1000000, duration_us=100000)
    assert len(frames) == 4
    for pkt, dur in frames:
        assert len(pkt) == 48000 * 25 // 1000 * 4
        assert dur == 25000
