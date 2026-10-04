from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiosendspin.clock import ManualClock, RawMonotonicClock
from sounddevice import CallbackFlags

from sendspin.audio_devices import AudioDevice
from sendspin.audio_input import AudioCapture


@pytest.fixture
def mock_input_device() -> AudioDevice:
    return AudioDevice(
        index=1,
        name="Mock Microphone",
        input_channels=2,
        output_channels=0,
        sample_rate=48000.0,
        is_default_input=True,
    )


def test_audio_capture_init_unsupported_bit_depth(mock_input_device: AudioDevice) -> None:
    with pytest.raises(ValueError, match="Unsupported bit depth"):
        AudioCapture(device=mock_input_device, bit_depth=8)


def test_audio_capture_properties(mock_input_device: AudioDevice) -> None:
    cap = AudioCapture(
        device=mock_input_device,
        sample_rate=44100,
        channels=1,
        bit_depth=24,
        block_size=512,
    )
    assert cap.device == mock_input_device
    assert cap.sample_rate == 44100
    assert cap.channels == 1
    assert cap.bit_depth == 24
    assert cap.block_size == 512
    assert isinstance(cap.clock, RawMonotonicClock)
    assert not cap.is_running


@pytest.mark.asyncio
async def test_audio_capture_start_stop(mock_input_device: AudioDevice) -> None:
    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        cap = AudioCapture(device=mock_input_device, sample_rate=48000, channels=2, bit_depth=16)
        assert not cap.is_running

        await cap.start()
        assert cap.is_running
        mock_stream_cls.assert_called_once()
        _, kwargs = mock_stream_cls.call_args
        assert kwargs["samplerate"] == 48000
        assert kwargs["channels"] == 2
        assert kwargs["dtype"] == "int16"
        assert kwargs["device"] == 1
        mock_stream.start.assert_called_once()

        # Repeated start is no-op
        await cap.start()
        assert mock_stream_cls.call_count == 1

        await cap.stop()
        assert not cap.is_running
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

        # Repeated stop is no-op
        await cap.stop()


@pytest.mark.asyncio
async def test_audio_capture_stream_and_context_manager(mock_input_device: AudioDevice) -> None:
    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        cap = AudioCapture(device=mock_input_device, sample_rate=48000, channels=2, bit_depth=16)

        async with cap:
            assert cap.is_running
            # Simulate PortAudio callback
            cb = mock_stream_cls.call_args[1]["callback"]
            dummy_pcm = b"\x01\x02\x03\x04" * 100
            time_info = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.0)
            status = CallbackFlags()

            cb(dummy_pcm, 100, time_info, status)

            # Read from async stream
            generator = cap.stream()
            chunk, ts = await anext(generator)
            assert chunk == dummy_pcm
            assert isinstance(ts, int)
            assert ts > 0

        assert not cap.is_running


@pytest.mark.asyncio
async def test_audio_capture_callback_overflow_handling(mock_input_device: AudioDevice) -> None:
    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        cap = AudioCapture(
            device=mock_input_device,
            sample_rate=48000,
            channels=2,
            bit_depth=16,
            max_queue_size=2,
        )
        await cap.start()

        cb = mock_stream_cls.call_args[1]["callback"]
        status = SimpleNamespace(input_overflow=True, input_underflow=False)
        time_info = SimpleNamespace(inputBufferAdcTime=0.0, currentTime=0.0)

        # Push 3 chunks into a queue of size 2
        cb(b"chunk1", 1, time_info, status)
        cb(b"chunk2", 1, time_info, status)
        cb(b"chunk3", 1, time_info, status)

        # Give event loop a moment to run queue pushes
        await asyncio.sleep(0.01)

        # The oldest chunk (chunk1) should have been dropped to make room for chunk3
        assert cap._queue is not None
        assert cap._queue.qsize() == 2
        item1 = await cap._queue.get()
        assert item1 is not None and item1[0] == b"chunk2"
        item2 = await cap._queue.get()
        assert item2 is not None and item2[0] == b"chunk3"

        await cap.stop()


@pytest.mark.asyncio
async def test_audio_capture_stream_encoded_flac_and_decode(mock_input_device: AudioDevice) -> None:
    import numpy as np
    from sendspin.decoder import FlacDecoder

    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        cap = AudioCapture(device=mock_input_device, sample_rate=48000, channels=2, bit_depth=16)

        # Generate 100ms sine wave (4 chunks of 25ms / 1200 samples = 4800 samples total)
        sample_rate = 48000
        total_samples = 4800
        t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
        sig = (np.sin(2 * np.pi * 440.0 * t) * 30000).astype(np.int16)
        raw_pcm = np.column_stack([sig, sig]).tobytes()

        encoded_frames: list[bytes] = []

        async def _consume() -> None:
            async for frame, _dur in cap.stream_encoded(codec="flac", chunk_duration_us=25000):
                encoded_frames.append(frame)

        consume_task = asyncio.create_task(_consume())

        # Wait for capture to start
        while not cap.is_running:
            await asyncio.sleep(0.001)

        cb = mock_stream_cls.call_args[1]["callback"]
        time_info = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.0)
        status = CallbackFlags()

        # Feed raw PCM in chunks
        chunk_size = 48000 * 25 // 1000 * 4  # 4800 bytes per 25ms chunk
        for i in range(0, len(raw_pcm), chunk_size):
            cb(raw_pcm[i : i + chunk_size], 1200, time_info, status)
            await asyncio.sleep(0.005)

        await cap.stop()
        await consume_task

        assert len(encoded_frames) > 0

        # Verify decoding with FlacDecoder using standard FLAC header
        from sendspin.encoder import create_encoder

        ref_encoder = create_encoder(
            "flac", sample_rate=48000, bit_depth=16, channels=2, chunk_duration_us=25000
        )
        header = ref_encoder.get_codec_header()

        decoder_fmt = SimpleNamespace(
            pcm_format=SimpleNamespace(
                sample_rate=48000,
                channels=2,
                bit_depth=16,
            ),
            codec_header=header,
        )
        decoder = FlacDecoder(decoder_fmt)
        decoded = bytearray()
        for f in encoded_frames:
            decoded.extend(decoder.decode(f))

        assert bytes(decoded)[: len(raw_pcm)] == raw_pcm


@pytest.mark.asyncio
async def test_audio_capture_deterministic_sample_timestamps(
    mock_input_device: AudioDevice,
) -> None:
    """Verify timestamps advance deterministically by sample count without wall-clock jitter."""
    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        sample_rate = 48000
        cap = AudioCapture(
            device=mock_input_device, sample_rate=sample_rate, channels=2, bit_depth=16
        )
        await cap.start()

        cb = mock_stream_cls.call_args[1]["callback"]
        time_info = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.0)
        status = CallbackFlags()

        # Simulate 5 chunks of 1024 frames each
        frames_per_chunk = 1024
        dummy_pcm = b"\x00\x00\x00\x00" * frames_per_chunk

        # Intentionally inject varying wall clock time jitter in time_info
        cb(dummy_pcm, frames_per_chunk, time_info, status)
        time_info_jitter1 = SimpleNamespace(inputBufferAdcTime=1.5, currentTime=1.6)
        cb(dummy_pcm, frames_per_chunk, time_info_jitter1, status)
        time_info_jitter2 = SimpleNamespace(inputBufferAdcTime=2.0, currentTime=2.3)
        cb(dummy_pcm, frames_per_chunk, time_info_jitter2, status)
        time_info_jitter3 = SimpleNamespace(inputBufferAdcTime=0.0, currentTime=0.0)
        cb(dummy_pcm, frames_per_chunk, time_info_jitter3, status)
        time_info_jitter4 = SimpleNamespace(inputBufferAdcTime=5.0, currentTime=5.0)
        cb(dummy_pcm, frames_per_chunk, time_info_jitter4, status)

        generator = cap.stream()
        timestamps: list[int] = []
        for _ in range(5):
            _, ts = await anext(generator)
            timestamps.append(ts)

        await cap.stop()

        # Verify initial timestamp exists
        anchor = timestamps[0]
        for i, ts in enumerate(timestamps):
            expected_ts = anchor + (i * frames_per_chunk * 1_000_000) // sample_rate
            assert ts == expected_ts, f"Chunk {i} expected {expected_ts} but got {ts}"


@pytest.mark.asyncio
async def test_audio_capture_manual_clock_timestamping(mock_input_device: AudioDevice) -> None:
    """Verify AudioCapture accurately derives timestamps with injected ManualClock."""
    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        clock = ManualClock(now_us_value=100_000_000)
        sample_rate = 48000
        cap = AudioCapture(
            device=mock_input_device,
            sample_rate=sample_rate,
            channels=2,
            bit_depth=16,
            clock=clock,
        )
        assert cap.clock is clock
        await cap.start()

        cb = mock_stream_cls.call_args[1]["callback"]
        # ADC was 20ms (0.02s) before current PortAudio time
        time_info = SimpleNamespace(inputBufferAdcTime=1.00, currentTime=1.02)
        status = CallbackFlags()

        frames = 960  # 20ms at 48kHz
        dummy_pcm = b"\x00\x00\x00\x00" * frames

        # First chunk
        cb(dummy_pcm, frames, time_info, status)

        # Advance manual clock by 100ms in background (simulating time passing)
        clock.advance_us(100_000)

        # Second chunk (ADC time jitter ignored because anchor is established)
        time_info_2 = SimpleNamespace(inputBufferAdcTime=1.02, currentTime=1.04)
        cb(dummy_pcm, frames, time_info_2, status)

        generator = cap.stream()
        _, ts1 = await anext(generator)
        _, ts2 = await anext(generator)

        await cap.stop()

        # Anchor should be clock.now_us() at callback time (100_000_000) minus 20ms (20_000us) = 99_980_000
        assert ts1 == 99_980_000
        # Second chunk should advance exactly 960 frames = 20_000us
        assert ts2 == 99_980_000 + 20_000


@pytest.mark.asyncio
async def test_audio_capture_overflow_underflow_counts(mock_input_device: AudioDevice) -> None:
    """Verify AudioCapture counts PortAudio buffer overflows and underflows."""
    with patch("sounddevice.RawInputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        cap = AudioCapture(device=mock_input_device, sample_rate=48000, channels=2, bit_depth=16)
        assert cap.overflow_count == 0
        assert cap.underflow_count == 0
        await cap.start()

        cb = mock_stream_cls.call_args[1]["callback"]
        time_info = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.0)

        status_overflow = CallbackFlags()
        status_overflow.input_overflow = True
        cb(b"\x00\x00\x00\x00" * 100, 100, time_info, status_overflow)
        assert cap.overflow_count == 1
        assert cap.underflow_count == 0

        status_underflow = CallbackFlags()
        status_underflow.input_underflow = True
        cb(b"\x00\x00\x00\x00" * 100, 100, time_info, status_underflow)
        assert cap.overflow_count == 1
        assert cap.underflow_count == 1

        await cap.stop()
