"""Audio capture engine for recording raw PCM audio from input devices."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Final

import sounddevice
from aiosendspin.clock import Clock, RawMonotonicClock
from aiosendspin.audio.codecs import create_encoder
from sounddevice import CallbackFlags

from sendspin.audio_devices import (
    SOUNDDEVICE_DTYPE_MAP,
    AudioDevice,
    DeviceDirection,
    resolve_audio_device,
)

if TYPE_CHECKING:
    from sounddevice import _CData

logger = logging.getLogger(__name__)

_DEFAULT_MAX_QUEUE_SIZE: Final[int] = 100


class AudioCapture:
    """Manages low-latency audio capture from an input AudioDevice.

    Captures raw PCM audio chunks from a PortAudio input stream using
    sounddevice.RawInputStream, queues them asynchronously, and yields
    them alongside capture timestamps.
    """

    def __init__(
        self,
        device: AudioDevice | None = None,
        sample_rate: int = 48000,
        channels: int = 2,
        bit_depth: int = 16,
        block_size: int = 1024,
        latency: float | str | None = "low",
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the audio capture engine.

        Args:
            device: AudioDevice to capture from (default input if None).
            sample_rate: Sample rate in Hz (e.g. 44100, 48000).
            channels: Number of audio channels (1=mono, 2=stereo).
            bit_depth: Bits per sample (16, 24, 32).
            block_size: PortAudio block size in samples per chunk.
            latency: Requested latency ("low", "high", or float in seconds).
            max_queue_size: Maximum buffered chunks before dropping on backlog.
            clock: Monotonic clock protocol instance for timestamping.
        """
        if bit_depth not in SOUNDDEVICE_DTYPE_MAP:
            raise ValueError(
                f"Unsupported bit depth: {bit_depth}. Supported: {list(SOUNDDEVICE_DTYPE_MAP.keys())}"
            )

        self._device = (
            device if device is not None else resolve_audio_device(None, DeviceDirection.INPUT)
        )
        self._sample_rate = sample_rate
        self._channels = channels
        self._bit_depth = bit_depth
        self._block_size = block_size
        self._latency = latency
        self._max_queue_size = max_queue_size
        self._clock: Clock = clock if clock is not None else RawMonotonicClock()

        self._stream: sounddevice.RawInputStream | None = None
        self._queue: asyncio.Queue[tuple[bytes, int] | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._is_running = False
        self._closed = False
        self._anchor_ts_us: int | None = None
        self._total_frames: int = 0
        self._overflow_count: int = 0
        self._underflow_count: int = 0

    @property
    def clock(self) -> Clock:
        """The clock instance used for capture timestamps."""
        return self._clock

    @property
    def overflow_count(self) -> int:
        """Total number of PortAudio input buffer overflows detected."""
        return self._overflow_count

    @property
    def underflow_count(self) -> int:
        """Total number of PortAudio input buffer underflows detected."""
        return self._underflow_count

    @property
    def device(self) -> AudioDevice:
        """The configured audio input device."""
        return self._device

    @property
    def sample_rate(self) -> int:
        """Configured sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Configured channel count."""
        return self._channels

    @property
    def bit_depth(self) -> int:
        """Configured bit depth."""
        return self._bit_depth

    @property
    def block_size(self) -> int:
        """Configured block size in samples."""
        return self._block_size

    @property
    def is_running(self) -> bool:
        """Whether audio capture is currently active."""
        return self._is_running and self._stream is not None

    def _audio_callback(
        self,
        indata: bytes | _CData,
        frames: int,
        time_info: sounddevice._CData,
        status: CallbackFlags,
    ) -> None:
        """PortAudio input stream callback executed on real-time audio thread."""
        if status:
            if status.input_overflow:
                self._overflow_count += 1
                logger.warning("Audio capture input overflow detected")
            if status.input_underflow:
                self._underflow_count += 1
                logger.warning("Audio capture input underflow detected")

        if not self._is_running or self._queue is None or self._loop is None:
            return

        now_us = self._clock.now_us()

        if self._anchor_ts_us is None:
            adc_time = getattr(time_info, "inputBufferAdcTime", 0.0)
            curr_time = getattr(time_info, "currentTime", 0.0)
            if adc_time > 0 and curr_time > 0:
                offset_us = int((curr_time - adc_time) * 1_000_000)
                self._anchor_ts_us = now_us - offset_us
            else:
                self._anchor_ts_us = now_us
            self._total_frames = 0

        timestamp_us = self._anchor_ts_us + (self._total_frames * 1_000_000) // self._sample_rate
        frame_count = (
            frames if frames > 0 else len(indata) // ((self._bit_depth // 8) * self._channels)
        )
        self._total_frames += frame_count
        pcm_bytes = bytes(indata)

        def _push() -> None:
            if self._queue is None:
                return
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    logger.debug("Dropped oldest captured audio chunk due to full queue")
                except asyncio.QueueEmpty:
                    pass
            try:
                self._queue.put_nowait((pcm_bytes, timestamp_us))
            except asyncio.QueueFull:
                pass

        try:
            self._loop.call_soon_threadsafe(_push)
        except RuntimeError:
            # Event loop might be closing
            pass

    async def start(self) -> None:
        """Start capturing audio from the input device."""
        if self._is_running:
            return

        self._closed = False
        self._anchor_ts_us = None
        self._total_frames = 0
        self._overflow_count = 0
        self._underflow_count = 0
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._is_running = True

        dtype = SOUNDDEVICE_DTYPE_MAP[self._bit_depth]
        self._stream = sounddevice.RawInputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=dtype,
            blocksize=self._block_size,
            callback=self._audio_callback,
            latency=self._latency,
            device=self._device.device_id,
        )
        self._stream.start()
        logger.info(
            "AudioCapture started on device %s (%s, %d Hz, %d ch, %d-bit)",
            self._device.device_id,
            self._device.name,
            self._sample_rate,
            self._channels,
            self._bit_depth,
        )

    async def stop(self) -> None:
        """Stop capturing audio and signal end-of-stream to active consumers."""
        if not self._is_running:
            return

        self._is_running = False
        self._anchor_ts_us = None
        self._total_frames = 0

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing capture stream: %s", e)
            finally:
                self._stream = None

        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        logger.info("AudioCapture stopped")

    async def close(self) -> None:
        """Alias for stop(); ensures stream is closed and resources released."""
        await self.stop()
        self._closed = True

    async def stream(self) -> AsyncGenerator[tuple[bytes, int], None]:
        """Asynchronous generator yielding captured (pcm_bytes, timestamp_us) tuples.

        Yields:
            tuple[bytes, int]: Raw PCM chunk and monotonic capture timestamp in microseconds.
        """
        if not self._is_running:
            await self.start()

        assert self._queue is not None
        while self._is_running:
            try:
                item = await self._queue.get()
                if item is None:
                    break
                yield item
            except asyncio.CancelledError:
                break

    async def stream_encoded(
        self,
        codec: str = "flac",
        chunk_duration_us: int = 25_000,
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """Asynchronous generator yielding encoded audio frames from captured input.

        Args:
            codec: Audio codec to encode with ("flac", "opus", "pcm").
            chunk_duration_us: Target duration of encoded frames in microseconds.

        Yields:
            tuple[bytes, int]: Encoded audio frame and frame duration in microseconds.
        """
        encoder = create_encoder(
            codec,
            sample_rate=self._sample_rate,
            bit_depth=self._bit_depth,
            channels=self._channels,
            chunk_duration_us=chunk_duration_us,
        )
        async for pcm, ts in self.stream():
            for frame, duration_us in encoder.process(pcm, ts, 0):
                yield frame, duration_us

        for frame, duration_us in encoder.flush():
            yield frame, duration_us

    async def __aenter__(self) -> AudioCapture:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.stop()
