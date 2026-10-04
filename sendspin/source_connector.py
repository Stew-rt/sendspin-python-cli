"""Source connector for bridging local audio capture to a Sendspin client (source@v1)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerCommandPayload
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec

from sendspin.audio_devices import AudioDevice
from sendspin.audio_input import AudioCapture
from sendspin.utils import create_task

if TYPE_CHECKING:
    from aiosendspin.client import AudioFormat, SendspinClient
    from aiosendspin.client.source import SourceCapture
    from aiosendspin.clock import Clock

logger = logging.getLogger(__name__)


class SourceHandler:
    """Bridges between local AudioCapture and SendspinClient source streaming.

    Handles server commands ('start' and 'stop' streaming), manages the lifecycle
    of audio capture and frame encoding/dispatching via SourceCapture, and coordinates
    disconnections and state callbacks.
    """

    def __init__(
        self,
        audio_device: AudioDevice | None = None,
        codec: AudioCodec | str = AudioCodec.FLAC,
        sample_rate: int = 48000,
        channels: int = 2,
        bit_depth: int = 16,
        block_size: int = 1024,
        latency: float | str | None = "low",
        clock: Clock | None = None,
        on_state_change: Callable[[bool], None] | None = None,
        on_event: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the source handler.

        Args:
            audio_device: Audio input device to capture from (default input if None).
            codec: Audio codec to use for stream encoding (FLAC, Opus, PCM).
            sample_rate: Audio sample rate in Hz (e.g. 44100, 48000).
            channels: Number of audio channels (1=mono, 2=stereo).
            bit_depth: Bits per sample (16, 24, 32).
            block_size: Block size in samples per capture buffer.
            latency: Requested capture latency ("low", "high", or seconds).
            clock: Monotonic clock protocol instance for timestamping.
            on_state_change: Callback invoked when streaming state changes (True=streaming, False=stopped).
            on_event: Callback invoked on stream lifecycle events ("start", "stop").
            on_error: Callback invoked when capture or streaming errors occur.
        """
        self._audio_device = audio_device
        self._current_format: AudioFormat | None = None
        self._codec = AudioCodec(codec) if isinstance(codec, str) else codec
        self._sample_rate = sample_rate
        self._channels = channels
        self._bit_depth = bit_depth
        self._block_size = block_size
        self._latency = latency
        self._clock = clock

        self._on_state_change = on_state_change
        self._on_event = on_event
        self._on_error = on_error

        self._client: SendspinClient | None = None
        self._unsubscribes: list[Callable[[], None]] = []

        self._audio_capture: AudioCapture | None = None
        self._source_capture: SourceCapture | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._is_streaming = False
        self._lock = asyncio.Lock()

    @property
    def is_streaming(self) -> bool:
        """Whether audio capture and streaming are currently active."""
        return self._is_streaming

    @property
    def audio_device(self) -> AudioDevice | None:
        """The configured audio input device."""
        return self._audio_device

    @property
    def codec(self) -> AudioCodec:
        """The configured audio stream codec."""
        return self._codec

    @property
    def sample_rate(self) -> int:
        """The configured sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """The configured channel count."""
        return self._channels

    @property
    def bit_depth(self) -> int:
        """The configured bit depth."""
        return self._bit_depth

    @property
    def audio_capture(self) -> AudioCapture | None:
        """The active AudioCapture engine instance, if initialized."""
        return self._audio_capture

    @property
    def source_capture(self) -> SourceCapture | None:
        """The active SourceCapture protocol instance, if streaming."""
        return self._source_capture

    @property
    def overflow_count(self) -> int:
        """Total number of PortAudio input buffer overflows detected."""
        return self._audio_capture.overflow_count if self._audio_capture else 0

    def attach_client(self, client: SendspinClient) -> None:
        """Attach to a SendspinClient and register listeners."""
        self._client = client
        self._unsubscribes = [
            client.add_server_command_listener(self._on_server_command),
            client.add_disconnect_listener(self._on_disconnect),
        ]

    def detach_client(self) -> None:
        """Detach from the client and unregister listeners."""
        for unsub in self._unsubscribes:
            unsub()
        self._unsubscribes = []
        self._client = None

    def _on_server_command(self, payload: ServerCommandPayload) -> None:
        """Handle incoming server commands targeting the source role."""
        if payload.source is None:
            return

        command = payload.source.command
        if command == "start":
            logger.info("Received server command to start source streaming")
            create_task(self.start_streaming())
        elif command == "stop":
            logger.info("Received server command to stop source streaming")
            create_task(self.stop_streaming())

    def _on_disconnect(self) -> None:
        """Handle client disconnect event."""
        create_task(self.handle_disconnect())

    async def handle_disconnect(self) -> None:
        """Reset streaming state upon client disconnection."""
        logger.debug("Client disconnected; stopping source streaming")
        await self.stop_streaming()

    async def start_streaming(self) -> None:
        """Start capturing and streaming audio to the Sendspin server."""
        async with self._lock:
            if self._is_streaming:
                logger.debug("Source streaming already active, ignoring start request")
                return

            if self._client is None or not self._client.connected:
                logger.warning("Cannot start source streaming: client is not connected")
                return

            logger.info(
                "Starting source streaming (%s, %d Hz, %d ch, %d-bit)",
                self._codec.value,
                self._sample_rate,
                self._channels,
                self._bit_depth,
            )

            try:
                audio_format = SupportedAudioFormat(
                    codec=self._codec,
                    channels=self._channels,
                    sample_rate=self._sample_rate,
                    bit_depth=self._bit_depth,
                )
                self._source_capture = self._client.create_source_capture(audio_format)
                await self._source_capture.start()

                capture_clock = self._clock or getattr(self._client, "_clock", None)
                self._audio_capture = AudioCapture(
                    device=self._audio_device,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    bit_depth=self._bit_depth,
                    block_size=self._block_size,
                    latency=self._latency,
                    clock=capture_clock,
                )
                await self._audio_capture.start()

                self._is_streaming = True
                self._stream_task = create_task(
                    self._stream_loop(self._audio_capture, self._source_capture),
                    name="sendspin-source-stream-loop",
                )

                if self._on_state_change:
                    self._on_state_change(True)
                if self._on_event:
                    self._on_event("start")

            except Exception as exc:
                logger.exception("Failed to start source streaming: %s", exc)
                self._is_streaming = False
                await self._cleanup()
                if self._on_error:
                    self._on_error(exc)

    async def stop_streaming(self) -> None:
        """Stop capturing and streaming audio."""
        async with self._lock:
            if not self._is_streaming:
                return

            logger.info("Stopping source streaming")
            self._is_streaming = False

            if self._stream_task is not None:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._stream_task = None

            await self._cleanup()

            if self._on_state_change:
                self._on_state_change(False)
            if self._on_event:
                self._on_event("stop")

    async def _cleanup(self) -> None:
        """Clean up audio capture and source capture resources."""
        if self._audio_capture is not None:
            try:
                await self._audio_capture.stop()
            except Exception as e:
                logger.debug("Error stopping audio capture: %s", e)
            finally:
                self._audio_capture = None

        if self._source_capture is not None:
            try:
                await self._source_capture.stop()
            except Exception as e:
                logger.debug("Error stopping source capture: %s", e)
            finally:
                self._source_capture = None

    async def _stream_loop(
        self,
        audio_capture: AudioCapture,
        source_capture: SourceCapture,
    ) -> None:
        """Capture PCM chunks from AudioCapture and feed them to SourceCapture."""
        try:
            async for pcm, timestamp_us in audio_capture.stream():
                if not self._is_streaming:
                    break
                await source_capture.feed(pcm, timestamp_us)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error in source streaming loop: %s", exc)
            if self._on_error:
                self._on_error(exc)
        finally:
            if self._is_streaming:
                # Unexpected termination of the stream loop
                create_task(self.stop_streaming())

    async def shutdown(self) -> None:
        """Shut down the source handler, detach client, and release all resources."""
        self.detach_client()
        await self.stop_streaming()
