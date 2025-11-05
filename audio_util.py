from __future__ import annotations

import io
import base64
import asyncio
import threading
import logging
from typing import Callable, Awaitable

import numpy as np
import sounddevice as sd

from openai.resources.realtime.realtime import AsyncRealtimeConnection

logger = logging.getLogger(__name__)

CHUNK_LENGTH_S = 0.05  # 100ms
SAMPLE_RATE = 24000
CHANNELS = 1

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    """Convert audio bytes to PCM16 format. Requires pydub and pyaudioop."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio_to_pcm16_base64. "
            "Install with: pip install pydub pyaudioop"
        )
    
    # load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    print(f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}")
    # resample to 24kHz mono pcm16
    pcm_audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2).raw_data
    return pcm_audio


class AudioPlayerAsync:
    def __init__(self):
        logger.info("Initializing AudioPlayerAsync...")
        self.queue = []
        self.lock = threading.Lock()
        try:
            default_output = sd.default.device[1] if sd.default.device[1] is not None else "default"
            logger.info(f"Default output device: {default_output}")
            logger.info(f"Sample rate: {SAMPLE_RATE}, Channels: {CHANNELS}")
            
            self.stream = sd.OutputStream(
                callback=self.callback,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.int16,
                blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
            )
            logger.info("Audio output stream created")
        except Exception as e:
            logger.error(f"Error creating audio output stream: {e}", exc_info=True)
            raise
        self.playing = False
        self._frame_count = 0

    def callback(self, outdata, frames, time, status):  # noqa
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        with self.lock:
            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))
                logger.debug(f"Padding audio with {frames - len(data)} zeros")

        outdata[:] = data.reshape(-1, 1)
        logger.debug(f"Audio callback: wrote {len(data)} frames, queue size: {len(self.queue)}")

    def reset_frame_count(self):
        logger.info("Resetting frame count")
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16)
            logger.debug(f"Adding audio data: {len(np_data)} samples ({len(np_data)/SAMPLE_RATE*1000:.1f}ms)")
            self.queue.append(np_data)
            logger.debug(f"Queue size after add: {len(self.queue)}")
            if not self.playing:
                logger.info("Starting audio playback...")
                self.start()

    def start(self):
        try:
            self.playing = True
            self.stream.start()
            logger.info("Audio stream started successfully")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}", exc_info=True)
            self.playing = False

    def stop(self):
        logger.info("Stopping audio playback...")
        self.playing = False
        try:
            self.stream.stop()
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}", exc_info=True)
        with self.lock:
            self.queue = []

    def terminate(self):
        logger.info("Terminating audio stream...")
        try:
            self.stream.close()
        except Exception as e:
            logger.error(f"Error closing audio stream: {e}", exc_info=True)


async def send_audio_worker_sounddevice(
    connection: AsyncRealtimeConnection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
):
    sent_audio = False

    device_info = sd.query_devices()
    print(device_info)

    read_size = int(SAMPLE_RATE * 0.02)

    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
    )
    stream.start()

    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            data, _ = stream.read(read_size)

            if should_send() if should_send else True:
                if not sent_audio and start_send:
                    await start_send()
                await connection.send(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode("utf-8")}
                )
                sent_audio = True

            elif sent_audio:
                print("Done, triggering inference")
                await connection.send({"type": "input_audio_buffer.commit"})
                await connection.send({"type": "response.create", "response": {}})
                sent_audio = False

            await asyncio.sleep(0)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()