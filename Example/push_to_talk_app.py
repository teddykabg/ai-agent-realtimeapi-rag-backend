#!/usr/bin/env python3
####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
# Run with: python push_to_talk_app.py                             #
#                                                                  #
# Make sure you have OPENAI_API_KEY set in your .env file or      #
# environment variables.                                           #
#                                                                  #
# Required dependencies:                                          #
#   pip install textual sounddevice numpy pyaudio pydub openai     #
#   python-dotenv                                                  #
####################################################################
from __future__ import annotations

import base64
import asyncio
import os
import logging
from typing import Any, cast, TYPE_CHECKING
from typing_extensions import override
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from textual import events
from audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
from textual.containers import Container

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection

# Type hints - only import Session for type checking
if TYPE_CHECKING:
    try:
        from openai.types.realtime.session import Session
    except ImportError:
        Session = Any
else:
    Session = Any


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please create a .env file with your OpenAI API key or set it as an environment variable."
            )
        
        # Initialize OpenAI client with API key from environment
        self.client = AsyncOpenAI(api_key=api_key)
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        logger.info("App mounted, starting workers...")
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

    async def handle_realtime_connection(self) -> None:
        logger.info("Connecting to OpenAI Realtime API...")
        try:
            async with self.client.realtime.connect(model="gpt-realtime") as conn:
                logger.info("Connected to OpenAI Realtime API")
                self.connection = conn
                self.connected.set()
                logger.info("Connection event set, updating session...")

                # note: this is the default and can be omitted
                # if you want to manually handle VAD yourself, then set `'turn_detection': None`
                await conn.session.update(
                    session={
                        "audio": {
                            "input": {"turn_detection": {"type": "server_vad"}},
                        },
                        "model": "gpt-realtime",
                        "type": "realtime",
                    }
                )
                logger.info("Session updated with server_vad configuration")

                acc_items: dict[str, Any] = {}

                logger.info("Starting event loop...")
                async for event in conn:
                    logger.debug(f"Received event: {event.type}")
                    
                    if event.type == "session.created":
                        logger.info(f"Session created: {event.session.id}")
                        self.session = event.session
                        session_display = self.query_one(SessionDisplay)
                        assert event.session.id is not None
                        session_display.session_id = event.session.id
                        continue

                    if event.type == "session.updated":
                        logger.info("Session updated")
                        self.session = event.session
                        continue

                    # Handle audio output - check for both possible event names
                    if event.type == "response.output_audio.delta":
                        logger.debug(f"Received audio delta, item_id: {event.item_id}, delta length: {len(event.delta)}")
                        if event.item_id != self.last_audio_item_id:
                            logger.info(f"New audio item started: {event.item_id}")
                            self.audio_player.reset_frame_count()
                            self.last_audio_item_id = event.item_id

                        try:
                            bytes_data = base64.b64decode(event.delta)
                            logger.debug(f"Decoded audio data: {len(bytes_data)} bytes")
                            self.audio_player.add_data(bytes_data)
                            logger.debug("Audio data added to player queue")
                        except Exception as e:
                            logger.error(f"Error processing audio delta: {e}", exc_info=True)
                        continue

                    # Handle transcript - check for both possible event names
                    # Similar to text output, transcripts might use different event names
                    if event.type == "response.output_audio_transcript.delta":
                        # Try to get delta from event
                        delta = getattr(event, 'delta', '')
                        if not delta and hasattr(event, 'text'):
                            delta = event.text
                        
                        if delta:
                            logger.debug(f"Received transcript delta: {delta[:50]}...")
                            try:
                                text = acc_items[event.item_id]
                            except KeyError:
                                acc_items[event.item_id] = delta
                                logger.info(f"New transcript item started: {event.item_id}")
                            else:
                                acc_items[event.item_id] = text + delta

                            # Clear and update the entire content
                            bottom_pane = self.query_one("#bottom-pane", RichLog)
                            bottom_pane.clear()
                            bottom_pane.write(acc_items[event.item_id])
                        continue
                    
                    # Also check for text output events (in case audio mode also sends text)
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, 'delta', '')
                        if delta:
                            logger.debug(f"Received text delta (from audio response): {delta[:50]}...")
                            try:
                                text = acc_items.get(event.item_id, '')
                                acc_items[event.item_id] = text + delta
                            except (KeyError, AttributeError):
                                acc_items[event.item_id] = delta
                            
                            bottom_pane = self.query_one("#bottom-pane", RichLog)
                            bottom_pane.clear()
                            bottom_pane.write(acc_items[event.item_id])
                        continue
                    
                    # Log any other event types to help debug
                    if event.type not in [
                        "session.created", "session.updated", 
                        "response.output_audio.delta", "response.output_audio_transcript.delta",
                        "response.output_text.delta", "response.output_text.done",
                        "response.done", "response.created", "rate_limits.updated"
                    ]:
                        logger.info(f"Unhandled event type: {event.type}")
        except Exception as e:
            logger.error(f"Error in realtime connection: {e}", exc_info=True)
            raise

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        logger.info("Initializing microphone input stream...")
        sent_audio = False

        device_info = sd.query_devices()
        logger.info(f"Available audio devices:\n{device_info}")
        logger.info(f"Default input device: {sd.default.device[0]}")

        read_size = int(SAMPLE_RATE * 0.02)
        logger.info(f"Audio read size: {read_size} samples ({read_size/SAMPLE_RATE*1000:.1f}ms)")

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()
        logger.info("Microphone stream started")

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                if not status_indicator.is_recording:
                    logger.info("Recording started (K key pressed)")
                status_indicator.is_recording = True

                data, _ = stream.read(read_size)
                logger.debug(f"Read audio chunk: {len(data)} samples")

                connection = await self._get_connection()
                if not sent_audio:
                    logger.info("Cancelling any existing response and starting new audio input")
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                audio_base64 = base64.b64encode(cast(Any, data)).decode("utf-8")
                await connection.input_audio_buffer.append(audio=audio_base64)
                logger.debug(f"Sent audio chunk to OpenAI: {len(audio_base64)} base64 chars")

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            pass
        except Exception as e:
            logger.error(f"Error in send_mic_audio: {e}", exc_info=True)
        finally:
            logger.info("Stopping microphone stream...")
            stream.stop()
            stream.close()
            logger.info("Microphone stream closed")

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                logger.info("Stopping recording (K key pressed again)")
                self.should_send_audio.clear()
                status_indicator.is_recording = False

                # Check if we're in manual turn_detection mode (not server_vad)
                # If turn_detection is None, we need to manually commit and create response
                turn_detection = None
                if self.session:
                    try:
                        # turn_detection is nested under audio.input
                        if hasattr(self.session, 'audio') and self.session.audio:
                            if hasattr(self.session.audio, 'input') and self.session.audio.input:
                                turn_detection = getattr(self.session.audio.input, 'turn_detection', None)
                                logger.debug(f"Turn detection type: {type(turn_detection)}")
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Error checking turn_detection: {e}")
                
                if turn_detection is None:
                    # Manual turn_detection mode - need to manually commit and create response
                    logger.info("Manual turn detection mode - committing audio and creating response")
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
                else:
                    logger.info("Server VAD mode - turn detection handled automatically")
            else:
                logger.info("Starting recording (K key pressed)")
                self.should_send_audio.set()
                status_indicator.is_recording = True


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()