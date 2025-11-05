"""WebSocket handler for OpenAI Realtime API relay."""
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection

logger = logging.getLogger(__name__)


class RealtimeWebSocketHandler:
    """Handles WebSocket connections and relays to OpenAI Realtime API."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.active_connections: Dict[str, Any] = {}
    
    async def handle_websocket(self, websocket: WebSocket, client_id: str):
        """Handle a WebSocket connection and relay to OpenAI Realtime API."""
        await websocket.accept()
        logger.info(f"WebSocket connection accepted: {client_id}")
        
        realtime_task = None
        client_task = None
        
        try:
            # Connect to OpenAI Realtime API
            async with self.openai_client.realtime.connect(model="gpt-realtime") as conn:
                logger.info(f"Connected to OpenAI Realtime API for client {client_id}")
                
                # Update session configuration
                await conn.session.update(
                    session={
                        "audio": {
                            "input": {"turn_detection": {"type": "server_vad"}},
                        },
                        "model": "gpt-realtime",
                        "type": "realtime",
                    }
                )
                logger.info(f"Session configured for client {client_id}")
                
                # Start task to forward events from OpenAI to WebSocket
                realtime_task = asyncio.create_task(
                    self._forward_realtime_events(conn, websocket, client_id)
                )
                
                # Start task to forward messages from WebSocket to OpenAI
                client_task = asyncio.create_task(
                    self._forward_client_messages(conn, websocket, client_id)
                )
                
                # Wait for session to be ready (via realtime_task)
                # Then wait for either task to complete
                done, pending = await asyncio.wait(
                    [realtime_task, client_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
        except Exception as e:
            logger.error(f"Error in Realtime API connection for client {client_id}: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "error", "message": f"Connection error: {str(e)}"})
            except:
                pass
        finally:
            logger.info(f"Connection closed for client {client_id}")
    
    async def _forward_realtime_events(self, conn: AsyncRealtimeConnection, websocket: WebSocket, client_id: str):
        """Forward events from OpenAI Realtime API to WebSocket client."""
        try:
            async for event in conn:
                # Convert event to JSON-serializable format
                event_data = self._event_to_dict(event)
                
                # Send to WebSocket client
                try:
                    await websocket.send_json(event_data)
                    logger.debug(f"Sent event to client {client_id}: {event.type}")
                except Exception as e:
                    logger.error(f"Error sending event to client {client_id}: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"Event forwarding cancelled for client {client_id}")
        except Exception as e:
            logger.error(f"Error forwarding events for client {client_id}: {e}", exc_info=True)
    
    async def _forward_client_messages(self, conn: AsyncRealtimeConnection, websocket: WebSocket, client_id: str):
        """Forward messages from WebSocket client to OpenAI Realtime API."""
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    logger.debug(f"Received from client {client_id}: {message.get('type')}")
                    
                    # Forward to OpenAI Realtime API
                    await conn.send(message)
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected: {client_id}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client {client_id}: {e}")
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                except Exception as e:
                    logger.error(f"Error handling message from client {client_id}: {e}", exc_info=True)
                    await websocket.send_json({"type": "error", "message": str(e)})
        except asyncio.CancelledError:
            logger.info(f"Client message forwarding cancelled for client {client_id}")
        except Exception as e:
            logger.error(f"Error forwarding client messages for client {client_id}: {e}", exc_info=True)
    
    def _event_to_dict(self, event) -> Dict[str, Any]:
        """Convert OpenAI event to dictionary."""
        try:
            # Use model_dump if available (Pydantic v2)
            if hasattr(event, 'model_dump'):
                return event.model_dump(exclude_none=True)
            # Use dict() if available (Pydantic v1)
            elif hasattr(event, 'dict'):
                return event.dict(exclude_none=True)
            # Fallback: try to get attributes
            else:
                result = {"type": getattr(event, 'type', 'unknown')}
                
                # Add common attributes
                for attr in ['delta', 'item_id', 'response_id', 'event_id', 'item', 'part', 'session', 'text', 'transcript']:
                    if hasattr(event, attr):
                        value = getattr(event, attr)
                        # Handle nested objects
                        if hasattr(value, 'model_dump'):
                            result[attr] = value.model_dump(exclude_none=True)
                        elif hasattr(value, 'dict'):
                            result[attr] = value.dict(exclude_none=True)
                        else:
                            result[attr] = value
                
                return result
        except Exception as e:
            logger.error(f"Error converting event to dict: {e}")
            return {"type": getattr(event, 'type', 'unknown'), "error": str(e)}

