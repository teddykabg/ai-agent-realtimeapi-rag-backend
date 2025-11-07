"""WebSocket handler for OpenAI Realtime API relay."""
import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RealtimeWebSocketHandler:
    """Handles WebSocket connections and relays to OpenAI Realtime API."""
    
    def __init__(self, openai_client: AsyncOpenAI, system_instructions: Optional[str] = None, enable_rag: bool = False):
        self.openai_client = openai_client
        self.active_connections: Dict[str, Any] = {}
        self.system_instructions = system_instructions or os.getenv("REALTIME_SYSTEM_INSTRUCTIONS", "")
        self.enable_rag = enable_rag or os.getenv("ENABLE_RAG", "false").lower() == "true"
        if self.enable_rag:
            logger.info("RAG is ENABLED - will inject context from knowledge base")
        else:
            logger.info("RAG is DISABLED - using standard Realtime API mode")
    
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
                session_config = {
                    "type":"realtime",
                    "model": "gpt-realtime",
                    "audio":{
                        "input":{
                            "turn_detection": {
                                "type": "server_vad"
                            },
                            "transcription":{
                                "model": "gpt-4o-mini-transcribe",
                                "prompt":"",
                                "language": "en"
                            },
                            "noise_reduction": {
                                "type": "near_field",
                            },
                        },
                        "output":{
                            "speed": 0.9,
                            "voice":"cedar"
                        }
                    },   
                }
                
                # Add system instructions if provided
                if self.system_instructions:
                    session_config["instructions"] = self.system_instructions
                    logger.info(f"System instructions set for client {client_id}")
                
                await conn.session.update(session=session_config)
                logger.info(f"Session configured for client {client_id} (input_audio_transcription enabled)")
                
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
                # Log all events for debugging
                event_type = getattr(event, 'type', 'unknown')
                logger.info(f"[EVENT] Received event type: '{event_type}' for client {client_id}")
                
                # Detailed logging for error events
                if event_type == "error":
                    error_message = getattr(event, 'message', None)
                    error_code = getattr(event, 'code', None)
                    error_param = getattr(event, 'param', None)
                    error_type = getattr(event, 'type', None)
                    event_id = getattr(event, 'event_id', None)
                    
                    # Get all available error attributes
                    error_attrs = {attr: getattr(event, attr, None) for attr in dir(event) if not attr.startswith('_') and not callable(getattr(event, attr, None))}
                    
                    logger.error(f"[ERROR] Error event received for client {client_id}:")
                    logger.error(f"[ERROR]   - Message: {error_message}")
                    logger.error(f"[ERROR]   - Code: {error_code}")
                    logger.error(f"[ERROR]   - Param: {error_param}")
                    logger.error(f"[ERROR]   - Type: {error_type}")
                    logger.error(f"[ERROR]   - Event ID: {event_id}")
                    logger.error(f"[ERROR]   - All attributes: {list(error_attrs.keys())}")
                    logger.error(f"[ERROR]   - Full error data: {error_attrs}")
                
                # RAG: Intercept user transcript completion and inject context
                if self.enable_rag:
                    logger.debug(f"[RAG] Checking RAG trigger for event type: '{event_type}'")
                    
                    # Check for audio transcript completion (both variants)
                    if event_type == "conversation.item.input_audio_transcript.completed" or \
                       event_type == "conversation.item.input_audio_transcription.completed":
                        logger.info(f"[RAG] Matched audio transcript event type for client {client_id}")
                        transcript = getattr(event, 'transcript', None) or getattr(event, 'text', None)
                        logger.debug(f"[RAG] Transcript attribute value: {transcript is not None} (type: {type(transcript)})")
                        if transcript:
                            logger.info(f"[RAG] Audio transcript completed for client {client_id}: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
                            await self._inject_rag_context(conn, transcript, client_id)
                        else:
                            logger.warning(f"[RAG] Audio transcript event received but transcript is None or empty")
                    else:
                        # Log if this was a transcript-related event but with different type
                        if "transcript" in event_type.lower() or "input_audio" in event_type.lower():
                            logger.info(f"[RAG] Received transcript-related event but type mismatch: '{event_type}' (expected: 'conversation.item.input_audio_transcript.completed')")
                            # Try to get transcript anyway to see what's available
                            transcript = getattr(event, 'transcript', None)
                            logger.debug(f"[RAG] Available attributes: {[attr for attr in dir(event) if not attr.startswith('_')]}")
                            if transcript:
                                logger.info(f"[RAG] Found transcript in alternate event type: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
                                await self._inject_rag_context(conn, transcript, client_id)
                
                # Convert event to JSON-serializable format
                event_data = self._event_to_dict(event)
                
                # Send to WebSocket client
                try:
                    await websocket.send_json(event_data)
                    logger.debug(f"Sent event to client {client_id}: {event_type}")
                except Exception as e:
                    logger.error(f"Error sending event to client {client_id}: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"Event forwarding cancelled for client {client_id}")
        except Exception as e:
            logger.error(f"Error forwarding events for client {client_id}: {e}", exc_info=True)
    
    async def _inject_rag_context(self, conn: AsyncRealtimeConnection, query: str, client_id: str):
        """Retrieve relevant documents and inject as context."""
        try:
            logger.info(f"[RAG] Starting context retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
            from services.vectorstore import search_documents
            from services.reranker import rerank_documents
            
            # Search for relevant documents
            results = await search_documents(query, top_k=5)
            if not results:
                logger.warning(f"[RAG] No documents found in knowledge base for query")
                return
            
            logger.info(f"[RAG] Found {len(results)} relevant documents (top score: {results[0]['score']:.4f})")
            
            chunks = [r['chunk'] for r in results]
            original_chunk_count = len(chunks)
            
            # Rerank if Cohere API key is available
            if os.getenv("COHERE_API_KEY"):
                logger.info(f"[RAG] Reranking {len(chunks)} documents using Cohere...")
                chunks = await rerank_documents(query, chunks, top_k=3)
                logger.info(f"[RAG] Reranked to {len(chunks)} top documents")
            else:
                logger.info(f"[RAG] Cohere API key not set - using top {min(3, len(chunks))} documents without reranking")
            
            # Use top 3 chunks for context
            context_chunks = chunks[:3]
            context = "\n\n".join(context_chunks)
            context_length = len(context)
            
            logger.info(f"[RAG] Injecting context ({len(context_chunks)} chunks, {context_length} chars) into conversation for client {client_id}")
            
            await conn.send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": f"Context from knowledge base:\n{context}\n\nUser question: {query}"
                }
            })
            logger.info(f"[RAG] Successfully injected RAG context for client {client_id}")
        except Exception as e:
            logger.error(f"[RAG] RAG injection failed for client {client_id}: {e}", exc_info=True)
    
    async def _forward_client_messages(self, conn: AsyncRealtimeConnection, websocket: WebSocket, client_id: str):
        """Forward messages from WebSocket client to OpenAI Realtime API."""
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    logger.debug(f"Received from client {client_id}: {message.get('type')}")
                    
                    # RAG: Inject context for text messages before forwarding
                    if self.enable_rag:
                        message_type = message.get("type")
                        logger.debug(f"[RAG] Checking text message trigger for message type: '{message_type}'")
                        
                        if message_type == "conversation.item.create":
                            logger.debug(f"[RAG] Matched conversation.item.create message type")
                            item = message.get("item", {})
                            item_type = item.get("type")
                            item_role = item.get("role")
                            logger.debug(f"[RAG] Item type: '{item_type}', role: '{item_role}'")
                            
                            if item_type == "message" and item_role == "user":
                                content = item.get("content")
                                logger.debug(f"[RAG] Content type: {type(content)}, is string: {isinstance(content, str)}")
                                if isinstance(content, str):
                                    logger.info(f"[RAG] Text message received for client {client_id}: '{content[:100]}{'...' if len(content) > 100 else ''}'")
                                    await self._inject_rag_context(conn, content, client_id)
                                else:
                                    logger.warning(f"[RAG] Content is not a string: {type(content)}")
                            else:
                                logger.debug(f"[RAG] Message item not matching criteria (type='{item_type}', role='{item_role}')")
                        else:
                            logger.debug(f"[RAG] Message type '{message_type}' does not match 'conversation.item.create'")
                    
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

