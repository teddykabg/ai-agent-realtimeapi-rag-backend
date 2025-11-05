#!/usr/bin/env python3
"""Simple test script to verify OpenAI Realtime API connection."""
import asyncio
import os
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def test_realtime_connection():
    """Test connection to OpenAI Realtime API and send/receive a message."""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment variables!")
        logger.info("Please set it in your .env file or as an environment variable.")
        return False
    
    logger.info("✅ API key found")
    
    # Create client
    try:
        client = AsyncOpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"❌ Failed to create OpenAI client: {e}")
        return False
    
    # Test connection and send message
    logger.info("🔌 Connecting to Realtime API...")
    try:
        async with client.realtime.connect(model="gpt-realtime") as conn:
            logger.info("✅ Connected")
            
            # Update session with text output modality
            await conn.session.update(
                session={
                    "model": "gpt-realtime",
                    "type": "realtime",
                    "output_modalities": ["text"],  # Only text, no audio
                }
            )
            
            # Wait for session to be ready
            session_ready = False
            timeout = 10
            start_time = asyncio.get_event_loop().time()
            
            async for event in conn:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    logger.error("⏱️  Timeout waiting for session")
                    return False
                
                if event.type == "session.created":
                    session_ready = True
                    break
                elif event.type == "session.updated":
                    session_ready = True
                    break
            
            if not session_ready:
                logger.error("❌ Session not ready")
                return False
            
            # Send a text message
            test_message = "Say hello in one sentence."
            logger.info(f"📤 Sending: '{test_message}'")
            
            await conn.send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": test_message
                }
            })
            
            # Create a response
            await conn.send({
                "type": "response.create"
            })
            
            # Wait for response
            logger.info("⏳ Waiting for response...")
            response_text = ""
            response_received = False
            timeout = 30
            start_time = asyncio.get_event_loop().time()
            
            async for event in conn:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    logger.warning("⏱️  Timeout waiting for response")
                    break
                
                # Handle response events - focus on text extraction
                try:
                    if event.type == "response.output_text.delta":
                        # This is the correct event type for text output!
                        # The API sends response.output_text.delta (not response.text.delta)
                        if hasattr(event, 'delta') and event.delta:
                            response_text += event.delta
                    
                    elif event.type == "response.output_text.done":
                        # Response text is complete
                        response_received = True
                        break
                    
                    elif event.type == "response.done":
                        # Fallback: response done
                        if response_text:
                            response_received = True
                        break
                            
                except Exception as e:
                    logger.warning(f"Error processing event {event.type}: {e}")
            
            if response_received and response_text:
                print("\n" + "=" * 70)
                print(" " * 20 + "AI RESPONSE")
                print("=" * 70)
                print()
                print(response_text)
                print()
                print("=" * 70)
                logger.info("✅ Test PASSED - Received text response!")
                return True
            else:
                logger.warning(f"⚠️  No text response received. Partial text: {response_text[:100] if response_text else 'None'}")
                return False
            
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}", exc_info=True)
        return False


async def main():
    """Main function."""
    print("=" * 60)
    print("OpenAI Realtime API Connection Test")
    print("=" * 60)
    print()
    
    success = await test_realtime_connection()
    
    print()
    print("=" * 60)
    if success:
        print("✅ TEST PASSED - Connection is working!")
    else:
        print("❌ TEST FAILED - Check the errors above")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

