from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# RAG-related imports
from services.vectorstore import search_documents
from services.reranker import rerank_documents
from services.realtime_websocket import RealtimeWebSocketHandler
from routes import upload

from openai import AsyncOpenAI
import uvicorn
from dotenv import load_dotenv
import json
import os
import uuid
import logging

# Load environment variables
load_dotenv()

# Validate API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please create a .env file with your OpenAI API key or set it as an environment variable."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize OpenAI client for ephemeral token generation
openai_client = AsyncOpenAI(api_key=api_key)

# Get system instructions from environment variable (optional)
system_instructions = os.getenv("REALTIME_SYSTEM_INSTRUCTIONS", "")
enable_rag = os.getenv("ENABLE_RAG", "false").lower() == "true"

# Initialize Realtime WebSocket handler
realtime_handler = RealtimeWebSocketHandler(openai_client, system_instructions=system_instructions, enable_rag=enable_rag)

app = FastAPI(title="RAG + OpenAI Realtime API Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["upload"])

@app.get("/")
async def root():
    """Serve the main UI"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/realtime/client_secrets")
async def generate_client_secret():
    """
    Generate an ephemeral client token for direct client connections to Realtime API.
    This allows clients to connect directly using WebRTC without going through the relay.
    
    Returns:
        JSON with "value" field containing the ephemeral key (starts with "ek_")
    """
    try:
        response = await openai_client.realtime.client_secrets.create(
            session={
                "type": "realtime",
                "model": "gpt-realtime"
            }
        )
        return {"value": response.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating client secret: {str(e)}")


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint that relays messages between client and OpenAI Realtime API.
    Uses the same connection logic as the working push-to-talk app.
    """
    client_id = str(uuid.uuid4())
    await realtime_handler.handle_websocket(websocket, client_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

