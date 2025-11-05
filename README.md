# RAG + OpenAI Realtime API Backend

A FastAPI backend that implements Retrieval-Augmented Generation (RAG) with OpenAI Realtime API streaming, Qdrant vector store, and Cohere reranking.

## Features

- **File Upload & Processing**: Upload documents, chunk them, generate embeddings, and store in Qdrant
- **RAG Chat**: Query documents with retrieval, reranking, and streaming responses
- **WebSocket Support**: Real-time streaming chat via WebSocket
- **Vector Search**: Semantic search using Qdrant vector database
- **Reranking**: Improve retrieval quality with Cohere reranking API

## Project Structure

```
./
├─ main.py                   # FastAPI entrypoint
├─ routes/
│   ├─ upload.py             # file ingestion endpoint
│   └─ chat.py               # chat/retrieval endpoint
├─ services/
│   ├─ embeddings.py         # OpenAI embeddings
│   ├─ vectorstore.py        # Qdrant operations
│   ├─ reranker.py           # Cohere reranking
│   └─ openai_stream.py      # Realtime API streaming logic
├─ utils/
│   └─ chunking.py           # text chunking helper
├─ requirements.txt
└─ .env
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Fill in your API keys:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `COHERE_API_KEY`: Your Cohere API key
     - `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
     - `QDRANT_API_KEY`: Optional, for cloud Qdrant
     - `QDRANT_COLLECTION_NAME`: Collection name (default: documents)

3. **Start Qdrant** (if using local instance):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```
   Or with uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### POST `/upload`
Upload a file for processing and storage.

**Request**: Multipart form data with `file` field

**Response**:
```json
{
  "message": "File uploaded and processed successfully",
  "filename": "example.txt",
  "chunks_processed": 10,
  "document_ids": ["uuid1", "uuid2", ...]
}
```

### POST `/chat`
Query the RAG system and get streaming response.

**Request**:
```json
{
  "query": "What is the main topic?",
  "top_k": 10,
  "rerank_top_k": 5
}
```

**Response**: Server-Sent Events (SSE) stream with chunks:
```
data: {"content": "chunk1"}
data: {"content": "chunk2"}
data: [DONE]
```

### WebSocket `/ws/chat`
Real-time chat via WebSocket.

**Send**:
```json
{
  "query": "What is the main topic?",
  "top_k": 10,
  "rerank_top_k": 5
}
```

**Receive**:
```json
{"content": "chunk", "done": false}
...
{"content": "", "done": true}
```

## Usage Example

### Upload a document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.txt"
```

### Query via POST
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 10, "rerank_top_k": 5}'
```

### Query via WebSocket (Python example)
```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/ws/chat"
    async with websockets.connect(uri) as websocket:
        message = {
            "query": "What is this document about?",
            "top_k": 10,
            "rerank_top_k": 5
        }
        await websocket.send(json.dumps(message))
        
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("done"):
                break
            print(data.get("content", ""), end="", flush=True)

asyncio.run(chat())
```

## Configuration

- **Embedding Model**: `text-embedding-3-small` (OpenAI)
- **Reranking Model**: `rerank-english-v3.0` (Cohere)
- **Chat Model**: `gpt-4` (OpenAI, configurable)
- **Chunk Size**: 1000 characters (configurable in `utils/chunking.py`)
- **Chunk Overlap**: 200 characters (configurable in `utils/chunking.py`)

## License

MIT

