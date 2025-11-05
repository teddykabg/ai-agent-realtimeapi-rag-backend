from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")

# Log Qdrant configuration (mask API key for security)
api_key_preview = f"{QDRANT_API_KEY[:8]}..." if QDRANT_API_KEY and len(QDRANT_API_KEY) > 8 else ("SET" if QDRANT_API_KEY else "NOT SET")
logger.info(f"Qdrant Configuration - URL: {QDRANT_URL}, API Key: {api_key_preview}, Collection: {COLLECTION_NAME}")

client = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

async def ensure_collection():
    """
    Ensure the collection exists, create if it doesn't.
    """
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_URL} to ensure collection '{COLLECTION_NAME}' exists...")
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]
        logger.info(f"Existing collections: {collection_names}")
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Collection '{COLLECTION_NAME}' not found, creating it...")
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection '{COLLECTION_NAME}' created successfully")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")
    except Exception as e:
        logger.error(f"Failed to ensure collection '{COLLECTION_NAME}' at {QDRANT_URL}: {str(e)}")
        raise Exception(f"Error ensuring collection: {str(e)}")

async def store_documents(chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> List[str]:
    """
    Store documents (chunks) with their embeddings in Qdrant.
    """
    try:
        logger.info(f"Preparing to store {len(chunks)} documents in collection '{COLLECTION_NAME}'...")
        await ensure_collection()
        
        points = []
        document_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = str(uuid.uuid4())
            document_ids.append(doc_id)
            
            point_metadata = {
                "chunk": chunk,
                "chunk_index": i,
                **(metadata or {})
            }
            
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=point_metadata
                )
            )
        
        logger.info(f"Upserting {len(points)} points to Qdrant collection '{COLLECTION_NAME}' in batches...")
        # Batch upload points in chunks of 50 to avoid timeout
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await client.upsert(collection_name=COLLECTION_NAME, points=batch)
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
        logger.info(f"Successfully stored {len(document_ids)} documents in Qdrant")
        
        return document_ids
    
    except Exception as e:
        logger.error(f"Error storing documents in Qdrant: {str(e)}", exc_info=True)
        raise Exception(f"Error storing documents: {str(e)}")

async def search_documents(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using query embedding.
    """
    try:
        from services.embeddings import generate_embeddings
        
        logger.info(f"Searching documents for query: '{query[:100]}{'...' if len(query) > 100 else ''}' (top_k={top_k})")
        await ensure_collection()
        
        # Generate embedding for query
        logger.debug("Generating query embedding...")
        query_embedding = await generate_embeddings([query])
        query_vector = query_embedding[0]
        logger.debug(f"Query embedding generated (dimension: {len(query_vector)})")
        
        # Search in Qdrant
        logger.info(f"Searching in Qdrant collection '{COLLECTION_NAME}'...")
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        logger.info(f"Found {len(results)} results from Qdrant")
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "id": str(result.id),
                "chunk": result.payload.get("chunk", ""),
                "score": float(result.score),
                "metadata": {k: v for k, v in result.payload.items() if k != "chunk"}
            })
            logger.debug(f"Result {i+1}: score={result.score:.4f}, chunk_length={len(result.payload.get('chunk', ''))}")
        
        if formatted_results:
            logger.info(f"Search completed: {len(formatted_results)} results returned (top score: {formatted_results[0]['score']:.4f})")
        else:
            logger.warning("No search results found")
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        raise Exception(f"Error searching documents: {str(e)}")

