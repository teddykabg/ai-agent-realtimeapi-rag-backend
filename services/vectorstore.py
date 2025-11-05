from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
from typing import List, Dict, Any

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")

client = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

async def ensure_collection():
    """
    Ensure the collection exists, create if it doesn't.
    """
    try:
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimension
                    distance=Distance.COSINE
                )
            )
    except Exception as e:
        raise Exception(f"Error ensuring collection: {str(e)}")

async def store_documents(chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> List[str]:
    """
    Store documents (chunks) with their embeddings in Qdrant.
    """
    try:
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
        
        # Batch upload points
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        return document_ids
    
    except Exception as e:
        raise Exception(f"Error storing documents: {str(e)}")

async def search_documents(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using query embedding.
    """
    try:
        from services.embeddings import generate_embeddings
        
        await ensure_collection()
        
        # Generate embedding for query
        query_embedding = await generate_embeddings([query])
        query_vector = query_embedding[0]
        
        # Search in Qdrant
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": str(result.id),
                "chunk": result.payload.get("chunk", ""),
                "score": float(result.score),
                "metadata": {k: v for k, v in result.payload.items() if k != "chunk"}
            })
        
        return formatted_results
    
    except Exception as e:
        raise Exception(f"Error searching documents: {str(e)}")

