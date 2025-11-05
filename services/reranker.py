import cohere
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy initialization - client created only when needed
_cohere_client: Optional[cohere.AsyncClient] = None

def _get_cohere_client() -> Optional[cohere.AsyncClient]:
    """Get or create Cohere client (lazy initialization)."""
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            _cohere_client = cohere.AsyncClient(api_key=api_key)
            logger.info("Cohere rerank client initialized")
        else:
            logger.info("COHERE_API_KEY not set - reranking will be skipped")
    return _cohere_client

async def rerank_documents(query: str, documents: List[str], top_k: int = 5) -> List[str]:
    """
    Rerank documents using Cohere rerank API.
    """
    try:
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        logger.info(f"Reranking {len(documents)} documents for query (top_k={top_k})")
        client = _get_cohere_client()
        if not client:
            # No API key, return original documents
            logger.info(f"Cohere API key not available - returning top {top_k} documents without reranking")
            return documents[:top_k]
        
        response = await client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents))
        )
        
        # Extract reranked documents in order
        reranked_docs = []
        for result in response.results:
            reranked_docs.append(documents[result.index])
        
        logger.info(f"Successfully reranked {len(reranked_docs)} documents")
        return reranked_docs
    
    except Exception as e:
        # If reranking fails, return original documents (fallback)
        logger.error(f"Reranking error: {str(e)} - falling back to original documents", exc_info=True)
        return documents[:top_k]

