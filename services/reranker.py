import cohere
import os
from typing import List, Optional

# Lazy initialization - client created only when needed
_cohere_client: Optional[cohere.AsyncClient] = None

def _get_cohere_client() -> Optional[cohere.AsyncClient]:
    """Get or create Cohere client (lazy initialization)."""
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            _cohere_client = cohere.AsyncClient(api_key=api_key)
    return _cohere_client

async def rerank_documents(query: str, documents: List[str], top_k: int = 5) -> List[str]:
    """
    Rerank documents using Cohere rerank API.
    """
    try:
        if not documents:
            return []
        
        client = _get_cohere_client()
        if not client:
            # No API key, return original documents
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
        
        return reranked_docs
    
    except Exception as e:
        # If reranking fails, return original documents (fallback)
        print(f"Reranking error: {str(e)}")
        return documents[:top_k]

