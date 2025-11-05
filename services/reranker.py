import cohere
import os
from typing import List

# Initialize Cohere client
cohere_client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY"))

async def rerank_documents(query: str, documents: List[str], top_k: int = 5) -> List[str]:
    """
    Rerank documents using Cohere rerank API.
    """
    try:
        if not documents:
            return []
        
        response = await cohere_client.rerank(
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

