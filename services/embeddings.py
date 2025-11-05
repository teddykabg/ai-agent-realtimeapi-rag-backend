from openai import AsyncOpenAI
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lazy initialization - client created on first use
_client = None

def get_client():
    """Get or create OpenAI client (lazy initialization)"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client

async def generate_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI.
    """
    try:
        client = get_client()
        response = await client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

