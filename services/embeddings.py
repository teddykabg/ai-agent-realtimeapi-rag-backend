from openai import AsyncOpenAI
import os
import logging
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

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
        logger.info("OpenAI embeddings client initialized")
    return _client

async def generate_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI.
    """
    try:
        logger.info(f"Generating embeddings for {len(texts)} text(s) using model: {model}")
        client = get_client()
        response = await client.embeddings.create(
            model=model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Successfully generated {len(embeddings)} embedding(s) with dimension {len(embeddings[0]) if embeddings else 0}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        raise Exception(f"Error generating embeddings: {str(e)}")

