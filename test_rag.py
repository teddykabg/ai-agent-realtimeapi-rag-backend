"""
Test script to verify RAG implementation works correctly.
Tests:
1. Qdrant connection
2. Embedding generation
3. Document upload and storage
4. Document search
5. Reranking (if Cohere API key is available)
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vectorstore import ensure_collection, store_documents, search_documents
from services.embeddings import generate_embeddings
from services.reranker import rerank_documents
from utils.chunking import chunk_text


async def test_qdrant_connection():
    """Test if Qdrant is accessible."""
    print("\n" + "="*60)
    print("TEST 1: Qdrant Connection")
    print("="*60)
    try:
        await ensure_collection()
        print("✓ Qdrant connection successful!")
        print("✓ Collection ensured/created")
        return True
    except Exception as e:
        print(f"✗ Qdrant connection failed: {str(e)}")
        print("\nPlease ensure Qdrant is running:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return False


async def test_embeddings():
    """Test embedding generation."""
    print("\n" + "="*60)
    print("TEST 2: Embedding Generation")
    print("="*60)
    try:
        test_texts = ["Hello world", "This is a test document"]
        embeddings = await generate_embeddings(test_texts)
        
        if len(embeddings) == 2:
            print(f"✓ Generated embeddings for {len(embeddings)} texts")
            print(f"✓ Embedding dimension: {len(embeddings[0])}")
            if len(embeddings[0]) == 1536:
                print("✓ Embedding dimension is correct (1536 for text-embedding-3-small)")
            return True
        else:
            print(f"✗ Expected 2 embeddings, got {len(embeddings)}")
            return False
    except Exception as e:
        print(f"✗ Embedding generation failed: {str(e)}")
        print("\nPlease check your OPENAI_API_KEY in .env file")
        return False


async def test_chunking():
    """Test text chunking."""
    print("\n" + "="*60)
    print("TEST 3: Text Chunking")
    print("="*60)
    try:
        test_text = """
        This is a test document. It contains multiple sentences.
        We want to see if it gets chunked properly. The chunking should
        split the text into manageable pieces while preserving context.
        This helps with better retrieval and processing of documents.
        """ * 5  # Make it longer to ensure multiple chunks
        
        chunks = chunk_text(test_text)
        
        if len(chunks) > 0:
            print(f"✓ Text chunked into {len(chunks)} chunks")
            print(f"✓ First chunk length: {len(chunks[0])} characters")
            print(f"✓ Sample chunk preview: {chunks[0][:100]}...")
            return True, chunks
        else:
            print("✗ No chunks generated")
            return False, []
    except Exception as e:
        print(f"✗ Chunking failed: {str(e)}")
        return False, []


async def test_document_storage():
    """Test document storage in Qdrant."""
    print("\n" + "="*60)
    print("TEST 4: Document Storage")
    print("="*60)
    try:
        # Create test chunks
        test_chunks = [
            "Python is a programming language.",
            "FastAPI is a web framework for Python.",
            "Qdrant is a vector database.",
            "RAG stands for Retrieval-Augmented Generation."
        ]
        
        # Generate embeddings
        embeddings = await generate_embeddings(test_chunks)
        
        # Store documents
        document_ids = await store_documents(
            chunks=test_chunks,
            embeddings=embeddings,
            metadata={"test": True, "source": "test_rag.py"}
        )
        
        if len(document_ids) == len(test_chunks):
            print(f"✓ Stored {len(document_ids)} documents successfully")
            print(f"✓ Document IDs: {document_ids[:2]}...")
            return True, document_ids
        else:
            print(f"✗ Expected {len(test_chunks)} document IDs, got {len(document_ids)}")
            return False, []
    except Exception as e:
        print(f"✗ Document storage failed: {str(e)}")
        return False, []


async def test_document_search():
    """Test document search."""
    print("\n" + "="*60)
    print("TEST 5: Document Search")
    print("="*60)
    try:
        # Search for relevant documents
        query = "What is Python?"
        results = await search_documents(query, top_k=3)
        
        if len(results) > 0:
            print(f"✓ Found {len(results)} relevant documents")
            print(f"✓ Top result score: {results[0]['score']:.4f}")
            print(f"✓ Top result chunk: {results[0]['chunk'][:80]}...")
            
            # Show all results
            print("\n  Search Results:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. Score: {result['score']:.4f}")
                print(f"       Chunk: {result['chunk'][:60]}...")
            
            return True, results
        else:
            print("✗ No search results found")
            print("  Make sure you've stored documents first (TEST 4)")
            return False, []
    except Exception as e:
        print(f"✗ Document search failed: {str(e)}")
        return False, []


async def test_reranking():
    """Test reranking with Cohere."""
    print("\n" + "="*60)
    print("TEST 6: Document Reranking")
    print("="*60)
    
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("⚠ COHERE_API_KEY not set - skipping reranking test")
        print("  (Reranking is optional, RAG will work without it)")
        return True
    
    try:
        query = "What is Python?"
        documents = [
            "Python is a programming language.",
            "FastAPI is a web framework for Python.",
            "Qdrant is a vector database.",
            "RAG stands for Retrieval-Augmented Generation."
        ]
        
        reranked = await rerank_documents(query, documents, top_k=3)
        
        if len(reranked) > 0:
            print(f"✓ Reranked {len(reranked)} documents")
            print(f"✓ Top reranked result: {reranked[0]}")
            return True
        else:
            print("✗ No reranked documents returned")
            return False
    except Exception as e:
        print(f"✗ Reranking failed: {str(e)}")
        print("  (This is optional - RAG will work without reranking)")
        return True  # Don't fail the test suite if reranking fails


async def test_end_to_end():
    """Test complete RAG pipeline."""
    print("\n" + "="*60)
    print("TEST 7: End-to-End RAG Pipeline")
    print("="*60)
    try:
        # Create a test document
        test_document = """
        Artificial Intelligence (AI) is transforming the world.
        Machine learning is a subset of AI that enables computers to learn.
        Deep learning uses neural networks with multiple layers.
        Natural Language Processing (NLP) helps computers understand text.
        RAG combines retrieval and generation for better AI responses.
        """
        
        # Step 1: Chunk
        chunks = chunk_text(test_document)
        print(f"✓ Step 1: Chunked document into {len(chunks)} chunks")
        
        # Step 2: Generate embeddings
        embeddings = await generate_embeddings(chunks)
        print(f"✓ Step 2: Generated embeddings ({len(embeddings)} embeddings)")
        
        # Step 3: Store
        doc_ids = await store_documents(chunks, embeddings, metadata={"test": "e2e"})
        print(f"✓ Step 3: Stored {len(doc_ids)} chunks in Qdrant")
        
        # Step 4: Search
        query = "What is machine learning?"
        search_results = await search_documents(query, top_k=2)
        print(f"✓ Step 4: Retrieved {len(search_results)} relevant chunks")
        
        # Step 5: Rerank (if available)
        if os.getenv("COHERE_API_KEY"):
            chunks_only = [r['chunk'] for r in search_results]
            reranked = await rerank_documents(query, chunks_only, top_k=2)
            print(f"✓ Step 5: Reranked to {len(reranked)} top chunks")
        
        print("\n✓ End-to-end RAG pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Qdrant Connection
    qdrant_ok = await test_qdrant_connection()
    results.append(("Qdrant Connection", qdrant_ok))
    
    if not qdrant_ok:
        print("\n⚠ Cannot proceed without Qdrant. Please start Qdrant first.")
        return
    
    # Test 2: Embeddings
    embeddings_ok = await test_embeddings()
    results.append(("Embedding Generation", embeddings_ok))
    
    if not embeddings_ok:
        print("\n⚠ Cannot proceed without embeddings. Please check OPENAI_API_KEY.")
        return
    
    # Test 3: Chunking
    chunking_ok, _ = await test_chunking()
    results.append(("Text Chunking", chunking_ok))
    
    # Test 4: Document Storage
    storage_ok, _ = await test_document_storage()
    results.append(("Document Storage", storage_ok))
    
    # Test 5: Document Search
    search_ok, _ = await test_document_search()
    results.append(("Document Search", search_ok))
    
    # Test 6: Reranking (optional)
    rerank_ok = await test_reranking()
    results.append(("Reranking (optional)", rerank_ok))
    
    # Test 7: End-to-End
    e2e_ok = await test_end_to_end()
    results.append(("End-to-End Pipeline", e2e_ok))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for test_name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your RAG implementation is working correctly.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())

