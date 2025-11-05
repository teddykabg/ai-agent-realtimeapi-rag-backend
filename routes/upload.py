from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.embeddings import generate_embeddings
from services.vectorstore import store_documents
from utils.chunking import chunk_text

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, chunk it, generate embeddings, and store in Qdrant.
    """
    try:
        # Validate file type (optional: add more validation)
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Chunk the text
        chunks = chunk_text(text_content)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from file")
        
        # Generate embeddings for chunks
        embeddings = await generate_embeddings(chunks)
        
        # Store in Qdrant
        document_ids = await store_documents(chunks, embeddings, metadata={
            "filename": file.filename,
            "file_type": file.content_type or "text/plain",
            "total_chunks": len(chunks)
        })
        
        return JSONResponse(content={
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "document_ids": document_ids
        })
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be a text file (UTF-8 encoded)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

