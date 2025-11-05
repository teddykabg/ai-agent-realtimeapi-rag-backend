from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.embeddings import generate_embeddings
from services.vectorstore import store_documents
from utils.chunking import chunk_text
import io
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        import pypdf
        logger.info(f"Extracting text from PDF ({len(content)} bytes)")
        pdf_file = io.BytesIO(content)
        reader = pypdf.PdfReader(pdf_file)
        logger.info(f"PDF has {len(reader.pages)} pages")
        text_parts = [page.extract_text() for page in reader.pages]
        total_text_length = sum(len(text) for text in text_parts)
        logger.info(f"Extracted {total_text_length} characters from PDF")
        return "\n".join(text_parts)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF support requires 'pypdf' library. Install with: pip install pypdf"
        )
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, chunk it, generate embeddings, and store in Qdrant.
    Supports: .txt, .pdf
    """
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        logger.info(f"Reading file content... (filename: {file.filename}, content_type: {file.content_type})")
        content = await file.read()
        logger.info(f"File read successfully: {len(content)} bytes")
        
        content_type = file.content_type or ""
        filename_lower = file.filename.lower() if file.filename else ""
        
        # Extract text based on file type
        if filename_lower.endswith('.pdf') or content_type == 'application/pdf':
            logger.info("Detected PDF file, extracting text...")
            text_content = _extract_text_from_pdf(content)
        else:
            logger.info("Detected text file, decoding UTF-8...")
            # Try UTF-8 decoding for text files
            try:
                text_content = content.decode('utf-8')
                logger.info(f"Decoded text file: {len(text_content)} characters")
            except UnicodeDecodeError:
                logger.error(f"Unicode decode error for file: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Supported formats: .txt, .pdf"
                )
        
        # Chunk the text
        logger.info(f"Chunking text content...")
        chunks = chunk_text(text_content)
        logger.info(f"Created {len(chunks)} chunks from text")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from file")
        
        # Generate embeddings for chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = await generate_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Store in Qdrant
        logger.info(f"Storing {len(chunks)} chunks in Qdrant...")
        document_ids = await store_documents(chunks, embeddings, metadata={
            "filename": file.filename,
            "file_type": file.content_type or "text/plain",
            "total_chunks": len(chunks)
        })
        logger.info(f"Successfully stored {len(document_ids)} documents in Qdrant")
        
        return JSONResponse(content={
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "document_ids": document_ids
        })
    
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error: {str(e)}")
        raise HTTPException(status_code=400, detail="File must be a text file (UTF-8 encoded)")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

