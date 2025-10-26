"""
REST API for KORA - Knowledge Oriented Retrieval Assistant.
Provides endpoints to query the RAG system and manage the knowledge base.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

from .auth import get_authenticator
from .rag import answer_question, rebuild_index, build_or_load_index
from .store import VectorStore
from .config import get_default_model, get_default_temperature, get_default_top_k


# FastAPI app
app = FastAPI(
    title="KORA API",
    description="Knowledge Oriented Retrieval Assistant - RAG API powered by Ollama",
    version="0.1.0"
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for question queries."""
    question: str = Field(..., description="The question to ask", min_length=1)
    top_k: int = Field(default=8, description="Number of context chunks to retrieve", ge=1, le=50)
    model: Optional[str] = Field(default=None, description="Ollama model to use (defaults to configured model)")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation (0.0-1.0)", ge=0.0, le=1.0)


class ContextChunk(BaseModel):
    """Model for a context chunk."""
    source: str = Field(..., description="Source file name")
    text: str = Field(..., description="Text content of the chunk")
    score: float = Field(..., description="Similarity score")


class QueryResponse(BaseModel):
    """Response model for question queries."""
    answer: str = Field(..., description="The generated answer")
    context: List[ContextChunk] = Field(..., description="Retrieved context chunks")
    model: str = Field(..., description="Model used for generation")


class RebuildRequest(BaseModel):
    """Request model for rebuilding the index."""
    force: bool = Field(default=False, description="Force rebuild even if index exists")


class RebuildResponse(BaseModel):
    """Response model for rebuild operations."""
    success: bool = Field(..., description="Whether rebuild succeeded")
    message: str = Field(..., description="Status message")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str = Field(..., description="API status")
    rag_dir: str = Field(..., description="RAG directory path")
    index_exists: bool = Field(..., description="Whether index exists")
    total_chunks: int = Field(..., description="Total chunks in index")
    default_model: str = Field(..., description="Default Ollama model")


class SearchRequest(BaseModel):
    """Request model for vector search."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=8, description="Number of results to return", ge=1, le=50)


class SearchResponse(BaseModel):
    """Response model for vector search."""
    results: List[ContextChunk] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")


# Dependency for API key validation
async def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validate API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    authenticator = get_authenticator()
    if not authenticator.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key"
        )
    return api_key


# Helper function to get VectorStore
def get_vector_store() -> VectorStore:
    """Get the VectorStore instance."""
    from .rag import DEFAULT_DATA_DIR
    return VectorStore(index_dir=DEFAULT_DATA_DIR)


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "KORA API",
        "version": "0.1.0",
        "description": "Knowledge Oriented Retrieval Assistant",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status(api_key: str = Depends(validate_api_key)):
    """
    Get API status and configuration.
    
    Requires valid API key.
    """
    from .rag import DEFAULT_RAG_DIR, DEFAULT_DATA_DIR
    
    store = get_vector_store()
    index_exists = store.load()
    total_chunks = len(store.metadatas) if index_exists else 0
    
    return StatusResponse(
        status="operational",
        rag_dir=DEFAULT_RAG_DIR,
        index_exists=index_exists,
        total_chunks=total_chunks,
        default_model=get_default_model()
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    request: QueryRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Query the RAG system with a question.
    
    This endpoint:
    1. Retrieves relevant context from the vector store
    2. Sends the question + context to Ollama
    3. Returns the generated answer with context
    
    Requires valid API key.
    """
    try:
        # Use default model if not specified
        model = request.model or get_default_model()
        
        # Query the RAG system
        result = answer_question(
            query=request.question,
            top_k=request.top_k,
            model=model
        )
        
        # Format response
        context_chunks = [
            ContextChunk(
                source=ctx.get("source", "Unknown"),
                text=ctx.get("text", ""),
                score=ctx.get("score", 0.0)
            )
            for ctx in result.get("context", [])
        ]
        
        return QueryResponse(
            answer=result.get("answer", ""),
            context=context_chunks,
            model=model
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse, tags=["RAG"])
async def search(
    request: SearchRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Search the vector store without generating an answer.
    
    Returns relevant context chunks based on semantic similarity.
    
    Requires valid API key.
    """
    try:
        store = get_vector_store()
        
        if not store.load():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Index not available. Please build the index first."
            )
        
        results = store.search(query=request.query, top_k=request.top_k)
        
        context_chunks = [
            ContextChunk(
                source=ctx.get("source", "Unknown"),
                text=ctx.get("text", ""),
                score=ctx.get("score", 0.0)
            )
            for ctx in results
        ]
        
        return SearchResponse(
            results=context_chunks,
            query=request.query
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/rebuild", response_model=RebuildResponse, tags=["Management"])
async def rebuild(
    request: RebuildRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Rebuild the vector index from RAG directory.
    
    This will:
    1. Scan the RAG directory for documents
    2. Convert documents to markdown
    3. Split into chunks
    4. Generate embeddings
    5. Build FAISS index
    
    Requires valid API key.
    """
    try:
        # Always rebuild when this endpoint is called
        result = rebuild_index()
        
        return RebuildResponse(
            success=True,
            message="Index rebuilt successfully",
            chunks_indexed=result.get("num_chunks", 0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rebuild failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }
