"""
API endpoints for KORA that can be used behind a reverse proxy.
"""

import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from .rag import answer_question, build_or_load_index
from .auth import get_authenticator


HARD_CODED_MODEL = "granite3.3:2b"

# FastAPI app
app = FastAPI(
    title="KORA API",
    description="Knowledge Oriented Retrieval Assistant API",
    version="1.0.0"
)

# Security
security = HTTPBearer()


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    question: str
    top_k: int = 8


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    username: Optional[str] = None
    timestamp: float


class AuthRequest(BaseModel):
    """Request model for authentication."""
    username: str
    password: str
    demo_mode: bool = False


class AuthResponse(BaseModel):
    """Response model for authentication."""
    api_key: str
    username: str
    message: str


def get_current_user_from_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user from API key in Authorization header."""
    auth = get_authenticator()
    api_key = credentials.credentials
    
    if not auth.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    # Get username
    api_keys = auth._load_api_keys()
    key_data = api_keys.get(api_key, {})
    username = key_data.get("username", "Unknown")
    
    return username


def get_current_user_from_session(session_token: str = Cookie(None)) -> str:
    """Get current user from session cookie."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Session token required")
    
    auth = get_authenticator()
    
    if not auth.validate_session(session_token):
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    username = auth.get_user_from_session(session_token)
    if not username:
        raise HTTPException(status_code=401, detail="Could not get user from session")
    
    return username


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest) -> AuthResponse:
    """Authenticate user and generate API key."""
    auth = get_authenticator()
    
    api_key = auth.generate_api_key(request.username, request.password, demo_mode=request.demo_mode)
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    return AuthResponse(
        api_key=api_key,
        username=request.username,
        message="Authentication successful"
    )


@app.post("/auth/session")
async def create_session(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:
    """Create a session from API key."""
    auth = get_authenticator()
    api_key = credentials.credentials
    
    if not auth.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    session_token = auth.create_session(api_key)
    
    if not session_token:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    return {"session_token": session_token}


@app.post("/chat", response_model=QueryResponse)
async def chat_with_api_key(
    request: QueryRequest,
    username: str = Depends(get_current_user_from_api_key)
) -> QueryResponse:
    """Chat endpoint using API key authentication."""
    import time
    
    try:
        result = answer_question(
            query=request.question,
            top_k=request.top_k,
            model=HARD_CODED_MODEL
        )
        
        print(f"[KORA API] Query from {username}: {request.question[:50]}...")
        
        return QueryResponse(
            answer=result["answer"],
            username=username,
            timestamp=time.time()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/chat/session", response_model=QueryResponse)
async def chat_with_session(
    request: QueryRequest,
    username: str = Depends(get_current_user_from_session)
) -> QueryResponse:
    """Chat endpoint using session cookie authentication."""
    import time
    
    try:
        result = answer_question(
            query=request.question,
            top_k=request.top_k,
            model=HARD_CODED_MODEL
        )
        
        print(f"[KORA API] Session query from {username}: {request.question[:50]}...")
        
        return QueryResponse(
            answer=result["answer"],
            username=username,
            timestamp=time.time()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "KORA API"}


@app.get("/index/status")
async def index_status(username: str = Depends(get_current_user_from_api_key)) -> Dict[str, Any]:
    """Get index status."""
    try:
        store, status = build_or_load_index(force_rebuild=False)
        return {
            "status": status,
            "num_chunks": len(store.metadatas),
            "user": username
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get index status: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)