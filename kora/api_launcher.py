#!/usr/bin/env python3
"""
Launch script for KORA API server.
"""

import uvicorn
from .api import app
from .rag import ensure_dirs


def main():
    """Launch the KORA API server."""
    # Ensure directories
    ensure_dirs()
    
    print("[KORA API] Starting API server...")
    print("[KORA API] Available at: http://127.0.0.1:8000")
    print("[KORA API] API Documentation: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()