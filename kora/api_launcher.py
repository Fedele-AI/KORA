#!/usr/bin/env python3
"""
Launch script for KORA API server.
"""

import argparse
import uvicorn
from .api import app
from .rag import ensure_dirs


def main():
    """Launch the KORA API server."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Launch KORA REST API server')
    parser.add_argument(
        '--host',
        type=str,
        default="0.0.0.0",
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    args = parser.parse_args()
    
    # Ensure directories
    ensure_dirs()
    
    print("[KORA API] Starting API server...")
    print(f"[KORA API] Available at: http://{args.host}:{args.port}")
    print(f"[KORA API] API Documentation: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()