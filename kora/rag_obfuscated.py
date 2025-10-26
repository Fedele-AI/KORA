"""
Extension to rag.py for working with obfuscated data packages.
"""

import os
from typing import Dict, Any, Optional
from .obfuscate import ObfuscatedVectorStore
from .rag import call_ollama, format_context
from .config import get_system_prompt_text


def answer_question_obfuscated(
    query: str,
    binary_path: str,
    password: str,
    top_k: int = 8,
    model: str = "granite3.3:2b"
) -> Dict[str, Any]:
    """
    Answer questions using an obfuscated data package instead of RAG folder.
    
    Args:
        query: User question
        binary_path: Path to obfuscated .bin file
        password: Decryption password
        top_k: Number of chunks to retrieve
        model: Ollama model name
    
    Returns:
        Dictionary with answer and context
    """
    # Load obfuscated store
    store = ObfuscatedVectorStore(binary_path, password)
    
    if not store.load():
        return {
            "answer": "Error: Failed to load obfuscated data package. Check password and file path.",
            "context": []
        }
    
    # Search using encrypted embeddings
    results = store.search(query=query, top_k=top_k)
    context_block = format_context(results) if results else ""
    
    # Get the configured system prompt
    system = get_system_prompt_text()
    
    prompt = (
        f"System: {system}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    response = call_ollama(prompt=prompt, model=model)
    
    return {"answer": response, "context": results}
