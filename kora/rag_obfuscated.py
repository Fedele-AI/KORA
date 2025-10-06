"""
Extension to rag.py for working with obfuscated data packages.
"""

import os
from typing import Dict, Any, Optional
from .obfuscate import ObfuscatedVectorStore
from .rag import call_ollama, format_context


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
    
    system = (
        "You are KORA - the Knowledge Oriented Retrieval Assistant. You are a helpful assistant created by researchers at Georgia Tech to help students with course content. Use ONLY the provided context to answer. If the answer is not in the context, say you don't know. Be concise."
    )
    
    prompt = (
        f"System: {system}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    response = call_ollama(prompt=prompt, model=model)
    
    return {"answer": response, "context": results}


def create_obfuscated_ui(binary_path: str, password: str) -> Any:
    """
    Create a Gradio interface for obfuscated data package.
    
    Args:
        binary_path: Path to obfuscated .bin file
        password: Decryption password
    
    Returns:
        Gradio Blocks interface
    """
    import gradio as gr
    
    def query_handler(question: str, top_k: int) -> tuple[str, str]:
        result = answer_question_obfuscated(
            query=question,
            binary_path=binary_path,
            password=password,
            top_k=top_k
        )
        
        answer = result["answer"]
        context_details = ""
        
        for i, ctx in enumerate(result["context"], 1):
            source = ctx.get("source", "Unknown")
            score = ctx.get("score", 0.0)
            text_preview = ctx.get("text", "")[:200]
            context_details += f"\n**{i}. {source}** (Score: {score:.3f})\n{text_preview}...\n\n"
        
        return answer, context_details
    
    with gr.Blocks(title="KORA - Obfuscated Mode") as interface:
        gr.Markdown("# 🔒 KORA - Knowledge Oriented Retrieval Assistant (Secure Mode)")
        gr.Markdown(
            f"**Obfuscated Data Package:** `{os.path.basename(binary_path)}`\n\n"
            "This mode uses encrypted embeddings. Original documents cannot be accessed directly."
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the course materials...",
                    lines=3
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label="Number of chunks to retrieve (topK)"
                )
                submit_btn = gr.Button("Get Answer", variant="primary")
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                context_output = gr.Markdown(label="Retrieved Context")
        
        submit_btn.click(
            fn=query_handler,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, context_output]
        )
        
        question_input.submit(
            fn=query_handler,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, context_output]
        )
        
        gr.Markdown(
            """
            ---
            **Note:** This interface uses an obfuscated data package. 
            The original documents are encrypted and cannot be extracted.
            """
        )
    
    return interface
