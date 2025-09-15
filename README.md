# KORA - Knowledge Oriented Retrieval Assistant

<div align="center">

<img width=50% alt="KORA" src="https://github.com/user-attachments/assets/ef94d0c0-ea4a-49de-ac25-e2523bc0fe3d" />


</div>

KORA is a UV-based local RAG application that uses Docling for document ingestion, FAISS for vector search, and Ollama to chat with the IBM Granite 3.3:2b model. Documents are read from the `RAG/` folder and topK relevant chunks are provided as context for each query.

## Requirements
- macOS with Python 3.10+
- UV (venv already created as per instructions)
- Ollama installed and `granite3.3:2b` model already pulled

## Quickstart
```bash
# From project root
uv pip install -e .
# Launch (macOS/Linux)
./launch_kora.sh
```

For Windows (PowerShell or cmd):
```bat
launch_kora.bat
```

This will:
- Ensure dependencies are installed
- Create `RAG/` if missing
- Build the FAISS index on startup
- Launch a Gradio web UI at http://127.0.0.1:7860

## Usage
- Drop PDFs/DOCX/TXT into `RAG/`
- Use the UI to control topK and ask questions. The app retrieves topK chunks and queries Ollama `granite3.3:2b`.
- Click “Rebuild Index” after adding/removing files. A green notice confirms rebuild.

## Architecture
```mermaid
flowchart LR
    A["RAG folder (PDF, DOCX, TXT)"] --> B[Docling Convert]
    B --> C[Chunking]
    C --> D["Embeddings (Sentence-Transformers)"]
    D --> E[FAISS Index]
    F[User Query] --> G["Retriever (topK)"]
    E --> G
    G --> H["Prompt Builder (System: You are KORA)"]
    H --> I["Ollama granite3.3:2b"]
    I --> J[Gradio UI]
```

## Notes
- Index is stored under `.kora/index/faiss_index` and `.kora/index/meta.json`.
- To force rebuild, click the Rebuild button or delete `.kora/index`.
