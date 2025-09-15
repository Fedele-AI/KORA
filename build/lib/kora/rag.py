import os
import json
import subprocess
from typing import List, Dict, Any, Optional

from .ingest import list_files_in_directory, convert_files_to_markdown, split_markdown_into_chunks
from .store import VectorStore


DEFAULT_RAG_DIR = "RAG"
DEFAULT_DATA_DIR = ".kora/index"


def ensure_dirs() -> None:
	os.makedirs(DEFAULT_RAG_DIR, exist_ok=True)
	os.makedirs(os.path.dirname(DEFAULT_DATA_DIR), exist_ok=True)
	os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)


def build_or_load_index(force_rebuild: bool = False, store: Optional[VectorStore] = None) -> VectorStore:
	ensure_dirs()
	store = store or VectorStore(index_dir=DEFAULT_DATA_DIR)
	if (not force_rebuild) and store.load():
		return store
	# Build new
	files = list_files_in_directory(DEFAULT_RAG_DIR)
	md_docs = convert_files_to_markdown(files)
	chunks_with_meta: List[tuple[str, str, str]] = []
	for src, md in md_docs:
		chunks = split_markdown_into_chunks(md)
		for idx, chunk in enumerate(chunks):
			chunk_id = f"{os.path.basename(src)}::chunk_{idx}"
			chunks_with_meta.append((src, chunk, chunk_id))
	store.build(chunks_with_meta)
	return store


def format_context(results: List[Dict[str, Any]]) -> str:
	formatted = []
	for r in results:
		formatted.append(f"Source: {os.path.basename(r['source'])} | Score: {r['score']:.3f}\n{r['text']}")
	return "\n\n---\n\n".join(formatted)


def call_ollama(prompt: str, model: str = "granite3.3:2b") -> str:
	proc = subprocess.run(
		["ollama", "run", model, prompt],
		capture_output=True,
		text=True,
		check=False,
	)
	stdout = proc.stdout or ""
	stderr = proc.stderr or ""
	if proc.returncode != 0:
		return f"[ollama error] {stderr.strip()}"
	return stdout.strip()


def answer_question(query: str, top_k: int = 4, model: str = "granite3.3:2b", store: Optional[VectorStore] = None) -> Dict[str, Any]:
	store = build_or_load_index(force_rebuild=False, store=store)
	results = store.search(query=query, top_k=top_k)
	context_block = format_context(results) if results else ""
	system = (
		"You are KORA, a Knowledge Oriented Retrieval Assistant. Use the provided context to answer the user question concisely."
	)
	prompt = (
		f"System: {system}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
	)
	response = call_ollama(prompt=prompt, model=model)
	return {"answer": response, "context": results}


def rebuild_index() -> Dict[str, Any]:
	store = build_or_load_index(force_rebuild=True)
	return {"status": "rebuilt", "num_chunks": len(store.metadatas)}
