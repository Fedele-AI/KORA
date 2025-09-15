import os
import json
import subprocess
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from .ingest import list_files_in_directory, convert_files_to_markdown, split_markdown_into_chunks
from .store import VectorStore


DEFAULT_RAG_DIR = "RAG"
DEFAULT_DATA_DIR = ".kora/index"


def ensure_dirs() -> None:
	os.makedirs(DEFAULT_RAG_DIR, exist_ok=True)
	os.makedirs(os.path.dirname(DEFAULT_DATA_DIR), exist_ok=True)
	os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)


def _fingerprint_files(paths: List[str]) -> str:
	records: List[Tuple[str, float]] = []
	for p in sorted(paths):
		try:
			st = os.stat(p)
			records.append((os.path.basename(p), st.st_mtime))
		except FileNotFoundError:
			continue
	payload = json.dumps(records).encode("utf-8")
	return hashlib.sha256(payload).hexdigest()


def build_or_load_index(force_rebuild: bool = False, store: Optional[VectorStore] = None) -> Tuple[VectorStore, str]:
	ensure_dirs()
	store = store or VectorStore(index_dir=DEFAULT_DATA_DIR)
	files = list_files_in_directory(DEFAULT_RAG_DIR)
	current_fp = _fingerprint_files(files)
	
	# Try to load existing index
	if store.load() and not force_rebuild:
		# Check if rebuild is needed due to file changes
		if store.source_fingerprint == current_fp and not (len(store.metadatas) == 0 and len(files) > 0):
			return store, "loaded_from_disk"
	
	# Build new index
	md_docs = convert_files_to_markdown(files)
	chunks_with_meta: List[tuple[str, str, str]] = []
	for src, md in md_docs:
		chunks = split_markdown_into_chunks(md)
		for idx, chunk in enumerate(chunks):
			chunk_id = f"{os.path.basename(src)}::chunk_{idx}"
			chunks_with_meta.append((src, chunk, chunk_id))
	
	store.build(chunks_with_meta, source_fingerprint=current_fp)
	return store, "rebuilt"


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


def answer_question(query: str, top_k: int = 8, model: str = "granite3.3:2b", store: Optional[VectorStore] = None) -> Dict[str, Any]:
	store, _ = build_or_load_index(force_rebuild=False, store=store)
	results = store.search(query=query, top_k=top_k)
	context_block = format_context(results) if results else ""
	system = (
		"You are KORA. You are the Knowledge Oriented Retrieval Assistant. You are a helpful assistant created by researchers at Georgia Tech to help students with course content. Use ONLY the provided context to answer. If the answer is not in the context, say you don't know. Be concise."
	)
	prompt = (
		f"System: {system}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
	)
	response = call_ollama(prompt=prompt, model=model)
	return {"answer": response, "context": results}


def rebuild_index() -> Dict[str, Any]:
	store, _ = build_or_load_index(force_rebuild=True)
	return {"status": "rebuilt", "num_chunks": len(store.metadatas)}
