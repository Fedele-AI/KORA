from typing import List, Tuple

from kora.rag import answer_question, call_ollama, build_or_load_index
from kora.store import VectorStore, TfidfBackend


def test_answer_question_with_injected_store(monkeypatch, tmp_path):
	# Build a tiny store with TF-IDF
	index_dir = tmp_path / "idx"
	store = VectorStore(index_dir=str(index_dir), backend=TfidfBackend())
	chunks: List[Tuple[str, str, str]] = [
		("doc", "Cats are cute animals.", "c0"),
		("doc", "Dogs are loyal pets.", "c1"),
	]
	store.build(chunks)

	# Mock ollama to avoid external dependency
	def fake_call(prompt: str, model: str = "granite3.3:2b") -> str:
		return "Test answer"

	monkeypatch.setattr("kora.rag.call_ollama", fake_call)

	res = answer_question(query="What are cats?", top_k=1, model="granite3.3:2b", store=store)
	assert "answer" in res and res["answer"] == "Test answer"
	assert "context" in res and len(res["context"]) == 1
