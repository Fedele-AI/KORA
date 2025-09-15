import os
from kora.store import VectorStore, TfidfBackend


def test_build_and_search_with_tfidf(tmp_path):
	index_dir = tmp_path / ".kora_index"
	store = VectorStore(index_dir=str(index_dir), backend=TfidfBackend())
	chunks = [
		("doc1", "The quick brown fox jumps over the lazy dog", "d1c0"),
		("doc2", "Python is a great programming language", "d2c0"),
		("doc3", "Foxes are quick and smart", "d3c0"),
	]
	store.build(chunks)
	assert store.load() is True
	results = store.search("quick fox", top_k=2)
	assert len(results) == 2
	assert any("fox" in r["text"].lower() for r in results)
