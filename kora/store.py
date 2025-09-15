import os
import json
import re
from typing import List, Tuple, Dict, Any, Protocol, Optional

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingBackend(Protocol):
	def encode(self, texts: List[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
	def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
		self.model = SentenceTransformer(model_name)

	def encode(self, texts: List[str]) -> np.ndarray:
		emb = self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
		return emb.astype("float32")


class TfidfBackend:
	def __init__(self) -> None:
		from sklearn.feature_extraction.text import TfidfVectorizer
		self.vectorizer = TfidfVectorizer()
		self.fitted = False

	def fit(self, texts: List[str]) -> None:
		self.vectorizer.fit(texts)
		self.fitted = True

	def encode(self, texts: List[str]) -> np.ndarray:
		if not self.fitted:
			self.fit(texts)
		arr = self.vectorizer.transform(texts).astype(np.float32).toarray()
		return arr


class VectorStore:
	def __init__(self, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", backend: Optional[EmbeddingBackend] = None) -> None:
		self.index_dir = index_dir
		self.index_path = os.path.join(index_dir, "faiss_index")
		self.meta_path = os.path.join(index_dir, "meta.json")
		self.embedding_model_name = model_name
		self.embedding_backend: EmbeddingBackend = backend or SentenceTransformerBackend(model_name)
		self.index: faiss.IndexFlatIP | None = None
		self.metadatas: List[Dict[str, Any]] = []
		self.source_fingerprint: str = ""

	def _ensure_dir(self) -> None:
		os.makedirs(self.index_dir, exist_ok=True)

	def _normalize(self, vectors: np.ndarray) -> np.ndarray:
		if vectors.size == 0:
			return vectors
		norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
		return vectors / norms

	def build(self, chunks_with_meta: List[Tuple[str, str, str]], source_fingerprint: str = "") -> None:
		self._ensure_dir()
		texts = [c[1] for c in chunks_with_meta]
		# Handle empty corpus gracefully
		if len(texts) == 0:
			self.index = None
			self.metadatas = []
			self.source_fingerprint = source_fingerprint
			self._persist()
			return
		embeddings = self.embedding_backend.encode(texts)
		if embeddings.dtype != np.float32:
			embeddings = embeddings.astype("float32")
		# Ensure 2D shape
		if embeddings.ndim == 1:
			embeddings = embeddings.reshape(1, -1)
		embeddings = self._normalize(embeddings)
		index = faiss.IndexFlatIP(embeddings.shape[1])
		index.add(embeddings)
		self.index = index
		self.metadatas = [
			{"source": c[0], "chunk_id": c[2], "text": c[1]}
			for c in chunks_with_meta
		]
		self.source_fingerprint = source_fingerprint
		self._persist()

	def _persist(self) -> None:
		# Persist metadata always; index only if present
		if self.index is not None:
			faiss.write_index(self.index, self.index_path)
		with open(self.meta_path, "w", encoding="utf-8") as f:
			json.dump({
				"embedding_model": self.embedding_model_name,
				"metadatas": self.metadatas,
				"source_fingerprint": self.source_fingerprint,
			}, f)

	def load(self) -> bool:
		if not os.path.exists(self.meta_path):
			return False
		# Load metadata
		with open(self.meta_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		self.metadatas = data.get("metadatas", [])
		self.source_fingerprint = data.get("source_fingerprint", "")
		# Load index if available
		if os.path.exists(self.index_path):
			self.index = faiss.read_index(self.index_path)
		else:
			self.index = None
		return True

	def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
		# If no index or no data, return empty results
		if self.index is None or len(self.metadatas) == 0:
			return []
		query_vec = self.embedding_backend.encode([query])
		if query_vec.dtype != np.float32:
			query_vec = query_vec.astype("float32")
		if query_vec.ndim == 1:
			query_vec = query_vec.reshape(1, -1)
		query_vec = self._normalize(query_vec)
		# Initial embedding search (request a bit more to allow hybrid merge)
		embed_k = min(max(top_k * 2, top_k), len(self.metadatas))
		dists, idxs = self.index.search(query_vec, embed_k)
		indices = idxs[0].tolist()
		scores = dists[0].tolist()
		results_map: Dict[int, Dict[str, Any]] = {}
		for i, score in zip(indices, scores):
			if i < 0 or i >= len(self.metadatas):
				continue
			meta = self.metadatas[i]
			results_map[i] = {
				"score": float(score),
				"text": meta["text"],
				"source": meta["source"],
				"chunk_id": meta["chunk_id"],
			}

		# Keyword fallback / boost: ensure rare or explicit terms surface
		query_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 3]
		if query_tokens:
			keyword_scores: Dict[int, int] = {}
			for idx, meta in enumerate(self.metadatas):
				lower_text = meta["text"].lower()
				match_count = 0
				for qt in query_tokens:
					if qt in lower_text:
						match_count += 1
				if match_count > 0:
					keyword_scores[idx] = match_count
			# If none of the initial embedding results contain any keyword while others do, merge them
			if keyword_scores:
				contains_keyword_in_results = any(i in keyword_scores for i in results_map.keys())
				if not contains_keyword_in_results:
					# Add top keyword candidates
					for idx, kw_score in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
						meta = self.metadatas[idx]
						results_map[idx] = {
							"score": float(0.1 * kw_score),  # small base score; will re-rank below
							"text": meta["text"],
							"source": meta["source"],
							"chunk_id": meta["chunk_id"],
						}
			else:
				# Boost existing results that match keywords
				for idx, kw_score in keyword_scores.items():
					if idx in results_map:
						results_map[idx]["score"] += 0.05 * kw_score

		# Return top_k after potential keyword merge
		final = sorted(results_map.values(), key=lambda r: r["score"], reverse=True)[:top_k]
		return final
