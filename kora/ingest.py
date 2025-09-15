import os
from typing import List, Tuple
from docling.document_converter import DocumentConverter


def list_files_in_directory(directory: str) -> List[str]:
	return [
		os.path.join(directory, f)
		for f in os.listdir(directory)
		if not f.startswith(".") and os.path.isfile(os.path.join(directory, f))
	]


def convert_files_to_markdown(file_paths: List[str]) -> List[Tuple[str, str]]:
	converter = DocumentConverter()
	results: List[Tuple[str, str]] = []
	for path in file_paths:
		try:
			res = converter.convert(path)
			md = res.document.export_to_markdown()
			results.append((path, md))
		except Exception as exc:
			# Skip file on failure but continue
			print(f"[ingest] Failed to convert {path}: {exc}")
	return results


def split_markdown_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
	chunks: List[str] = []
	start = 0
	length = len(text)
	while start < length:
		end = min(start + chunk_size, length)
		chunks.append(text[start:end])
		if end == length:
			break
		start = max(end - overlap, 0)
	return chunks
