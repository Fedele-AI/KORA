import os
from typing import List, Tuple
from docling.document_converter import DocumentConverter


def list_files_in_directory(directory: str) -> List[str]:
	files = []
	for root, dirs, filenames in os.walk(directory):
		for filename in filenames:
			if not filename.startswith("."):
				full_path = os.path.join(root, filename)
				files.append(full_path)
	return files


def convert_files_to_markdown(file_paths: List[str]) -> List[Tuple[str, str]]:
	converter = DocumentConverter()
	results: List[Tuple[str, str]] = []
	for path in file_paths:
		try:
			# Handle markdown files directly
			if path.lower().endswith('.md'):
				with open(path, 'r', encoding='utf-8') as f:
					md_content = f.read()
				results.append((path, md_content))
			else:
				# Use docling for other file types (PDFs, etc.)
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
