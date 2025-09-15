import os
from kora.ingest import list_files_in_directory, split_markdown_into_chunks


def test_list_files_in_directory(tmp_path):
	# Create sample files and hidden file
	(tmp_path / "a.txt").write_text("hello")
	(tmp_path / ".hidden").write_text("secret")
	(tmp_path / "b.md").write_text("world")
	files = list_files_in_directory(str(tmp_path))
	names = sorted(os.path.basename(p) for p in files)
	assert names == ["a.txt", "b.md"]


def test_split_markdown_into_chunks_basic():
	text = "abcdefg" * 200
	chunks = split_markdown_into_chunks(text, chunk_size=50, overlap=10)
	assert len(chunks) > 1
	# Ensure no empty chunks
	assert all(len(c) > 0 for c in chunks)
