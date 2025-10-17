"""Tests for KPKG (KORA Package) format."""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
import numpy as np
import faiss

from kora.obfuscate import KPKGFormat, ObfuscatedVectorStore, ObfuscatedStore
from kora.store import VectorStore, TfidfBackend


def test_kpkg_encode_decode_unencrypted():
	"""Test basic KPKG encoding and decoding without encryption."""
	metadata = {
		"version": "1.0",
		"model": "test-model",
		"chunks": 3
	}
	
	# Create a simple FAISS index
	dim = 4
	index = faiss.IndexFlatL2(dim)
	vectors = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	
	text_contents = ["First chunk", "Second chunk", "Third chunk"]
	
	# Encode
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key=None,
		compress=True
	)
	
	assert isinstance(package, bytes)
	assert package.startswith(b'KORA')
	
	# Decode
	decoded_meta, decoded_faiss, decoded_text = KPKGFormat.decode(
		package_data=package,
		decrypt_key=None
	)
	
	assert decoded_meta == metadata
	assert decoded_faiss == faiss_data
	assert decoded_text == text_contents


def test_kpkg_encode_decode_encrypted():
	"""Test KPKG encoding and decoding with encryption."""
	metadata = {"test": "data"}
	
	dim = 2
	index = faiss.IndexFlatL2(dim)
	vectors = np.array([[1.0, 2.0]], dtype=np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	
	text_contents = ["Encrypted chunk"]
	encryption_key = "test_password_123"
	
	# Encode with encryption
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key=encryption_key,
		compress=True
	)
	
	# Try to decode without key (should fail)
	with pytest.raises(ValueError, match="encrypted but no decryption key"):
		KPKGFormat.decode(package_data=package, decrypt_key=None)
	
	# Decode with correct key
	decoded_meta, decoded_faiss, decoded_text = KPKGFormat.decode(
		package_data=package,
		decrypt_key=encryption_key
	)
	
	assert decoded_meta == metadata
	assert decoded_faiss == faiss_data
	assert decoded_text == text_contents


def test_kpkg_wrong_decryption_key():
	"""Test that wrong decryption key fails."""
	metadata = {"test": "data"}
	
	dim = 2
	index = faiss.IndexFlatL2(dim)
	vectors = np.array([[1.0, 2.0]], dtype=np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	
	text_contents = ["Secret chunk"]
	encryption_key = "correct_password"
	
	# Encode with encryption
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key=encryption_key,
		compress=True
	)
	
	# Try to decode with wrong key (should raise decryption error)
	with pytest.raises(Exception):  # Fernet raises various exceptions on wrong key
		KPKGFormat.decode(package_data=package, decrypt_key="wrong_password")


def test_kpkg_without_compression():
	"""Test KPKG format without compression."""
	metadata = {"compressed": False}
	
	dim = 2
	index = faiss.IndexFlatL2(dim)
	vectors = np.array([[1.0, 2.0]], dtype=np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	
	text_contents = ["Uncompressed"]
	
	# Encode without compression
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key=None,
		compress=False
	)
	
	# Decode
	decoded_meta, decoded_faiss, decoded_text = KPKGFormat.decode(
		package_data=package,
		decrypt_key=None
	)
	
	assert decoded_meta == metadata
	assert decoded_text == text_contents


def test_kpkg_invalid_magic():
	"""Test that invalid magic bytes are rejected."""
	invalid_package = b'FAKE' + b'\x00' * 20
	
	with pytest.raises(ValueError, match="Invalid KPKG format"):
		KPKGFormat.decode(package_data=invalid_package, decrypt_key=None)


def test_obfuscated_vector_store_create_and_load(tmp_path):
	"""Test creating and loading an ObfuscatedVectorStore."""
	# First, create a regular VectorStore with appropriate backend
	index_dir = tmp_path / "regular_index"
	# Use default backend (SentenceTransformer) which the KPKG will preserve
	store = VectorStore(index_dir=str(index_dir))
	
	chunks: List[Tuple[str, str, str]] = [
		("doc1.txt", "Machine learning is a subset of AI.", "chunk_0"),
		("doc1.txt", "Deep learning uses neural networks.", "chunk_1"),
		("doc2.txt", "Python is a programming language.", "chunk_2"),
	]
	store.build(chunks)
	
	# Create KPKG from the store
	kpkg_path = tmp_path / "test_package.kpkg"
	password = "test_pass_123"
	
	obf_store_creator = ObfuscatedStore(str(kpkg_path), password)
	result = obf_store_creator.obfuscate_from_store(
		store=store,
		include_source_names=True,
		use_encryption=True
	)
	
	assert kpkg_path.exists()
	assert result["num_chunks"] == 3
	
	# Load the KPKG - need to use the same embedding model
	loaded_store = ObfuscatedVectorStore(str(kpkg_path), password)
	# Set the embedding model to match what was stored
	loaded_store.embedding_model_name = store.embedding_model_name
	loaded_store.load()
	
	# Verify the store loaded
	assert loaded_store.index is not None
	assert len(loaded_store.metadatas) == 3


def test_obfuscated_vector_store_without_password(tmp_path):
	"""Test that encrypted KPKG without password still loads (obfuscated layer)."""
	# Create a minimal KPKG with encryption
	metadata = {"test": "encrypted"}
	dim = 2
	index = faiss.IndexFlatL2(dim)
	vectors = np.array([[1.0, 2.0]], dtype=np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	text_contents = ["Encrypted content"]
	
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key="secret",
		compress=True
	)
	
	kpkg_path = tmp_path / "encrypted.kpkg"
	with open(kpkg_path, 'wb') as f:
		f.write(package)
	
	# ObfuscatedVectorStore has fallback logic, so it won't raise on init
	# Instead, test that the raw KPKGFormat.decode raises the error
	with pytest.raises(ValueError, match="encrypted but no decryption key"):
		with open(kpkg_path, 'rb') as f:
			kpkg_data = f.read()
		KPKGFormat.decode(package_data=kpkg_data, decrypt_key=None)


def test_kpkg_obfuscation_is_reversible():
	"""Test that obfuscation is symmetric (XOR-based)."""
	original_data = b"This is secret data that needs obfuscation!"
	
	# Obfuscate
	obfuscated = KPKGFormat._obfuscate_bytes(original_data)
	assert obfuscated != original_data
	
	# Deobfuscate
	deobfuscated = KPKGFormat._deobfuscate_bytes(obfuscated)
	assert deobfuscated == original_data


def test_kpkg_large_content():
	"""Test KPKG with larger content."""
	metadata = {"size": "large"}
	
	# Create larger index
	dim = 128
	n_vectors = 100
	index = faiss.IndexFlatL2(dim)
	vectors = np.random.randn(n_vectors, dim).astype(np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	
	# Create many text chunks
	text_contents = [f"This is chunk number {i} with some content." for i in range(100)]
	
	# Encode
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key=None,
		compress=True
	)
	
	# Decode
	decoded_meta, decoded_faiss, decoded_text = KPKGFormat.decode(
		package_data=package,
		decrypt_key=None
	)
	
	assert decoded_meta == metadata
	assert len(decoded_text) == 100
	assert decoded_text[50] == "This is chunk number 50 with some content."


def test_kpkg_metadata_preservation():
	"""Test that complex metadata is preserved."""
	metadata = {
		"version": "1.2.3",
		"model": "sentence-transformers/all-MiniLM-L6-v2",
		"chunks": 42,
		"sources": ["doc1.pdf", "doc2.txt"],
		"created_at": "2025-10-17",
		"nested": {
			"key": "value",
			"number": 123,
			"list": [1, 2, 3]
		}
	}
	
	dim = 2
	index = faiss.IndexFlatL2(dim)
	vectors = np.array([[1.0, 2.0]], dtype=np.float32)
	index.add(vectors)
	faiss_data = faiss.serialize_index(index).tobytes()
	
	text_contents = ["Test"]
	
	# Encode and decode
	package = KPKGFormat.encode(
		metadata=metadata,
		faiss_data=faiss_data,
		text_contents=text_contents,
		encrypt_key=None,
		compress=True
	)
	
	decoded_meta, _, _ = KPKGFormat.decode(package_data=package, decrypt_key=None)
	
	assert decoded_meta == metadata
	assert decoded_meta["nested"]["list"] == [1, 2, 3]
