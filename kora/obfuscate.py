"""
Obfuscation module for KORA.
Creates a distributable binary format that preserves embeddings and limited metadata
but prevents direct access to original document content.
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy as np
import faiss

from .store import VectorStore


class ObfuscatedStore:
    """
    Stores embeddings and encrypted minimal metadata in a single binary file.
    Original text chunks are encrypted and can only be accessed through the RAG pipeline.
    """
    
    def __init__(self, binary_path: str, password: str = None):
        """
        Args:
            binary_path: Path to the obfuscated binary file
            password: Optional password for additional encryption layer (auto-generated if None)
        """
        self.binary_path = binary_path
        self.password = password or self._generate_key()
        self.cipher = self._create_cipher(self.password)
        
    def _generate_key(self) -> str:
        """Generate a random key for encryption."""
        return Fernet.generate_key().decode('utf-8')
    
    def _create_cipher(self, password: str) -> Fernet:
        """Create encryption cipher from password."""
        # Derive a key from password
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        # Use PBKDF2 to derive a key from password
        salt = b'kora_salt_v1'  # Fixed salt for deterministic key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password)
        
        # Fernet requires base64-encoded 32-byte key
        from base64 import urlsafe_b64encode
        fernet_key = urlsafe_b64encode(key)
        return Fernet(fernet_key)
    
    def obfuscate_from_store(self, store: VectorStore, include_source_names: bool = True) -> Dict[str, Any]:
        """
        Create obfuscated binary from existing VectorStore.
        
        Args:
            store: Loaded VectorStore instance
            include_source_names: If True, keeps source filenames (not content)
                                 If False, replaces with anonymous IDs
        
        Returns:
            Dictionary with obfuscation statistics
        """
        if store.index is None or len(store.metadatas) == 0:
            raise ValueError("VectorStore is empty. Build index first.")
        
        # Extract FAISS index (embeddings only)
        index_data = faiss.serialize_index(store.index)
        
        # Prepare minimal metadata (encrypted)
        obfuscated_metadata = []
        source_map = {}  # Map real source names to anonymous IDs
        source_counter = 0
        
        for meta in store.metadatas:
            # Encrypt the actual text content
            encrypted_text = self.cipher.encrypt(meta['text'].encode('utf-8'))
            
            # Handle source naming
            if include_source_names:
                source_identifier = os.path.basename(meta['source'])
            else:
                # Create anonymous source IDs
                real_source = meta['source']
                if real_source not in source_map:
                    source_map[real_source] = f"DOC_{source_counter:04d}"
                    source_counter += 1
                source_identifier = source_map[real_source]
            
            obfuscated_metadata.append({
                'source': source_identifier,
                'chunk_id': meta['chunk_id'],
                'encrypted_text': encrypted_text.decode('utf-8')  # Store as base64 string
            })
        
        # Create the obfuscated package
        package = {
            'version': '1.0',
            'embedding_model': store.embedding_model_name,
            'faiss_index': index_data,
            'metadata': obfuscated_metadata,
            'num_chunks': len(obfuscated_metadata),
            'hash': hashlib.sha256(index_data).hexdigest()[:16]
        }
        
        # Serialize and save
        with open(self.binary_path, 'wb') as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Return statistics
        file_size_mb = os.path.getsize(self.binary_path) / (1024 * 1024)
        return {
            'status': 'success',
            'output_file': self.binary_path,
            'file_size_mb': round(file_size_mb, 2),
            'num_chunks': len(obfuscated_metadata),
            'num_unique_sources': len(set(m['source'] for m in obfuscated_metadata)),
            'password': self.password if isinstance(self.password, str) else self.password.decode('utf-8')
        }
    
    def load(self) -> Tuple[faiss.Index, List[Dict[str, Any]], str]:
        """
        Load the obfuscated binary and return FAISS index and metadata.
        
        Returns:
            Tuple of (faiss_index, metadata_list, embedding_model_name)
        """
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Obfuscated binary not found: {self.binary_path}")
        
        with open(self.binary_path, 'rb') as f:
            package = pickle.load(f)
        
        # Deserialize FAISS index
        faiss_index = faiss.deserialize_index(package['faiss_index'])
        
        # Keep metadata encrypted (will decrypt on-demand during search)
        metadata = package['metadata']
        model_name = package['embedding_model']
        
        return faiss_index, metadata, model_name
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt a single text chunk (used during retrieval)."""
        encrypted_bytes = encrypted_text.encode('utf-8')
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')


class ObfuscatedVectorStore(VectorStore):
    """
    VectorStore variant that loads from obfuscated binary instead of RAG folder.
    Provides same search interface but prevents direct document access.
    """
    
    def __init__(self, obfuscated_path: str, password: str, **kwargs):
        """
        Args:
            obfuscated_path: Path to the obfuscated binary file
            password: Password for decryption
        """
        self.obfuscated_store = ObfuscatedStore(obfuscated_path, password)
        # Initialize parent without building
        super().__init__(index_dir=".kora/obfuscated_temp", **kwargs)
        
    def load(self) -> bool:
        """Load from obfuscated binary."""
        try:
            faiss_index, metadata, model_name = self.obfuscated_store.load()
            self.index = faiss_index
            self.embedding_model_name = model_name
            # Store encrypted metadata
            self._encrypted_metadata = metadata
            # Create decrypted version for search
            self.metadatas = []
            for enc_meta in metadata:
                decrypted_text = self.obfuscated_store.decrypt_text(enc_meta['encrypted_text'])
                self.metadatas.append({
                    'source': enc_meta['source'],
                    'chunk_id': enc_meta['chunk_id'],
                    'text': decrypted_text
                })
            return True
        except Exception as e:
            print(f"Failed to load obfuscated store: {e}")
            return False


def create_distributable_package(
    rag_dir: str = "RAG",
    output_path: str = "kora_data.bin",
    password: str = None,
    include_source_names: bool = True,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Main entry point: Create a distributable obfuscated binary from RAG folder.
    
    Args:
        rag_dir: Path to RAG folder with source documents
        output_path: Output path for the obfuscated binary
        password: Optional encryption password (auto-generated if None)
        include_source_names: Whether to keep original filenames visible
        force_rebuild: Force rebuilding the index
    
    Returns:
        Dictionary with creation statistics and password
    """
    print("Loading vector index...")
    from .store import VectorStore
    
    # Use VectorStore with default index directory
    index_dir = ".kora/index"
    store = VectorStore(index_dir=index_dir)
    
    # Try to load existing index
    if store.load():
        print(f"✓ Loaded existing index with {len(store.metadatas)} chunks")
        status = "loaded from cache"
    elif force_rebuild:
        # Only rebuild if explicitly forced
        print("Building new index from RAG folder...")
        print("  (This may take several minutes for large document collections)")
        from .rag import build_or_load_index
        store, status = build_or_load_index(force_rebuild=True, store=store)
        print(f"✓ Built index with {len(store.metadatas)} chunks")
    else:
        raise ValueError(
            "No existing index found. Please run KORA once to build the index first, "
            "or use the -f/--force flag to build a new index."
        )
    
    print(f"Index {status}. Creating obfuscated binary...")
    
    # Create obfuscated version
    obfuscator = ObfuscatedStore(output_path, password)
    result = obfuscator.obfuscate_from_store(store, include_source_names)
    
    print(f"✓ Obfuscated binary created: {output_path}")
    print(f"  Size: {result['file_size_mb']} MB")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Sources: {result['num_unique_sources']}")
    print(f"\n⚠️  IMPORTANT: Save this password to use the binary:")
    print(f"  Password: {result['password']}")
    
    return result
