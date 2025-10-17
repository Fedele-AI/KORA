"""
Protected package module for KORA.
Creates distributable KORA packages (.kpkg) using a custom binary format.
- AI can read the format without keys (obfuscated but decodable)
- Optional encryption layer for security (requires key to decompile)
"""

import os
import json
import pickle
import hashlib
import struct
import zlib
from typing import List, Dict, Any, Tuple, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy as np
import faiss

from .store import VectorStore

# KORA Package Format Constants
KPKG_MAGIC = b'KORA'
KPKG_VERSION = 1
KPKG_FLAG_ENCRYPTED = 0x01
KPKG_FLAG_COMPRESSED = 0x02


class KPKGFormat:
    """
    Custom binary format for KORA packages.
    
    Format Structure:
    [Header]
    - Magic: 4 bytes (KORA)
    - Version: 1 byte
    - Flags: 1 byte (bit 0: encrypted, bit 1: compressed)
    - Reserved: 2 bytes
    
    [Metadata Section]
    - Metadata Length: 4 bytes (uint32)
    - Metadata JSON (optionally encrypted/compressed)
    
    [Embeddings Section]
    - Embeddings Length: 4 bytes (uint32)
    - FAISS Index Bytes (optionally encrypted/compressed)
    
    [Content Section]
    - Content Length: 4 bytes (uint32)
    - Text Content (always obfuscated, optionally encrypted)
    """
    
    @staticmethod
    def _obfuscate_bytes(data: bytes) -> bytes:
        """Simple XOR obfuscation with a deterministic pattern."""
        key = b'KORA_OBFUSCATION_KEY_V1'
        key_len = len(key)
        return bytes(b ^ key[i % key_len] for i, b in enumerate(data))
    
    @staticmethod
    def _deobfuscate_bytes(data: bytes) -> bytes:
        """Reverse XOR obfuscation."""
        return KPKGFormat._obfuscate_bytes(data)  # XOR is symmetric
    
    @staticmethod
    def encode(
        metadata: Dict[str, Any],
        faiss_data: bytes,
        text_contents: List[str],
        encrypt_key: Optional[str] = None,
        compress: bool = True
    ) -> bytes:
        """
        Encode data into KPKG format.
        
        Args:
            metadata: Package metadata (model info, stats, etc.)
            faiss_data: Serialized FAISS index
            text_contents: List of text chunks (to be obfuscated)
            encrypt_key: Optional encryption key for additional security
            compress: Whether to compress the data
        """
        flags = 0
        if encrypt_key:
            flags |= KPKG_FLAG_ENCRYPTED
        if compress:
            flags |= KPKG_FLAG_COMPRESSED
        
        # Prepare metadata
        metadata_json = json.dumps(metadata).encode('utf-8')
        if compress:
            metadata_json = zlib.compress(metadata_json)
        if encrypt_key:
            cipher = KPKGFormat._create_cipher(encrypt_key)
            metadata_json = cipher.encrypt(metadata_json)
        
        # Prepare embeddings (FAISS index)
        embeddings_data = faiss_data
        if compress:
            embeddings_data = zlib.compress(embeddings_data)
        if encrypt_key:
            cipher = KPKGFormat._create_cipher(encrypt_key)
            embeddings_data = cipher.encrypt(embeddings_data)
        
        # Prepare content (always obfuscated, optionally encrypted)
        content_json = json.dumps(text_contents).encode('utf-8')
        content_data = KPKGFormat._obfuscate_bytes(content_json)
        if compress:
            content_data = zlib.compress(content_data)
        if encrypt_key:
            cipher = KPKGFormat._create_cipher(encrypt_key)
            content_data = cipher.encrypt(content_data)
        
        # Build package
        package = bytearray()
        
        # Header
        package.extend(KPKG_MAGIC)
        package.append(KPKG_VERSION)
        package.append(flags)
        package.extend(b'\x00\x00')  # Reserved
        
        # Metadata section
        package.extend(struct.pack('<I', len(metadata_json)))
        package.extend(metadata_json)
        
        # Embeddings section
        package.extend(struct.pack('<I', len(embeddings_data)))
        package.extend(embeddings_data)
        
        # Content section
        package.extend(struct.pack('<I', len(content_data)))
        package.extend(content_data)
        
        return bytes(package)
    
    @staticmethod
    def decode(
        package_data: bytes,
        decrypt_key: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bytes, List[str]]:
        """
        Decode KPKG format.
        
        Args:
            package_data: Raw KPKG bytes
            decrypt_key: Optional decryption key (if package is encrypted)
            
        Returns:
            Tuple of (metadata_dict, faiss_bytes, text_contents_list)
        """
        offset = 0
        
        # Parse header
        magic = package_data[offset:offset+4]
        if magic != KPKG_MAGIC:
            raise ValueError(f"Invalid KPKG format: magic mismatch (got {magic})")
        offset += 4
        
        version = package_data[offset]
        if version != KPKG_VERSION:
            raise ValueError(f"Unsupported KPKG version: {version}")
        offset += 1
        
        flags = package_data[offset]
        is_encrypted = bool(flags & KPKG_FLAG_ENCRYPTED)
        is_compressed = bool(flags & KPKG_FLAG_COMPRESSED)
        offset += 1
        
        offset += 2  # Skip reserved bytes
        
        # Check encryption requirements
        if is_encrypted and not decrypt_key:
            raise ValueError("Package is encrypted but no decryption key provided")
        
        cipher = None
        if is_encrypted and decrypt_key:
            cipher = KPKGFormat._create_cipher(decrypt_key)
        
        # Parse metadata section
        metadata_len = struct.unpack('<I', package_data[offset:offset+4])[0]
        offset += 4
        metadata_data = package_data[offset:offset+metadata_len]
        offset += metadata_len
        
        if cipher:
            metadata_data = cipher.decrypt(metadata_data)
        if is_compressed:
            metadata_data = zlib.decompress(metadata_data)
        metadata = json.loads(metadata_data.decode('utf-8'))
        
        # Parse embeddings section
        embeddings_len = struct.unpack('<I', package_data[offset:offset+4])[0]
        offset += 4
        embeddings_data = package_data[offset:offset+embeddings_len]
        offset += embeddings_len
        
        if cipher:
            embeddings_data = cipher.decrypt(embeddings_data)
        if is_compressed:
            embeddings_data = zlib.decompress(embeddings_data)
        
        # Parse content section
        content_len = struct.unpack('<I', package_data[offset:offset+4])[0]
        offset += 4
        content_data = package_data[offset:offset+content_len]
        
        if cipher:
            content_data = cipher.decrypt(content_data)
        if is_compressed:
            content_data = zlib.decompress(content_data)
        # Always deobfuscate content
        content_data = KPKGFormat._deobfuscate_bytes(content_data)
        text_contents = json.loads(content_data.decode('utf-8'))
        
        return metadata, embeddings_data, text_contents
    
    @staticmethod
    def _create_cipher(key: str) -> Fernet:
        """Create Fernet cipher from key."""
        key_bytes: bytes
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        salt = b'kora_salt_v1'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        derived_key = kdf.derive(key_bytes)
        
        from base64 import urlsafe_b64encode
        fernet_key = urlsafe_b64encode(derived_key)
        return Fernet(fernet_key)


class ObfuscatedStore:
    """
    Stores embeddings and obfuscated content in custom KPKG format.
    - Content is always obfuscated (readable by AI without key)
    - Optional encryption layer for security (requires key to decompile)
    """
    
    def __init__(self, binary_path: str, password: Optional[str] = None):
        """
        Args:
            binary_path: Path to the .kpkg file
            password: Optional password for encryption layer (None = obfuscated only)
        """
        self.binary_path = binary_path
        self.password = password
        
    def _generate_key(self) -> str:
        """Generate a random key for encryption."""
        return Fernet.generate_key().decode('utf-8')
    
    def obfuscate_from_store(self, store: VectorStore, include_source_names: bool = True, use_encryption: bool = False) -> Dict[str, Any]:
        """
        Create KPKG package from existing VectorStore.
        
        Args:
            store: Loaded VectorStore instance
            include_source_names: If True, keeps source filenames
            use_encryption: If True, adds encryption layer (requires password)
        
        Returns:
            Dictionary with package statistics
        """
        if store.index is None or len(store.metadatas) == 0:
            raise ValueError("VectorStore is empty. Build index first.")
        
        # Extract FAISS index (serialize to bytes)
        index_data_bytes: bytes = faiss.serialize_index(store.index)  # type: ignore
        
        # Prepare text contents and metadata
        text_contents = []
        source_map = {}
        source_counter = 0
        
        metadata_list = []
        for meta in store.metadatas:
            # Handle source naming
            if include_source_names:
                source_identifier = os.path.basename(meta['source'])
            else:
                real_source = meta['source']
                if real_source not in source_map:
                    source_map[real_source] = f"DOC_{source_counter:04d}"
                    source_counter += 1
                source_identifier = source_map[real_source]
            
            text_contents.append(meta['text'])
            metadata_list.append({
                'source': source_identifier,
                'chunk_id': meta['chunk_id'],
                'index': len(text_contents) - 1
            })
        
        # Prepare package metadata
        package_metadata = {
            'version': '2.0',
            'format': 'kpkg',
            'embedding_model': store.embedding_model_name,
            'num_chunks': len(text_contents),
            'metadata': metadata_list,
            'hash': hashlib.sha256(index_data_bytes).hexdigest()[:16],
            'encrypted': use_encryption
        }
        
        # Generate encryption key if needed
        encryption_key = None
        if use_encryption:
            encryption_key = self.password or self._generate_key()
        
        # Encode into KPKG format
        kpkg_data = KPKGFormat.encode(
            metadata=package_metadata,
            faiss_data=index_data_bytes,
            text_contents=text_contents,
            encrypt_key=encryption_key,
            compress=True
        )
        
        # Write to file
        with open(self.binary_path, 'wb') as f:
            f.write(kpkg_data)
        
        # Return statistics
        file_size_mb = len(kpkg_data) / (1024 * 1024)
        result = {
            'status': 'success',
            'output_file': self.binary_path,
            'file_size_mb': round(file_size_mb, 2),
            'num_chunks': len(text_contents),
            'num_unique_sources': len(set(m['source'] for m in metadata_list)),
            'encrypted': use_encryption,
            'obfuscated': True
        }
        
        if use_encryption and encryption_key:
            result['password'] = encryption_key if isinstance(encryption_key, str) else encryption_key.decode('utf-8')
        
        return result
    
    def load(self, decrypt_key: Optional[str] = None) -> Tuple[faiss.Index, List[Dict[str, Any]], str]:
        """
        Load KPKG package.
        
        Args:
            decrypt_key: Optional decryption key (if package is encrypted)
        
        Returns:
            Tuple of (faiss_index, metadata_list, embedding_model_name)
        """
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"KPKG file not found: {self.binary_path}")
        
        with open(self.binary_path, 'rb') as f:
            kpkg_data = f.read()
        
        # Decode KPKG format
        try:
            metadata, faiss_data, text_contents = KPKGFormat.decode(
                package_data=kpkg_data,
                decrypt_key=decrypt_key or self.password
            )
        except ValueError as e:
            if "encrypted but no decryption key" in str(e):
                # Try without key (obfuscated only)
                metadata, faiss_data, text_contents = KPKGFormat.decode(
                    package_data=kpkg_data,
                    decrypt_key=None
                )
            else:
                raise
        
        # Deserialize FAISS index - convert bytes to numpy array first
        faiss_data_array = np.frombuffer(faiss_data, dtype=np.uint8)
        faiss_index = faiss.deserialize_index(faiss_data_array)
        
        # Reconstruct metadata with text
        full_metadata = []
        for meta in metadata['metadata']:
            full_metadata.append({
                'source': meta['source'],
                'chunk_id': meta['chunk_id'],
                'text': text_contents[meta['index']]
            })
        
        model_name = metadata['embedding_model']
        
        return faiss_index, full_metadata, model_name


class ObfuscatedVectorStore(VectorStore):
    """
    VectorStore variant that loads from KPKG file instead of RAG folder.
    Provides same search interface but with obfuscated content.
    """
    
    def __init__(self, obfuscated_path: str, password: Optional[str] = None, **kwargs):
        """
        Args:
            obfuscated_path: Path to the .kpkg file
            password: Optional password for encrypted packages
        """
        self.obfuscated_store = ObfuscatedStore(obfuscated_path, password)
        # Initialize parent without building
        super().__init__(index_dir=".kora/obfuscated_temp", **kwargs)
        
    def load(self) -> bool:
        """Load from KPKG file."""
        try:
            faiss_index, metadata, model_name = self.obfuscated_store.load()
            self.index = faiss_index  # type: ignore
            self.embedding_model_name = model_name
            # Metadata already has text decoded
            self.metadatas = metadata
            return True
        except Exception as e:
            print(f"Failed to load KPKG: {e}")
            import traceback
            traceback.print_exc()
            return False


def create_distributable_package(
    rag_dir: str = "RAG",
    output_path: str = "kora_data.kpkg",
    password: Optional[str] = None,
    include_source_names: bool = True,
    force_rebuild: bool = False,
    use_encryption: bool = False
) -> Dict[str, Any]:
    """
    Main entry point: Create a distributable KORA package from RAG folder.
    
    Args:
        rag_dir: Path to RAG folder with source documents
        output_path: Output path for the KORA package (.kpkg)
        password: Optional encryption password (only used if use_encryption=True)
        include_source_names: Whether to keep original filenames visible
        force_rebuild: Force rebuilding the index
        use_encryption: If True, adds encryption layer (requires key to decompile)
    
    Returns:
        Dictionary with creation statistics
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
    
    encryption_status = "with encryption" if use_encryption else "obfuscated only"
    print(f"Index {status}. Creating KPKG package ({encryption_status})...")
    
    # Create KPKG package
    obfuscator = ObfuscatedStore(output_path, password)
    result = obfuscator.obfuscate_from_store(store, include_source_names, use_encryption)
    
    print(f"✓ KPKG package created: {output_path}")
    print(f"  Size: {result['file_size_mb']} MB")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Sources: {result['num_unique_sources']}")
    print(f"  Obfuscated: {result['obfuscated']}")
    print(f"  Encrypted: {result['encrypted']}")
    
    if result.get('password'):
        print(f"\n⚠️  IMPORTANT: Save this password to decompile the package:")
        print(f"  Password: {result['password']}")
    else:
        print(f"\n✓ Package is obfuscated but readable by KORA AI without additional keys")
    
    return result
