#!/usr/bin/env python3
"""
Command-line utility for creating and managing protected KORA data packages (.kpkg).
"""

import argparse
import sys
import os


def create_package(args):
    """Create obfuscated KPKG from RAG folder."""
    from kora.obfuscate import create_distributable_package
    
    try:
        result = create_distributable_package(
            rag_dir=args.rag_dir,
            output_path=args.output,
            password=args.password,
            include_source_names=args.include_sources,
            force_rebuild=args.force,
            use_encryption=args.encrypt
        )
        
        # Save password to file if requested and encryption is used
        if args.save_password and result.get('password'):
            password_file = args.output + ".password"
            with open(password_file, 'w') as f:
                f.write(result['password'])
            print(f"\n✓ Password saved to: {password_file}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def test_package(args):
    """Test querying KPKG package."""
    from kora.obfuscate import ObfuscatedVectorStore
    
    try:
        # Load password from file if provided
        password = args.password
        if args.password_file:
            with open(args.password_file, 'r') as f:
                password = f.read().strip()
        
        # Password is optional - only needed for encrypted packages
        print(f"Loading KPKG package: {args.binary}")
        store = ObfuscatedVectorStore(args.binary, password)
        
        if not store.load():
            print("Error: Failed to load KPKG package", file=sys.stderr)
            print("  Hint: If package is encrypted, provide password with --password or --password-file", file=sys.stderr)
            return 1
        
        print(f"✓ Loaded successfully!")
        print(f"  Total chunks: {len(store.metadatas)}")
        
        # Test query if provided
        if args.query:
            print(f"\nTesting query: '{args.query}'")
            results = store.search(args.query, top_k=3)
            print(f"\nTop {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. Source: {r['source']} | Score: {r['score']:.3f}")
                print(f"   Preview: {r['text'][:150]}...")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def info_package(args):
    """Display information about KPKG package."""
    from kora.obfuscate import KPKGFormat
    import struct
    
    try:
        with open(args.binary, 'rb') as f:
            package_data = f.read()
        
        # Parse header to get basic info
        magic = package_data[0:4]
        if magic != b'KORA':
            print(f"Error: Invalid KPKG format (magic mismatch)", file=sys.stderr)
            return 1
        
        version = package_data[4]
        flags = package_data[5]
        is_encrypted = bool(flags & 0x01)
        is_compressed = bool(flags & 0x02)
        
        # Try to decode metadata (may need password if encrypted)
        try:
            metadata, _, _ = KPKGFormat.decode(package_data, decrypt_key=None)
        except ValueError as e:
            if "encrypted" in str(e):
                # Package is encrypted, show limited info
                print(f"Package Information: {args.binary}")
                print(f"  Format: KPKG v{version}")
                print(f"  Encrypted: Yes (password required for full details)")
                print(f"  Compressed: {'Yes' if is_compressed else 'No'}")
                print(f"  File Size: {len(package_data) / (1024*1024):.2f} MB")
                print("\n  Use --password or --password-file to view full details")
                return 0
            raise
        
        # Display full package information
        print(f"Package Information: {args.binary}")
        print(f"  Format: KPKG v{metadata.get('version', 'unknown')}")
        print(f"  Embedding Model: {metadata.get('embedding_model', 'unknown')}")
        print(f"  Total Chunks: {metadata.get('num_chunks', 0)}")
        print(f"  Data Hash: {metadata.get('hash', 'N/A')}")
        print(f"  Encrypted: {'Yes' if metadata.get('encrypted') else 'No'}")
        print(f"  Compressed: {'Yes' if is_compressed else 'No'}")
        print(f"  Obfuscated: Yes")
        print(f"  File Size: {len(package_data) / (1024*1024):.2f} MB")
        
        # Show unique sources
        if 'metadata' in metadata:
            sources = set(m['source'] for m in metadata['metadata'])
            print(f"  Unique Sources: {len(sources)}")
            if not args.no_sources:
                print("\n  Sources:")
                for src in sorted(sources):
                    print(f"    - {src}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="KORA Hide - Create distributable encrypted KORA packages (.kpkg)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create obfuscated-only package (AI can read without key)
  kora-hide create --output kora_data.kpkg
  
  # Create encrypted package (requires key to decompile)
  kora-hide create --output kora_data.kpkg --encrypt --save-password
  
  # Create with custom password and anonymize sources
  kora-hide create -o data.kpkg -p mypassword --encrypt --no-include-sources
  
  # Test the package
  kora-hide test data.kpkg --password-file data.kpkg.password --query "machine learning"
  
  # View package info
  kora-hide info data.kpkg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create obfuscated package')
    create_parser.add_argument('-r', '--rag-dir', default='RAG', 
                              help='Path to RAG folder (default: RAG)')
    create_parser.add_argument('-o', '--output', default='kora_data.kpkg',
                              help='Output KORA package file (default: kora_data.kpkg)')
    create_parser.add_argument('-p', '--password', 
                              help='Encryption password (only used with --encrypt)')
    create_parser.add_argument('--encrypt', action='store_true',
                              help='Add encryption layer (requires key to decompile)')
    create_parser.add_argument('--save-password', action='store_true',
                              help='Save password to [output].password file (only for encrypted packages)')
    create_parser.add_argument('--no-include-sources', dest='include_sources', 
                              action='store_false', default=True,
                              help='Anonymize source filenames')
    create_parser.add_argument('-f', '--force', action='store_true',
                              help='Force rebuild index')
    create_parser.set_defaults(func=create_package)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test obfuscated package')
    test_parser.add_argument('binary', help='Path to KORA package file (.kpkg)')
    test_parser.add_argument('-p', '--password', help='Decryption password')
    test_parser.add_argument('--password-file', help='File containing password')
    test_parser.add_argument('-q', '--query', help='Test query to run')
    test_parser.set_defaults(func=test_package)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    info_parser.add_argument('binary', help='Path to KORA package file (.kpkg)')
    info_parser.add_argument('--no-sources', action='store_true',
                            help='Hide source list')
    info_parser.set_defaults(func=info_package)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
