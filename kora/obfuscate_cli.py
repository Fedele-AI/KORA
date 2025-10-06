#!/usr/bin/env python3
"""
Command-line utility for creating and managing obfuscated KORA data packages.
"""

import argparse
import sys
import os


def create_package(args):
    """Create obfuscated binary from RAG folder."""
    from kora.obfuscate import create_distributable_package
    
    try:
        result = create_distributable_package(
            rag_dir=args.rag_dir,
            output_path=args.output,
            password=args.password,
            include_source_names=args.include_sources,
            force_rebuild=args.force
        )
        
        # Save password to file if requested
        if args.save_password:
            password_file = args.output + ".password"
            with open(password_file, 'w') as f:
                f.write(result['password'])
            print(f"\n✓ Password saved to: {password_file}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def test_package(args):
    """Test querying obfuscated package."""
    from kora.obfuscate import ObfuscatedVectorStore
    
    try:
        # Load password from file if provided
        password = args.password
        if args.password_file:
            with open(args.password_file, 'r') as f:
                password = f.read().strip()
        
        if not password:
            print("Error: Password required (use --password or --password-file)", file=sys.stderr)
            return 1
        
        print(f"Loading obfuscated package: {args.binary}")
        store = ObfuscatedVectorStore(args.binary, password)
        
        if not store.load():
            print("Error: Failed to load obfuscated package", file=sys.stderr)
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
    """Display information about obfuscated package."""
    import pickle
    
    try:
        with open(args.binary, 'rb') as f:
            package = pickle.load(f)
        
        print(f"Package Information: {args.binary}")
        print(f"  Version: {package.get('version', 'unknown')}")
        print(f"  Embedding Model: {package.get('embedding_model', 'unknown')}")
        print(f"  Total Chunks: {package.get('num_chunks', 0)}")
        print(f"  Data Hash: {package.get('hash', 'N/A')}")
        print(f"  File Size: {os.path.getsize(args.binary) / (1024*1024):.2f} MB")
        
        # Show unique sources (without decrypting text)
        if 'metadata' in package:
            sources = set(m['source'] for m in package['metadata'])
            print(f"  Unique Sources: {len(sources)}")
            if not args.no_sources:
                print("\n  Sources:")
                for src in sorted(sources):
                    print(f"    - {src}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="KORA Obfuscation Tool - Create distributable encrypted data packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create obfuscated package from RAG folder
  python -m kora.obfuscate_cli create --output kora_data.bin --save-password
  
  # Create with custom password and anonymize sources
  python -m kora.obfuscate_cli create -o data.bin -p mypassword --no-include-sources
  
  # Test the package
  python -m kora.obfuscate_cli test data.bin --password-file data.bin.password --query "machine learning"
  
  # View package info
  python -m kora.obfuscate_cli info data.bin
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create obfuscated package')
    create_parser.add_argument('-r', '--rag-dir', default='RAG', 
                              help='Path to RAG folder (default: RAG)')
    create_parser.add_argument('-o', '--output', default='kora_data.bin',
                              help='Output binary file (default: kora_data.bin)')
    create_parser.add_argument('-p', '--password', 
                              help='Encryption password (auto-generated if not provided)')
    create_parser.add_argument('--save-password', action='store_true',
                              help='Save password to [output].password file')
    create_parser.add_argument('--no-include-sources', dest='include_sources', 
                              action='store_false', default=True,
                              help='Anonymize source filenames')
    create_parser.add_argument('-f', '--force', action='store_true',
                              help='Force rebuild index')
    create_parser.set_defaults(func=create_package)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test obfuscated package')
    test_parser.add_argument('binary', help='Path to obfuscated binary file')
    test_parser.add_argument('-p', '--password', help='Decryption password')
    test_parser.add_argument('--password-file', help='File containing password')
    test_parser.add_argument('-q', '--query', help='Test query to run')
    test_parser.set_defaults(func=test_package)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    info_parser.add_argument('binary', help='Path to obfuscated binary file')
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
