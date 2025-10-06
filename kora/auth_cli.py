#!/usr/bin/env python3
"""
Command-line interface for KORA API key management.
"""

import argparse
import sys
from typing import Optional

from .auth import get_authenticator


def generate_api_key(username: str, password: Optional[str] = None, demo_mode: bool = False) -> None:
    """Generate an API key for a user."""
    if not password and not demo_mode:
        import getpass
        password = getpass.getpass(f"Password for {username}: ")
    elif demo_mode and not password:
        password = "demo_password"
    
    auth = get_authenticator()
    api_key = auth.generate_api_key(username, password, demo_mode=demo_mode)
    
    if api_key:
        print(f"✅ API key generated successfully for {username}:")
        print(f"API Key: {api_key}")
        if demo_mode:
            print("\n🧪 DEMO MODE: This key was generated without Kerberos authentication")
        print("\n🔐 Please save this API key securely. You will need it to access KORA.")
        print("📋 This API key can be used in the web interface or for programmatic access.")
    else:
        print(f"❌ Failed to generate API key for {username}")
        if not demo_mode:
            print("Please check your credentials and ensure Kerberos is properly configured.")
            print("For testing, you can use --demo flag to skip Kerberos authentication.")
        sys.exit(1)


def validate_api_key(api_key: str) -> None:
    """Validate an API key."""
    auth = get_authenticator()
    
    if auth.validate_api_key(api_key):
        # Get additional info
        api_keys = auth._load_api_keys()
        key_data = api_keys.get(api_key, {})
        username = key_data.get("username", "Unknown")
        created_at = key_data.get("created_at", "Unknown")
        
        print(f"✅ API key is valid")
        print(f"Username: {username}")
        print(f"Created: {created_at}")
    else:
        print("❌ API key is invalid or expired")
        sys.exit(1)


def revoke_api_key(api_key: str) -> None:
    """Revoke an API key."""
    auth = get_authenticator()
    
    if auth.revoke_api_key(api_key):
        print(f"✅ API key revoked successfully")
    else:
        print("❌ Failed to revoke API key (key not found)")
        sys.exit(1)


def list_api_keys() -> None:
    """List all API keys."""
    auth = get_authenticator()
    api_keys = auth._load_api_keys()
    
    if not api_keys:
        print("No API keys found.")
        return
    
    print("API Keys:")
    print("-" * 80)
    print(f"{'API Key':<20} {'Username':<15} {'Status':<10} {'Created':<20}")
    print("-" * 80)
    
    for api_key, data in api_keys.items():
        username = data.get("username", "Unknown")
        status = "Active" if data.get("active", False) else "Revoked"
        created = data.get("created_at", "Unknown")
        
        # Truncate API key for display
        display_key = api_key[:16] + "..."
        
        print(f"{display_key:<20} {username:<15} {status:<10} {created:<20}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KORA API Key Management",
        prog="kora-auth"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate API key
    gen_parser = subparsers.add_parser("generate", help="Generate a new API key")
    gen_parser.add_argument("username", help="Kerberos username")
    gen_parser.add_argument("--password", help="Password (will prompt if not provided)")
    gen_parser.add_argument("--demo", action="store_true", help="Demo mode (skip Kerberos authentication)")
    
    # Validate API key
    val_parser = subparsers.add_parser("validate", help="Validate an API key")
    val_parser.add_argument("api_key", help="API key to validate")
    
    # Revoke API key
    rev_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    rev_parser.add_argument("api_key", help="API key to revoke")
    
    # List API keys
    list_parser = subparsers.add_parser("list", help="List all API keys")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "generate":
            generate_api_key(args.username, args.password, getattr(args, 'demo', False))
        elif args.command == "validate":
            validate_api_key(args.api_key)
        elif args.command == "revoke":
            revoke_api_key(args.api_key)
        elif args.command == "list":
            list_api_keys()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()