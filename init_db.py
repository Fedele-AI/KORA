#!/usr/bin/env python3
"""
Initialize KORA database
Generates Prisma client and applies migrations
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict

def run_command(cmd: list, env: Optional[Dict[str, str]] = None):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    if result.stdout:
        print(result.stdout)
    return result
    return result


def main():
    # Get project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Set up environment
    env = os.environ.copy()
    
    # Use SQLite by default
    if "DATABASE_URL" not in env:
        data_dir = os.path.expanduser("~/.kora")
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "kora.db")
        env["DATABASE_URL"] = f"file:{db_path}"
    
    print("=" * 60)
    print("  KORA Database Initialization")
    print("=" * 60)
    print(f"\nDatabase URL: {env['DATABASE_URL']}")
    print()
    
    # Check if prisma is available
    try:
        subprocess.run(["prisma", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing Prisma CLI...")
        run_command(["npm", "install", "-g", "prisma"], env=env)
    
    # Generate Prisma client
    print("\n" + "=" * 60)
    print("  Generating Prisma Client")
    print("=" * 60 + "\n")
    run_command(["prisma", "generate"], env=env)
    
    # Push schema to database (creates tables)
    print("\n" + "=" * 60)
    print("  Creating Database Schema")
    print("=" * 60 + "\n")
    run_command(["prisma", "db", "push", "--skip-generate"], env=env)
    
    print("\n" + "=" * 60)
    print("  ✅ Database Initialized Successfully")
    print("=" * 60)
    print("\nYou can now:")
    print("  - Run KORA with database logging enabled")
    print("  - Use 'prisma studio' to view database contents")
    print("  - Configure DATABASE_URL for PostgreSQL if needed")
    print()


if __name__ == "__main__":
    main()
