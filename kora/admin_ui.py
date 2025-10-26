"""
Admin Web GUI for KORA - API Key Management, Package Management, and Ollama Control
"""
import gradio as gr
import subprocess
import os
import secrets
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time

from .auth import get_authenticator


# Global admin token - generated on startup
ADMIN_TOKEN = None


def generate_admin_token() -> str:
	"""Generate a secure admin token."""
	return secrets.token_urlsafe(32)


def get_admin_token() -> str:
	"""Get or create the admin token."""
	global ADMIN_TOKEN
	if ADMIN_TOKEN is None:
		ADMIN_TOKEN = generate_admin_token()
	return ADMIN_TOKEN


def verify_admin_token(token: str) -> bool:
	"""Verify the admin token."""
	return token == get_admin_token()


# ============================================================================
# API Key Management Functions
# ============================================================================

def list_api_keys(admin_token: str) -> str:
	"""List all API keys."""
	if not verify_admin_token(admin_token):
		return "[X] Invalid admin token"
	
	auth = get_authenticator()
	api_keys = auth._load_api_keys()
	
	if not api_keys:
		return '<div style="text-align: center; padding: 20px; color: #888;">No API keys found</div>'
	
	result = '<div style="max-width: 100%; overflow-x: auto;">'
	for api_key, data in api_keys.items():
		username = data.get("username", "Unknown")
		created_timestamp = data.get("created_at", "Unknown")
		
		# Format timestamp to human-readable format
		try:
			# Convert to float if it's a string representation of a number
			if isinstance(created_timestamp, str):
				created_timestamp = float(created_timestamp)
			
			if isinstance(created_timestamp, (int, float)):
				dt = datetime.fromtimestamp(created_timestamp)
				created_at = dt.strftime('%B %d, %Y at %I:%M %p')
			else:
				created_at = str(created_timestamp)
		except (ValueError, TypeError, OSError):
			# If conversion fails, just use the raw value
			created_at = str(created_timestamp)
		
		result += f'''
		<div style="
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			border-radius: 12px;
			padding: 20px;
			margin-bottom: 15px;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			transition: transform 0.2s;
		" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
			<div style="color: white; font-weight: bold; font-size: 16px; margin-bottom: 10px;">
				{username}
			</div>
			<div style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin-bottom: 8px;">
				<strong>Created:</strong> {created_at}
			</div>
			<div style="display: flex; align-items: center; gap: 10px;">
				<code style="
					background: rgba(255, 255, 255, 0.2);
					padding: 8px 12px;
					border-radius: 6px;
					color: white;
					font-size: 13px;
					flex-grow: 1;
					overflow-x: auto;
				">{api_key[:16]}...{api_key[-8:]}</code>
				<button onclick="navigator.clipboard.writeText('{api_key}'); this.innerText='[+] Copied!'; setTimeout(() => this.innerText='Copy Full Key', 1000)" 
					style="
						background: rgba(255, 255, 255, 0.3);
						color: white;
						border: none;
						padding: 8px 16px;
						border-radius: 6px;
						cursor: pointer;
						font-size: 12px;
						white-space: nowrap;
					">Copy Full Key</button>
			</div>
		</div>
		'''
	
	result += '</div>'
	return result


def create_api_key(admin_token: str, username: str) -> str:
	"""Generate a new API key."""
	if not verify_admin_token(admin_token):
		return "[X] Invalid admin token"
	
	if not username:
		return "[X] Please provide a username"
	
	auth = get_authenticator()
	api_key = auth.generate_api_key(username)
	
	dt = datetime.now()
	created_at = dt.strftime('%B %d, %Y at %I:%M %p')
	
	return f'''
	<div style="
		background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
		border-radius: 12px;
		padding: 25px;
		color: white;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
		animation: slideIn 0.3s ease-out;
	">
		<div style="font-size: 20px; font-weight: bold; margin-bottom: 15px;">
			[+] API Key Generated Successfully!
		</div>
		<div style="margin-bottom: 10px;">
			<strong>Username:</strong> {username}
		</div>
		<div style="margin-bottom: 15px;">
			<strong>Created:</strong> {created_at}
		</div>
		<div style="
			background: rgba(255, 255, 255, 0.2);
			padding: 15px;
			border-radius: 8px;
			margin-bottom: 10px;
		">
			<div style="font-size: 12px; margin-bottom: 8px; opacity: 0.9;">API Key:</div>
			<code style="
				font-size: 14px;
				word-break: break-all;
				display: block;
			">{api_key}</code>
		</div>
		<button onclick="navigator.clipboard.writeText('{api_key}'); this.innerText='[+] Copied to Clipboard!'; setTimeout(() => this.innerText='Copy to Clipboard', 1500)" 
			style="
				background: white;
				color: #11998e;
				border: none;
				padding: 12px 24px;
				border-radius: 8px;
				cursor: pointer;
				font-weight: bold;
				font-size: 14px;
				width: 100%;
			">Copy to Clipboard</button>
		<div style="margin-top: 15px; font-size: 13px; opacity: 0.9;">
			[!] Save this key securely - it won't be shown again!
		</div>
	</div>
	<style>
		@keyframes slideIn {{
			from {{ opacity: 0; transform: translateY(-10px); }}
			to {{ opacity: 1; transform: translateY(0); }}
		}}
	</style>
	'''


def delete_api_key(admin_token: str, api_key: str) -> str:
	"""Revoke an API key."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not api_key:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please provide an API key</div>'
	
	auth = get_authenticator()
	if auth.revoke_api_key(api_key):
		return '<div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: 600; animation: slideIn 0.3s ease-out;">[+] API key revoked successfully</div>'
	else:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] API key not found</div>'


# ============================================================================
# KORA Package Management Functions
# ============================================================================

def list_packages(admin_token: str) -> str:
	"""List all .kpkg files in the project directory with action buttons."""
	if not verify_admin_token(admin_token):
		return "[X] Invalid admin token"
	
	cwd = Path.cwd()
	kpkg_files = list(cwd.glob("*.kpkg"))
	
	if not kpkg_files:
		return '<div style="text-align: center; padding: 20px; color: #888;">No .kpkg packages found in project directory</div>'
	
	result = '<div style="max-width: 100%; overflow-x: auto;">'
	
	for pkg in kpkg_files:
		modified_time = datetime.fromtimestamp(pkg.stat().st_mtime).strftime('%B %d, %Y')
		pkg_path = str(pkg.absolute())
		pkg_name = pkg.name
		result += f'''
		<div style="
			background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
			border-radius: 12px;
			padding: 20px;
			margin-bottom: 15px;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			transition: all 0.2s;
			position: relative;
		">
			<div style="color: white; font-weight: bold; font-size: 16px; margin-bottom: 10px;">
				{pkg_name}
			</div>
			<div style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin-bottom: 5px;">
				<strong>Size:</strong> {pkg.stat().st_size / 1024:.2f} KB
			</div>
			<div style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin-bottom: 12px;">
				<strong>Modified:</strong> {modified_time}
			</div>
			<div style="color: rgba(255, 255, 255, 0.8); font-size: 12px; margin-bottom: 8px; word-break: break-all;">
				{pkg_path}
			</div>
		</div>
		'''
	
	result += '</div>'
	return result


def create_package(admin_token: str, output_name: str, encrypt: bool, password: str) -> str:
	"""Create a new KORA package from RAG/ directory."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not output_name:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please provide an output filename</div>'
	
	if not output_name.endswith(".kpkg"):
		output_name += ".kpkg"
	
	try:
		from .obfuscate import create_distributable_package
		
		rag_dir = Path("RAG")
		if not rag_dir.exists():
			return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] RAG/ directory not found</div>'
		
		# Create package
		result = create_distributable_package(
			rag_dir=str(rag_dir),
			output_path=output_name,
			password=password if encrypt and password else None,
			use_encryption=encrypt,
			include_source_names=True
		)
		
		msg = f'''
		<div style="
			background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
			border-radius: 12px;
			padding: 25px;
			color: white;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			animation: slideIn 0.3s ease-out;
		">
			<div style="font-size: 20px; font-weight: bold; margin-bottom: 15px;">
				[+] Package Created Successfully!
			</div>
			<div style="margin-bottom: 8px;"><strong>File:</strong> {output_name}</div>
			<div style="margin-bottom: 8px;"><strong>Size:</strong> {result.get('file_size_mb', 'Unknown')} MB</div>
			<div style="margin-bottom: 8px;"><strong>Chunks:</strong> {result.get('num_chunks', 'Unknown')}</div>
			<div style="margin-bottom: 15px;"><strong>Encrypted:</strong> {'Yes [ENCRYPTED]' if encrypt else 'No'}</div>
		'''
		
		if result.get('password'):
			msg += f'''
			<div style="
				background: rgba(255, 255, 255, 0.2);
				padding: 15px;
				border-radius: 8px;
				margin-top: 10px;
			">
				<div style="font-size: 14px; margin-bottom: 8px;">Generated Password:</div>
				<code style="font-size: 16px; word-break: break-all;">{result['password']}</code>
			</div>
			<div style="margin-top: 10px; font-size: 13px; opacity: 0.9;">
				[!] Save this password securely - it's required to use the package!
			</div>
			'''
		
		msg += '</div>'
		return msg
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error creating package: {str(e)}</div>'


def get_package_info(admin_token: str, package_path: str) -> str:
	"""Get detailed information about a package using kora-hide info."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not package_path:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please select a package</div>'
	
	# Strip whitespace from path
	package_path = package_path.strip()
	
	try:
		# Use python -m to run kora-hide as a module
		import sys
		result = subprocess.run(
			[sys.executable, "-m", "kora.obfuscate_cli", "info", package_path],
			capture_output=True,
			text=True,
			timeout=30
		)
		
		if result.returncode == 0:
			return f'''
			<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">Package Information</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px; white-space: pre-wrap;">{result.stdout}</pre>
				</div>
			</div>
			'''
		else:
			return f'''
			<div style="background: #f5576c; color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">[X] Failed to Get Package Info</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px;">{result.stderr}</pre>
				</div>
			</div>
			'''
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error: {str(e)}</div>'


def test_package(admin_token: str, package_path: str) -> str:
	"""Test a package using kora-hide test."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not package_path:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please select a package</div>'
	
	# Strip whitespace from path
	package_path = package_path.strip()
	
	try:
		# Use python -m to run kora-hide as a module
		import sys
		result = subprocess.run(
			[sys.executable, "-m", "kora.obfuscate_cli", "test", package_path],
			capture_output=True,
			text=True,
			timeout=60
		)
		
		if result.returncode == 0:
			return f'''
			<div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">[+] Package Test Results</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px; white-space: pre-wrap;">{result.stdout}</pre>
				</div>
			</div>
			'''
		else:
			return f'''
			<div style="background: #f5576c; color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">[X] Package Test Failed</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px;">{result.stderr}</pre>
				</div>
			</div>
			'''
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error: {str(e)}</div>'


def create_package_from_folder(admin_token: str, folder_path: str, output_name: str, encrypt: bool, password: str) -> Tuple[str, Optional[str]]:
	"""Create a KORA package from a selected folder and return file path for download."""
	if not verify_admin_token(admin_token):
		return ('<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>', None)
	
	if not folder_path or not Path(folder_path).exists():
		return ('<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please provide a valid folder path</div>', None)
	
	if not output_name:
		return ('<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please provide an output filename</div>', None)
	
	if not output_name.endswith(".kpkg"):
		output_name += ".kpkg"
	
	try:
		from .obfuscate import create_distributable_package
		
		# Create package
		result = create_distributable_package(
			rag_dir=folder_path,
			output_path=output_name,
			password=password if encrypt and password else None,
			use_encryption=encrypt,
			include_source_names=True
		)
		
		msg = f'''
		<div style="
			background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
			border-radius: 12px;
			padding: 25px;
			color: white;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			animation: slideIn 0.3s ease-out;
		">
			<div style="font-size: 20px; font-weight: bold; margin-bottom: 15px;">
				✅ Package Created Successfully!
			</div>
			<div style="margin-bottom: 8px;"><strong>File:</strong> {output_name}</div>
			<div style="margin-bottom: 8px;"><strong>Size:</strong> {result.get('file_size_mb', 'Unknown')} MB</div>
			<div style="margin-bottom: 8px;"><strong>Chunks:</strong> {result.get('num_chunks', 'Unknown')}</div>
			<div style="margin-bottom: 15px;"><strong>Encrypted:</strong> {'Yes 🔒' if encrypt else 'No'}</div>
		'''
		
		if result.get('password'):
			msg += f'''
			<div style="
				background: rgba(255, 255, 255, 0.2);
				padding: 15px;
				border-radius: 8px;
				margin-top: 10px;
			">
				<div style="font-size: 14px; margin-bottom: 8px;">🔑 Generated Password:</div>
				<code style="font-size: 16px; word-break: break-all;">{result['password']}</code>
			</div>
			<div style="margin-top: 10px; font-size: 13px; opacity: 0.9;">
				⚠️ Save this password securely - it's required to use the package!
			</div>
			'''
		
		msg += '<div style="margin-top: 15px; font-size: 13px;">Package ready for download below</div></div>'
		
		# Return message and file path for download
		return (msg, output_name)
	except Exception as e:
		return (f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error creating package: {str(e)}</div>', None)


# ============================================================================
# Ollama Management Functions
# ============================================================================

def check_ollama_status(admin_token: str) -> tuple:
	"""Check if Ollama is installed and running. Returns (status_html, model_choices)."""
	if not verify_admin_token(admin_token):
		return ('<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>', [])
	
	try:
		# Check if ollama command exists
		result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
		if result.returncode != 0:
			return ('''
			<div style="text-align: center; padding: 40px;">
				<div style="font-size: 120px; margin-bottom: 20px;">❌</div>
				<div style="font-size: 24px; font-weight: bold; color: #f5576c;">Ollama Not Installed</div>
				<div style="font-size: 16px; color: #666; margin-top: 10px;">Install from <a href="https://ollama.com" target="_blank" style="color: #667eea;">ollama.com</a></div>
			</div>
			''', [])
		
		# Check if ollama is running
		result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
		if result.returncode == 0:
			# Parse model names from output
			lines = result.stdout.strip().split('\n')
			model_choices = []
			if len(lines) > 1:  # Skip header line
				for line in lines[1:]:
					parts = line.split()
					if parts:
						model_name = parts[0]
						model_choices.append(model_name)
			
			return ('''
			<div style="text-align: center; padding: 40px;">
				<div style="font-size: 120px; margin-bottom: 20px;">✅</div>
				<div style="font-size: 24px; font-weight: bold; color: #11998e;">Ollama is Running</div>
			</div>
			''', model_choices)
		else:
			return ('''
			<div style="text-align: center; padding: 40px;">
				<div style="font-size: 120px; margin-bottom: 20px;">⚠️</div>
				<div style="font-size: 24px; font-weight: bold; color: #f5576c;">Ollama Not Running</div>
				<div style="font-size: 16px; color: #666; margin-top: 10px;">Start with: <code>ollama serve</code></div>
			</div>
			''', [])
	except subprocess.TimeoutExpired:
		return ('<div style="text-align: center; padding: 40px;"><div style="font-size: 120px; margin-bottom: 20px;">❌</div><div style="font-size: 24px; font-weight: bold; color: #f5576c;">Ollama Not Responding</div></div>', [])
	except Exception as e:
		return (f'<div style="text-align: center; padding: 40px;"><div style="font-size: 120px; margin-bottom: 20px;">❌</div><div style="font-size: 24px; font-weight: bold; color: #f5576c;">Error: {str(e)}</div></div>', [])


def pull_ollama_model(admin_token: str, model_name: str) -> str:
	"""Pull a model from Ollama."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not model_name:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please provide a model name</div>'
	
	try:
		result = subprocess.run(
			["ollama", "pull", model_name],
			capture_output=True,
			text=True,
			timeout=300  # 5 minute timeout
		)
		
		if result.returncode == 0:
			return f'''
			<div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">[+] Successfully Installed: {model_name}</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px; max-height: 200px;">{result.stdout}</pre>
				</div>
			</div>
			'''
		else:
			return f'''
			<div style="background: #f5576c; color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">[X] Failed to Install: {model_name}</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px; max-height: 200px;">{result.stderr}</pre>
				</div>
			</div>
			'''
	except subprocess.TimeoutExpired:
		return f'<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">[!] Model pull timed out (may still be downloading in background)</div>'
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error pulling model: {str(e)}</div>'


def list_ollama_models(admin_token: str) -> str:
	"""List all installed Ollama models in a neat format."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	try:
		result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
		
		if result.returncode == 0:
			lines = result.stdout.strip().split('\n')
			if len(lines) <= 1:
				return '<div style="text-align: center; padding: 20px; color: #888;">No models installed</div>'
			
			# Parse the models
			models_html = '<div style="display: grid; gap: 15px;">'
			
			for line in lines[1:]:  # Skip header
				parts = line.split()
				if not parts:
					continue
				
				model_name = parts[0]
				model_id = parts[1] if len(parts) > 1 else ""
				size = parts[2] if len(parts) > 2 else ""
				modified = " ".join(parts[3:]) if len(parts) > 3 else ""
				
				models_html += f'''
				<div style="
					background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
					border-radius: 12px;
					padding: 20px;
					color: white;
					box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
					transition: transform 0.2s;
				" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
					<div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
						{model_name}
					</div>
					<div style="display: grid; gap: 5px; font-size: 14px; opacity: 0.9;">
						<div><strong>ID:</strong> {model_id}</div>
						<div><strong>Size:</strong> {size}</div>
						<div><strong>Modified:</strong> {modified}</div>
					</div>
				</div>
				'''
			
			models_html += '</div>'
			return models_html
		else:
			return f'''
			<div style="background: #f5576c; color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
				<div style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">[X] Failed to List Models</div>
				<div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
					<pre style="margin: 0; overflow-x: auto; font-size: 12px;">{result.stderr}</pre>
				</div>
			</div>
			'''
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error listing models: {str(e)}</div>'


# ============================================================================
# Default Model Configuration Functions
# ============================================================================

def get_current_default_model(admin_token: str) -> str:
	"""Get the current default model from config."""
	if not verify_admin_token(admin_token):
		return "granite3.3:2b"  # Fallback
	
	from .config import get_default_model
	return get_default_model()


def set_default_model_config(admin_token: str, model: str) -> str:
	"""Set the default model for user queries."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not model:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please select a model</div>'
	
	try:
		from .config import set_default_model
		set_default_model(model)
		
		return f'''
		<div style="
			background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
			border-radius: 12px;
			padding: 20px;
			color: white;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			animation: slideIn 0.3s ease-out;
			text-align: center;
		">
			<div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
				✅ Active Model Updated!
			</div>
			<div style="font-size: 14px;">
				New active model: <strong>{model}</strong>
			</div>
			<div style="font-size: 12px; margin-top: 10px; opacity: 0.9;">
				All users will now use this model for their queries.
			</div>
		</div>
		'''
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error setting default model: {str(e)}</div>'


def get_current_system_prompt(admin_token: str) -> str:
	"""Get the current system prompt name from config."""
	if not verify_admin_token(admin_token):
		return "default"  # Fallback
	
	from .config import get_system_prompt_name
	return get_system_prompt_name()


def set_system_prompt_config(admin_token: str, prompt_name: str) -> str:
	"""Set the system prompt for AI responses."""
	if not verify_admin_token(admin_token):
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Invalid admin token</div>'
	
	if not prompt_name:
		return '<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px; text-align: center;">[X] Please select a prompt</div>'
	
	try:
		from .config import set_system_prompt, get_system_prompt_text
		set_system_prompt(prompt_name)
		
		prompt_text = get_system_prompt_text(prompt_name)
		preview = prompt_text[:150] + "..." if len(prompt_text) > 150 else prompt_text
		
		return f'''
		<div style="
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			border-radius: 12px;
			padding: 20px;
			color: white;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			animation: slideIn 0.3s ease-out;
			text-align: center;
		">
			<div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
				✅ System Prompt Updated!
			</div>
			<div style="font-size: 14px; margin-bottom: 10px;">
				New prompt style: <strong>{prompt_name.title()}</strong>
			</div>
			<div style="
				background: rgba(255, 255, 255, 0.2);
				padding: 12px;
				border-radius: 8px;
				font-size: 12px;
				margin-top: 10px;
				text-align: left;
				font-family: monospace;
			">
				{preview}
			</div>
			<div style="font-size: 12px; margin-top: 10px; opacity: 0.9;">
				All users will now receive responses in this style.
			</div>
		</div>
		'''
	except Exception as e:
		return f'<div style="background: #f5576c; color: white; padding: 15px; border-radius: 8px;">[X] Error setting system prompt: {str(e)}</div>'


# ============================================================================
# Admin UI Builder
# ============================================================================

def build_admin_interface() -> gr.Blocks:
	"""Build the admin web interface."""
	
	# Custom CSS for modern, responsive design
	custom_css = """
	<style>
		/* Global responsive styles */
		@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
		
		* {
			font-family: 'Inter', sans-serif;
		}
		
		/* Smooth animations */
		@keyframes fadeIn {
			from { opacity: 0; transform: translateY(10px); }
			to { opacity: 1; transform: translateY(0); }
		}
		
		@keyframes slideIn {
			from { opacity: 0; transform: translateX(-10px); }
			to { opacity: 1; transform: translateX(0); }
		}
		
		@keyframes pulse {
			0%, 100% { transform: scale(1); }
			50% { transform: scale(1.05); }
		}
		
		@keyframes modalFadeIn {
			from { opacity: 0; transform: scale(0.95); }
			to { opacity: 1; transform: scale(1); }
		}
		
		/* Container responsiveness */
		.gradio-container {
			max-width: 1400px !important;
			margin: 0 auto !important;
			padding: 20px !important;
			background: #ffffff !important;
			min-height: 100vh;
		}
		
		/* Modal overlay styling */
		.gr-group {
			position: relative;
		}
		
		/* Make modal groups look like actual modals */
		.gr-group[style*="visible"] {
			animation: modalFadeIn 0.3s ease-out;
			background: white !important;
			border: 2px solid #667eea !important;
			border-radius: 16px !important;
			padding: 25px !important;
			margin: 20px 0 !important;
			box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
			z-index: 1000;
			max-height: 80vh;
			overflow-y: auto;
		}
		
		/* Card styles */
		.gr-box {
			border-radius: 16px !important;
			border: none !important;
			box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1) !important;
			animation: fadeIn 0.5s ease-out;
			background: white !important;
		}
		
		/* Tab styling */
		.gr-button-primary {
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
			border: none !important;
			border-radius: 8px !important;
			transition: all 0.3s ease !important;
			font-weight: 600 !important;
			width: 100% !important;
		}
		
		.gr-button-primary:hover {
			transform: translateY(-2px);
			box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
		}
		
		.gr-button-secondary {
			background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
			border: none !important;
			border-radius: 8px !important;
			transition: all 0.3s ease !important;
			font-weight: 600 !important;
			color: white !important;
			width: 100% !important;
		}
		
		.gr-button-secondary:hover {
			transform: translateY(-2px);
			box-shadow: 0 4px 12px rgba(240, 147, 251, 0.4) !important;
		}
		
		.gr-button-stop {
			background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%) !important;
			border: none !important;
			border-radius: 8px !important;
			transition: all 0.3s ease !important;
			font-weight: 600 !important;
			color: white !important;
			width: 100% !important;
		}
		
		.gr-button-stop:hover {
			transform: translateY(-2px);
			box-shadow: 0 4px 12px rgba(245, 87, 108, 0.4) !important;
		}
		
		/* Input fields */
		.gr-input, .gr-textbox {
			border-radius: 8px !important;
			border: 2px solid #e0e0e0 !important;
			transition: all 0.3s ease !important;
		}
		
		.gr-input:focus, .gr-textbox:focus {
			border-color: #667eea !important;
			box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
		}
		
		/* Logo styling */
		.kora-logo {
			max-width: 200px;
			width: 100%;
			height: auto;
			display: block;
			margin: 0 auto 15px auto;
		}
		
		/* Responsive layout */
		@media (max-width: 1024px) {
			.gradio-container {
				max-width: 100% !important;
				padding: 15px !important;
			}
		}
		
		@media (max-width: 768px) {
			.gradio-container {
				padding: 10px !important;
			}
			
			.gr-box {
				margin: 5px 0 !important;
			}
			
			.kora-logo {
				max-width: 150px;
			}
		}
		
		/* Header styling */
		.header-container {
			text-align: center;
			padding: 30px 20px;
			background: white;
			border-radius: 16px;
			margin-bottom: 20px;
			box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
			animation: slideIn 0.5s ease-out;
		}
		
		.header-title {
			font-size: clamp(20px, 4vw, 32px);
			font-weight: 700;
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			-webkit-background-clip: text;
			-webkit-text-fill-color: transparent;
			margin: 10px 0;
		}
		
		.header-subtitle {
			font-size: clamp(13px, 2.5vw, 16px);
			color: #666;
			margin-top: 8px;
		}
		
		/* Success/Error messages */
		.success-message {
			animation: slideIn 0.3s ease-out;
		}
		
		/* Loading animation */
		.loading {
			animation: pulse 1.5s ease-in-out infinite;
		}
	</style>
	"""
	
	with gr.Blocks(title="KORA Admin Panel", css=custom_css) as admin:
		gr.HTML(custom_css)
		
		# Header with logo
		logo_path = Path(__file__).parent.parent / ".github" / "media" / "KORA_Logo.png"
		
		with gr.Row():
			with gr.Column():
				if logo_path.exists():
					gr.Image(
						value=str(logo_path),
						show_label=False,
						show_download_button=False,
						container=False,
						height=120,
						width=200,
						show_share_button=False,
						show_fullscreen_button=False,
						elem_classes="kora-logo"
					)
				gr.HTML("""
				<div style="text-align: center; padding: 20px;">
					<h1 class="header-title">KORA Admin Panel</h1>
					<p class="header-subtitle">Manage API keys, packages, and AI settings</p>
				</div>
				""")
		
		# Admin token input
		with gr.Group():
			gr.Markdown("### Admin Authentication")
			admin_token_input = gr.Textbox(
				label="Admin Token",
				placeholder="Enter your admin token (check terminal output)",
				type="password",
				container=True
			)
			verify_token_btn = gr.Button("Unlock Panel", variant="primary", size="lg")
			token_status = gr.HTML()
		
		# Main admin interface (initially hidden)
		with gr.Group(visible=False) as admin_interface:
			with gr.Tabs():
				# ===== KORA Packages Tab =====
				with gr.Tab("📦 KORA Packages"):
					with gr.Row():
						with gr.Column(scale=1):
							gr.Markdown("### 📋 Package List")
							list_pkg_btn = gr.Button("Refresh Packages", variant="secondary", size="lg")
							pkg_list_output = gr.HTML()
							
							gr.Markdown("### Package Actions")
							with gr.Row():
								selected_pkg_path = gr.Textbox(
									label="Package Path",
									placeholder="Paste .kpkg file path here",
									interactive=True
								)
							with gr.Row():
								confirm_path_btn = gr.Button("✓ Confirm Selection", variant="primary")
							
							# Confirmation display
							path_confirmation = gr.HTML()
							
							with gr.Row():
								pkg_info_btn = gr.Button("Get Info", variant="secondary", interactive=False)
								pkg_test_btn = gr.Button("Test Package", variant="secondary", interactive=False)
							
							# Loading indicator
							info_loading = gr.HTML(visible=False)
							test_loading = gr.HTML(visible=False)
							
							# Modal-like container for results
							with gr.Group(visible=False) as info_modal:
								gr.Markdown("### Package Information")
								info_output = gr.HTML()
								info_close_btn = gr.Button("[X] Close", variant="stop")
							
							with gr.Group(visible=False) as test_modal:
								gr.Markdown("### Package Test Results")
								test_output = gr.HTML()
								test_close_btn = gr.Button("[X] Close", variant="stop")
						
						with gr.Column(scale=1):
							gr.Markdown("### Create Package from RAG/")
							pkg_output_name = gr.Textbox(
								label="Package Name",
								placeholder="my_package.kpkg",
								info="File will be created in project directory"
							)
							pkg_encrypt = gr.Checkbox(label="🔒 Encrypt Package", value=False)
							pkg_password = gr.Textbox(
								label="Password (optional)",
								placeholder="Leave empty for auto-generation",
								type="password",
								visible=False
							)
							create_pkg_btn = gr.Button("Create Package", variant="primary", size="lg")
							create_pkg_output = gr.HTML()
					
					gr.Markdown("---")
					
					# Create package from custom folder
					with gr.Row():
						with gr.Column():
							gr.Markdown("### Create Package from Custom Folder")
							
							folder_path_input = gr.Textbox(
								label="Folder Path",
								placeholder="Select folder to package",
								interactive=False
							)
							folder_select_btn = gr.Button("Select Folder", variant="secondary")
							
							folder_pkg_name = gr.Textbox(
								label="Package Name",
								placeholder="custom_package.kpkg"
							)
							folder_encrypt = gr.Checkbox(label="🔒 Encrypt Package", value=False)
							folder_password = gr.Textbox(
								label="Password (optional)",
								placeholder="Leave empty for auto-generation",
								type="password",
								visible=False
							)
							create_folder_pkg_btn = gr.Button("Create & Download", variant="primary", size="lg")
							folder_pkg_output = gr.HTML()
							folder_pkg_download = gr.File(label="Download Package", visible=False)
					
					# Event handlers
					list_pkg_btn.click(
						list_packages,
						inputs=[admin_token_input],
						outputs=[pkg_list_output]
					)
					
					# Folder selection for custom package
					def select_folder():
						import tkinter as tk
						from tkinter import filedialog
						root = tk.Tk()
						root.withdraw()
						root.attributes('-topmost', True)
						folder_selected = filedialog.askdirectory()
						root.destroy()
						if folder_selected:
							return folder_selected
						return ""
					
					folder_select_btn.click(
						select_folder,
						inputs=None,
						outputs=[folder_path_input]
					)
					
					# Path confirmation function
					def confirm_package_path(path):
						"""Confirm the selected package path and enable action buttons."""
						path = path.strip()
						if not path:
							return (
								"<div style='padding: 10px; background: #fee; color: #c00; border-radius: 5px;'>[!] Please enter a package path</div>",
								gr.update(interactive=False),
								gr.update(interactive=False)
							)
						
						import os
						if not os.path.exists(path):
							return (
								f"<div style='padding: 10px; background: #fee; color: #c00; border-radius: 5px;'>[!] Path not found: {path}</div>",
								gr.update(interactive=False),
								gr.update(interactive=False)
							)
						
						if not path.endswith('.kpkg'):
							return (
								f"<div style='padding: 10px; background: #ffc; color: #880; border-radius: 5px;'>[!] Warning: Path doesn't end with .kpkg</div>",
								gr.update(interactive=True),
								gr.update(interactive=True)
							)
						
						return (
							f"<div style='padding: 10px; background: #efe; color: #080; border-radius: 5px;'>[+] Path confirmed: {path}</div>",
							gr.update(interactive=True),
							gr.update(interactive=True)
						)
					
					confirm_path_btn.click(
						confirm_package_path,
						inputs=[selected_pkg_path],
						outputs=[path_confirmation, pkg_info_btn, pkg_test_btn]
					)
					
					# Show loading spinner for info
					def show_info_loading():
						return (
							gr.update(visible=True, value='''
								<div style="text-align: center; padding: 30px;">
									<div style="display: inline-block; width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite;"></div>
									<p style="margin-top: 15px; color: #667eea; font-weight: 600;">Loading package information...</p>
								</div>
								<style>
									@keyframes spin {
										0% { transform: rotate(0deg); }
										100% { transform: rotate(360deg); }
									}
								</style>
							'''),
							gr.update(visible=False),
							gr.update(visible=False),
							gr.update(visible=False)
						)
					
					# Info button with modal display
					def show_info_modal(admin_token, pkg_path):
						if not pkg_path:
							return gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=False), ""
						result = get_package_info(admin_token, pkg_path)
						return gr.update(visible=False), gr.update(visible=True), result, gr.update(visible=False), ""
					
					pkg_info_btn.click(
						show_info_loading,
						inputs=None,
						outputs=[info_loading, info_modal, test_modal, test_loading]
					).then(
						show_info_modal,
						inputs=[admin_token_input, selected_pkg_path],
						outputs=[info_loading, info_modal, info_output, test_modal]
					)
					
					info_close_btn.click(
						lambda: (gr.update(visible=False), ""),
						inputs=None,
						outputs=[info_modal, info_output]
					)
					
					# Show loading spinner for test
					def show_test_loading():
						return (
							gr.update(visible=True, value='''
								<div style="text-align: center; padding: 30px;">
									<div style="display: inline-block; width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #11998e; border-radius: 50%; animation: spin 1s linear infinite;"></div>
									<p style="margin-top: 15px; color: #11998e; font-weight: 600;">Running package tests...</p>
								</div>
								<style>
									@keyframes spin {
										0% { transform: rotate(0deg); }
										100% { transform: rotate(360deg); }
									}
								</style>
							'''),
							gr.update(visible=False),
							gr.update(visible=False),
							gr.update(visible=False)
						)
					
					# Test button with modal display
					def show_test_modal(admin_token, pkg_path):
						if not pkg_path:
							return gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=False)
						result = test_package(admin_token, pkg_path)
						return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), result
					
					pkg_test_btn.click(
						show_test_loading,
						inputs=None,
						outputs=[test_loading, info_modal, test_modal, info_loading]
					).then(
						show_test_modal,
						inputs=[admin_token_input, selected_pkg_path],
						outputs=[test_loading, info_modal, test_modal, test_output]
					)
					
					test_close_btn.click(
						lambda: (gr.update(visible=False), ""),
						inputs=None,
						outputs=[test_modal, test_output]
					)
					
					# Toggle password visibility
					pkg_encrypt.change(
						lambda x: gr.update(visible=x),
						inputs=[pkg_encrypt],
						outputs=[pkg_password]
					)
					
					folder_encrypt.change(
						lambda x: gr.update(visible=x),
						inputs=[folder_encrypt],
						outputs=[folder_password]
					)
					
					create_pkg_btn.click(
						create_package,
						inputs=[admin_token_input, pkg_output_name, pkg_encrypt, pkg_password],
						outputs=[create_pkg_output]
					)
					
					def create_and_show_download(admin_token, folder_path, output_name, encrypt, password):
						msg, file_path = create_package_from_folder(admin_token, folder_path, output_name, encrypt, password)
						if file_path:
							return msg, gr.update(value=file_path, visible=True)
						return msg, gr.update(visible=False)
					
					create_folder_pkg_btn.click(
						create_and_show_download,
						inputs=[admin_token_input, folder_path_input, folder_pkg_name, folder_encrypt, folder_password],
						outputs=[folder_pkg_output, folder_pkg_download]
					)
				# ===== API Keys Tab =====
				with gr.Tab("🔐 API Keys"):
					with gr.Row():
						with gr.Column(scale=1):
							gr.Markdown("### Existing API Keys")
							list_keys_btn = gr.Button("Refresh Keys", variant="secondary", size="lg")
							keys_output = gr.HTML()
						
						with gr.Column(scale=1):
							gr.Markdown("### Generate New Key")
							new_username = gr.Textbox(
								label="Username",
								placeholder="student_name",
								info="Enter a unique username for this API key"
							)
							create_key_btn = gr.Button("Generate API Key", variant="primary", size="lg")
							create_output = gr.HTML()
					
					with gr.Row():
						with gr.Column():
							gr.Markdown("### Revoke API Key")
							revoke_key_input = gr.Textbox(
								label="API Key to Revoke",
								placeholder="Paste full API key here",
								type="password"
							)
							revoke_btn = gr.Button("❌ Revoke Key", variant="stop", size="lg")
							revoke_output = gr.HTML()
					
					# Event handlers
					list_keys_btn.click(
						list_api_keys,
						inputs=[admin_token_input],
						outputs=[keys_output]
					)
					create_key_btn.click(
						create_api_key,
						inputs=[admin_token_input, new_username],
						outputs=[create_output]
					)
					revoke_btn.click(
						delete_api_key,
						inputs=[admin_token_input, revoke_key_input],
						outputs=[revoke_output]
					)
				
				# ===== AI Settings Tab =====
				with gr.Tab("🤖 AI Settings"):
					gr.Markdown("""
					### Default Model Configuration
					Set the model that all users will use for RAG queries. 
					This model applies globally to all users and cannot be changed by individual users.
					""")
					
					with gr.Group():
						gr.Markdown("### Current Active Model")
						current_model_display = gr.Textbox(
							label="Current Model",
							interactive=False,
							value=get_current_default_model(get_admin_token())
						)
						
						default_model_selector = gr.Dropdown(
							label="Select New Model",
							choices=[],
							value=None,
							info="Check Ollama status first to populate available models",
							interactive=True
						)
						
						set_default_btn = gr.Button("Set as Active Model", variant="primary", size="lg")
						set_default_output = gr.HTML()
					
					gr.Markdown("---")
					
					gr.Markdown("""
					### System Prompt Configuration
					Choose the AI's response style. This affects how the assistant formulates answers to user questions.
					""")
					
					with gr.Group():
						from .config import get_available_prompts
						
						prompt_descriptions = get_available_prompts()
						prompt_choices = [(f"{name.title()} - {desc}", name) for name, desc in prompt_descriptions.items()]
						
						gr.Markdown("### Current Prompt Style")
						current_prompt_display = gr.Textbox(
							label="Current Prompt",
							interactive=False,
							value=get_current_system_prompt(get_admin_token()).title()
						)
						
						system_prompt_selector = gr.Dropdown(
							label="Select Prompt Style",
							choices=prompt_choices,
							value=get_current_system_prompt(get_admin_token()),
							info="Choose how the AI should respond to questions",
							interactive=True
						)
						
						set_prompt_btn = gr.Button("Set Prompt Style", variant="primary", size="lg")
						set_prompt_output = gr.HTML()
					
					gr.Markdown("---")
					
					with gr.Row():
						with gr.Column(scale=1):
							gr.Markdown("### Ollama Status")
							status_btn = gr.Button("Check Status", variant="secondary", size="lg")
							status_output = gr.HTML()
						
						with gr.Column(scale=1):
							gr.Markdown("### Installed Models")
							list_models_btn = gr.Button("List Models", variant="secondary", size="lg")
							models_output = gr.HTML()
					
					gr.Markdown("---")
					
					gr.Markdown("""
					### 📦 Model Installation
					[Install Ollama models to make them available for use.](https://ollama.com/download) After installation, 
					you can set them as the default using the dropdown above.
					""")
					
					with gr.Row():
						with gr.Column():
							gr.Markdown("### Quick Model Install")
							
							with gr.Row():
								pull_granite_btn = gr.Button("Install granite3.3:2b", variant="primary")
								pull_qwen_btn = gr.Button("Install qwen2.5:3b", variant="primary")
								pull_phi_btn = gr.Button("Install phi3:3.8b", variant="primary")
							
							with gr.Row():
								pull_llama_btn = gr.Button("Install llama3.2:3b", variant="primary")
								pull_gemma_btn = gr.Button("Install gemma2:2b", variant="primary")
								pull_mistral_btn = gr.Button("Install mistral:7b", variant="primary")
							
							with gr.Row():
								pull_tinyllama_btn = gr.Button("Install tinyllama:1.1b", variant="primary")
								pull_phi3_mini_btn = gr.Button("Install phi3:mini", variant="primary")
								pull_gemma2_9b_btn = gr.Button("Install gemma2:9b", variant="primary")
							
							with gr.Row():
								pull_llama3_8b_btn = gr.Button("Install llama3.1:8b", variant="primary")
								pull_mistral_nemo_btn = gr.Button("Install mistral-nemo:latest", variant="primary")
								pull_phi3_medium_btn = gr.Button("Install phi3:medium", variant="primary")
						
						with gr.Column():
							gr.Markdown("### Custom Model Install")
							custom_model_input = gr.Textbox(
								label="Model Name",
								placeholder="model_name:version",
								info="Enter any Ollama model name from the registry"
							)
							pull_custom_btn = gr.Button("Install Model", variant="primary", size="lg")
					
					pull_output = gr.HTML()
					
					# Event handlers
					def check_status_and_populate(token):
						"""Check Ollama status and populate dropdown with available models."""
						status_html, model_choices = check_ollama_status(token)
						
						# Update dropdown with available models
						if model_choices:
							current_model = get_current_default_model(token)
							# Set current model as selected if it's in the list, otherwise first model
							selected = current_model if current_model in model_choices else (model_choices[0] if model_choices else None)
							return status_html, gr.update(choices=model_choices, value=selected)
						else:
							return status_html, gr.update(choices=[], value=None)
					
					status_btn.click(
						check_status_and_populate,
						inputs=[admin_token_input],
						outputs=[status_output, default_model_selector]
					)
					
					list_models_btn.click(
						list_ollama_models,
						inputs=[admin_token_input],
						outputs=[models_output]
					)
					
					# Set default model handler
					def update_default_model(token, model):
						result = set_default_model_config(token, model)
						new_current = get_current_default_model(token)
						return result, new_current
					
					set_default_btn.click(
						update_default_model,
						inputs=[admin_token_input, default_model_selector],
						outputs=[set_default_output, current_model_display]
					)
					
					# Set system prompt handler
					def update_system_prompt(token, prompt_name):
						result = set_system_prompt_config(token, prompt_name)
						new_current = get_current_system_prompt(token).title()
						return result, new_current
					
					set_prompt_btn.click(
						update_system_prompt,
						inputs=[admin_token_input, system_prompt_selector],
						outputs=[set_prompt_output, current_prompt_display]
					)
					
					# Quick install button handlers (wrapper functions to avoid lambda issues)
					def pull_granite(token):
						return pull_ollama_model(token, "granite3.3:2b")
					
					def pull_qwen(token):
						return pull_ollama_model(token, "qwen2.5:3b")
					
					def pull_phi(token):
						return pull_ollama_model(token, "phi3:3.8b")
					
					def pull_llama(token):
						return pull_ollama_model(token, "llama3.2:3b")
					
					def pull_gemma(token):
						return pull_ollama_model(token, "gemma2:2b")
					
					def pull_mistral(token):
						return pull_ollama_model(token, "mistral:7b")
					
					def pull_tinyllama(token):
						return pull_ollama_model(token, "tinyllama:1.1b")
					
					def pull_phi3_mini(token):
						return pull_ollama_model(token, "phi3:mini")
					
					def pull_gemma2_9b(token):
						return pull_ollama_model(token, "gemma2:9b")
					
					def pull_llama3_8b(token):
						return pull_ollama_model(token, "llama3.1:8b")
					
					def pull_mistral_nemo(token):
						return pull_ollama_model(token, "mistral-nemo:latest")
					
					def pull_phi3_medium(token):
						return pull_ollama_model(token, "phi3:medium")
					
					# Quick install buttons
					pull_granite_btn.click(
						pull_granite,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_qwen_btn.click(
						pull_qwen,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_phi_btn.click(
						pull_phi,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_llama_btn.click(
						pull_llama,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_gemma_btn.click(
						pull_gemma,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_mistral_btn.click(
						pull_mistral,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_tinyllama_btn.click(
						pull_tinyllama,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_phi3_mini_btn.click(
						pull_phi3_mini,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_gemma2_9b_btn.click(
						pull_gemma2_9b,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_llama3_8b_btn.click(
						pull_llama3_8b,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_mistral_nemo_btn.click(
						pull_mistral_nemo,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_phi3_medium_btn.click(
						pull_phi3_medium,
						inputs=[admin_token_input],
						outputs=[pull_output]
					)
					pull_custom_btn.click(
						pull_ollama_model,
						inputs=[admin_token_input, custom_model_input],
						outputs=[pull_output]
					)
		
		# Token verification handler
		def verify_token(token: str):
			"""Verify admin token and show/hide interface."""
			if verify_admin_token(token):
				return (
					'<div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: 600; animation: slideIn 0.3s ease-out;">✅ Access Granted!</div>',
					gr.update(visible=True)
				)
			else:
				return (
					'<div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: 600; animation: slideIn 0.3s ease-out;">[X] Invalid admin token. Please check the terminal output.</div>',
					gr.update(visible=False)
				)
		
		verify_token_btn.click(
			verify_token,
			inputs=[admin_token_input],
			outputs=[token_status, admin_interface]
		)
	
	return admin


def main() -> None:
	"""Launch the admin interface."""
	import logging
	
	# Suppress HTTP logs
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)
	
	# Generate and print admin token
	token = get_admin_token()
	
	print("\n" + "=" * 70)
	print("  🔐 KORA Admin Panel - Web Admin Interface")
	print("=" * 70)
	print(f"\n  🔑 Admin Token: {token}")
	print(f"\n  ➜  Local:   http://127.0.0.1:7861")
	print(f"  ➜  Network: http://{_get_local_ip()}:7861")
	print("\n  ⚠️  Keep your admin token secure!")
	print("=" * 70 + "\n")
	
	# Build and launch
	admin = build_admin_interface()
	admin.launch(
		server_name="0.0.0.0",
		server_port=7861,
		share=False,
		quiet=True,
		show_api=False
	)


def _get_local_ip() -> str:
	"""Get the local network IP address."""
	import socket
	try:
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(("8.8.8.8", 80))
		local_ip = s.getsockname()[0]
		s.close()
		return local_ip
	except Exception:
		return "0.0.0.0"


if __name__ == "__main__":
	main()
