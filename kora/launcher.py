import os
import sys
import subprocess
import socket
import logging
import threading
import argparse
from typing import Optional

import gradio as gr

from .rag import ensure_dirs
from .ui import build_interface
from .admin_ui import build_admin_interface, get_admin_token
from .config import get_default_model, set_default_model


def _which(cmd: str) -> Optional[str]:
	from shutil import which
	return which(cmd)


def _get_local_ip() -> str:
	"""Get the local network IP address."""
	try:
		# Create a socket to get the local IP
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# Connect to a public DNS server (doesn't actually send data)
		s.connect(("8.8.8.8", 80))
		local_ip = s.getsockname()[0]
		s.close()
		return local_ip
	except Exception:
		return "0.0.0.0"


def _print_startup_banner(port: int, admin_port: int, admin_token: str) -> None:
	"""Print a clean startup banner with local and network URLs."""
	local_ip = _get_local_ip()
	default_model = get_default_model()
	
	print("\n" + "=" * 60)
	print("  KORA: Knowledge Oriented Retrieval Assistant")
	print("=" * 60)
	print(f"\n  🌐 Main Interface:")
	print(f"    ➜  Local:   http://127.0.0.1:{port}")
	print(f"    ➜  Network: http://{local_ip}:{port}")
	print(f"\n  🔐 Admin Panel:")
	print(f"    ➜  Local:   http://127.0.0.1:{admin_port}")
	print(f"    ➜  Network: http://{local_ip}:{admin_port}")
	print(f"    🔑 Token:   {admin_token}")
	print("\n  📝 Authentication: Use kora-auth to generate API keys")
	print("  🔧 API Server:     Use kora-api for REST API access")
	print(f"  🤖 LLM Model:      {default_model} (configurable via Admin Panel)")
	print("\n" + "=" * 60 + "\n")


def main() -> None:
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description='Launch KORA: Knowledge Oriented Retrieval Assistant')
	parser.add_argument(
		'--model',
		type=str,
		help='Set the LLM model to use (e.g., granite3.3:2b, qwen2.5:3b). This overrides the admin panel setting.'
	)
	parser.add_argument(
		'--server-name',
		type=str,
		default='0.0.0.0',
		help='Server name to bind to (default: 0.0.0.0)'
	)
	parser.add_argument(
		'--server-port',
		type=int,
		default=7860,
		help='Main UI port (default: 7860)'
	)
	parser.add_argument(
		'--admin-port',
		type=int,
		default=7861,
		help='Admin UI port (default: 7861)'
	)
	args = parser.parse_args()
	
	# If model is specified via command line, set it
	if args.model:
		set_default_model(args.model)
		print(f"[KORA] Model set to: {args.model}")
	
	# Suppress HTTP request logs from httpx/gradio for cleaner output
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)
	
	# Ensure environment and directories
	ensure_dirs()

	# Verify Ollama availability (best effort quick check)
	# Skip if OLLAMA_HOST is set (containerized environment)
	ollama_host = os.getenv("OLLAMA_HOST")
	if not ollama_host:
		if not _which("ollama"):
			print("[KORA] Ollama not found in PATH. Please install Ollama and ensure 'ollama' is available.")
			sys.exit(1)

		# Best-effort check for any installed model
		try:
			ls_proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
			if not ls_proc.stdout or "NAME" not in ls_proc.stdout:
				print("[KORA] Warning: No Ollama models found.")
				print("[KORA] Please install a model. Recommended: ollama pull qwen2.5:3b")
		except Exception:
			pass
	else:
		print(f"[KORA] Using remote Ollama at: {ollama_host}")

	# Launch UI with quiet mode to suppress Gradio's default output
	demo = build_interface()
	admin_demo = build_admin_interface()
	
	# Get admin token
	admin_token = get_admin_token()
	
	# Print our custom banner
	_print_startup_banner(args.server_port, args.admin_port, admin_token)
	
	# Launch admin panel in background thread
	def launch_admin():
		admin_demo.launch(
			server_name=args.server_name,
			server_port=args.admin_port,
			share=False,
			quiet=True,
			show_api=False,
			prevent_thread_lock=True
		)
	
	admin_thread = threading.Thread(target=launch_admin, daemon=True)
	admin_thread.start()
	
	# Launch main UI (this will block)
	demo.launch(
		server_name=args.server_name,
		server_port=args.server_port,
		share=False,
		quiet=True,
		show_api=False
	)


if __name__ == "__main__":
	main()
