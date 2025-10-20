import os
import sys
import subprocess
import socket
import logging
from typing import Optional

import gradio as gr

from .rag import ensure_dirs
from .ui import build_interface


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


def _print_startup_banner(port: int) -> None:
	"""Print a clean startup banner with local and network URLs."""
	local_ip = _get_local_ip()
	
	print("\n" + "=" * 60)
	print("  KORA: Knowledge Oriented Retrieval Assistant")
	print("=" * 60)
	print(f"\n  ➜  Local:   http://127.0.0.1:{port}")
	print(f"  ➜  Network: http://{local_ip}:{port}")
	print("\n  📝 Authentication: Use kora-auth to generate API keys")
	print("  🔧 API Server:     Use kora-api for REST API access")
	print("  🤖 LLM Model:      granite3.3:2b via Ollama")
	print("\n" + "=" * 60 + "\n")


def main() -> None:
	# Suppress HTTP request logs from httpx/gradio for cleaner output
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)
	
	# Ensure environment and directories
	ensure_dirs()

	# Verify Ollama availability and model presence (best effort quick checks)
	if not _which("ollama"):
		print("[KORA] Ollama not found in PATH. Please install Ollama and ensure 'ollama' is available.")
		sys.exit(1)

	# Best-effort model presence check
	try:
		ls_proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
		if "granite3.3:2b" not in (ls_proc.stdout or ""):
			print("[KORA] Warning: Granite model 'granite3.3:2b' not found in ollama list.")
			print("[KORA] Please run: ollama pull granite3.3:2b")
	except Exception:
		pass

	# Launch UI with quiet mode to suppress Gradio's default output
	demo = build_interface()
	
	# Print our custom banner
	_print_startup_banner(7860)
	
	# Launch with quiet mode and 0.0.0.0 to allow network access
	demo.launch(
		server_name="0.0.0.0",
		server_port=7860,
		share=False,
		quiet=True,
		show_api=False
	)


if __name__ == "__main__":
	main()
