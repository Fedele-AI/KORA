import os
import sys
import subprocess
from typing import Optional

import gradio as gr

from .rag import ensure_dirs
from .ui import build_interface


def _which(cmd: str) -> Optional[str]:
	from shutil import which
	return which(cmd)


def main() -> None:
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
			print("[KORA] Granite model 'granite3.3:2b' not found in ollama list. It should already be installed as per your note.")
	except Exception:
		pass

	# Launch UI
	demo = build_interface()
	demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
	main()
