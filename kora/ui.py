import gradio as gr
from typing import List, Dict, Any, Optional
import time

from .rag import answer_question, rebuild_index, build_or_load_index
from .auth import get_authenticator


HARD_CODED_MODEL = "granite3.3:2b"


def _chatbot_response(history: List[Dict[str, str]], message: str, top_k: int, temperature: float, session_token: str) -> List[Dict[str, str]]:
	"""Process chatbot response with authentication check."""
	auth = get_authenticator()
	
	# Validate session
	if not session_token or not auth.validate_session(session_token):
		history = history + [
			{"role": "user", "content": message},
			{"role": "assistant", "content": "❌ Authentication required. Please provide a valid API key."},
		]
		return history
	
	res = answer_question(query=message, top_k=top_k, model=HARD_CODED_MODEL, temperature=temperature)
	answer = res["answer"]
	
	# Get username for logging
	username = auth.get_user_from_session(session_token)
	if username:
		print(f"[KORA] Query from {username}: {message[:50]}...")
	
	# Do not print context sources in the UI
	history = history + [
		{"role": "user", "content": message},
		{"role": "assistant", "content": answer},
	]
	return history


def _authenticate_with_api_key(api_key: str) -> tuple[str, str, str]:
	"""Authenticate using API key and return status, message, and session token."""
	if not api_key or len(api_key) != 64:
		return "❌ Authentication Failed", "Please provide a valid 64-character API key.", ""
	
	auth = get_authenticator()
	
	if not auth.validate_api_key(api_key):
		return "❌ Authentication Failed", "Invalid or expired API key.", ""
	
	# Create session
	session_token = auth.create_session(api_key)
	
	if not session_token:
		return "❌ Authentication Failed", "Failed to create session.", ""
	
	# Get username for display
	api_keys = auth._load_api_keys()
	username = api_keys.get(api_key, {}).get("username", "Unknown")
	
	return "✅ Authentication Successful", f"Welcome back, {username}!", session_token


def build_interface() -> gr.Blocks:
	# Try to load existing index first, only build if necessary
	store, status = build_or_load_index(force_rebuild=False)
	startup_info = {"status": status, "num_chunks": len(store.metadatas)}
	
	# Format status message based on load type
	if status == "loaded_from_kpkg":
		status_text = "loaded from .kpkg package"
	elif status == "loaded_from_disk":
		status_text = "loaded from disk"
	else:
		status_text = "built"
	
	startup_msg = f"<span style='color: green;'>Index {status_text}. Chunks: {startup_info['num_chunks']}</span>"

	with gr.Blocks(title="KORA: Knowledge oriented retrieval assistant - BETA") as demo:
		# Display logo
		gr.Image(".github/media/KORA_Logo.png", show_label=False, show_download_button=False, container=False, height=150)
		
		gr.Markdown("""
		**KORA: Knowledge oriented retrieval assistant - BETA**

		Designed by researchers at [Georgia Tech](https://gatech.edu).

		Uses Docling + FAISS to retrieve content files, queries Ollama on port `granite3.3:2b`.
		
		**Authentication Required**: Use `kora-auth` CLI tool to generate an API key, then provide it here to access the system.
		""")
		
		# Session state
		session_token = gr.State("")
		
		# Authentication section
		with gr.Group():
			gr.Markdown("### Authentication")
			gr.Markdown("**Note:** Generate API keys using the `kora-auth` command-line tool.")
			
			api_key_input = gr.Textbox(label="API Key", placeholder="Your 64-character API key", max_lines=1)
			api_login_btn = gr.Button("Login with API Key", variant="primary")
			
			auth_status = gr.Markdown("")
			auth_message = gr.Textbox(label="Authentication Message", visible=False, interactive=False)
		
		# Main interface (initially hidden)
		with gr.Group(visible=False) as main_interface:
			with gr.Row():
				topk = gr.Slider(label="Top-K", minimum=1, maximum=20, value=8, step=1)
				temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.1)
				rebuild_btn = gr.Button("Rebuild Index")
			
			chatbot = gr.Chatbot(height=500, type='messages', show_copy_button=True, latex_delimiters=[
				{"left": "$$", "right": "$$", "display": True},
				{"left": "$", "right": "$", "display": False}
			])
			msg = gr.Textbox(label="Your question")
			send = gr.Button("Send")
			
			status = gr.Markdown(startup_msg)
		
		# Authentication handlers
		def handle_api_login(api_key: str):
			auth_result, message, token = _authenticate_with_api_key(api_key)
			
			if token:
				return (
					auth_result,
					gr.update(value=message, visible=True),
					token,
					gr.update(visible=True),  # Show main interface
					""  # Clear API key
				)
			else:
				return (
					auth_result,
					gr.update(value=message, visible=True),
					"",
					gr.update(visible=False),  # Keep main interface hidden
					api_key  # Keep API key
				)
		
		api_login_btn.click(
			handle_api_login,
			inputs=[api_key_input],
			outputs=[auth_status, auth_message, session_token, main_interface, api_key_input]
		)
		
		# Chat handlers
		def on_send(history: List[Dict[str, str]], message: str, k: int, temp: float, token: str):
			if not message:
				return history, ""
			
			new_history = _chatbot_response(history, message, k, temp, token)
			return new_history, ""
		
		send.click(on_send, inputs=[chatbot, msg, topk, temperature, session_token], outputs=[chatbot, msg])
		msg.submit(on_send, inputs=[chatbot, msg, topk, temperature, session_token], outputs=[chatbot, msg])

		def on_rebuild():
			res = rebuild_index()
			return f"<span style='color: green;'>Index {res['status']}. Chunks: {res['num_chunks']}</span>"

		rebuild_btn.click(on_rebuild, outputs=status)

	return demo
