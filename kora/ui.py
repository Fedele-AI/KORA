import gradio as gr
from typing import List, Dict, Any

from .rag import answer_question, rebuild_index, build_or_load_index


HARD_CODED_MODEL = "granite3.3:2b"


def _chatbot_response(history: List[Dict[str, str]], message: str, top_k: int) -> List[Dict[str, str]]:
	res = answer_question(query=message, top_k=top_k, model=HARD_CODED_MODEL)
	answer = res["answer"]
	# Do not print context sources in the UI
	history = history + [
		{"role": "user", "content": message},
		{"role": "assistant", "content": answer},
	]
	return history


def build_interface() -> gr.Blocks:
	# Try to load existing index first, only build if necessary
	store, status = build_or_load_index(force_rebuild=False)
	startup_info = {"status": status, "num_chunks": len(store.metadatas)}
	status_text = "loaded from disk" if status == "loaded_from_disk" else "built"
	startup_msg = f"<span style='color: green;'>Index {status_text}. Chunks: {startup_info['num_chunks']}</span>"

	with gr.Blocks(title="KORA: Knowledge oriented reterival assistant - BETA") as demo:
		gr.Markdown("""
		**KORA: Knowledge oriented reterival assistant - BETA**
		
		Uses Docling + FAISS to retrieve from files in `RAG/` and queries Ollama `granite3.3:2b`.
		""")
		with gr.Row():
			topk = gr.Slider(label="topK", minimum=1, maximum=20, value=8, step=1)
			rebuild_btn = gr.Button("Rebuild Index")
		
		chatbot = gr.Chatbot(height=500, type='messages')
		msg = gr.Textbox(label="Your question")
		send = gr.Button("Send")

		def on_send(history: List[Dict[str, str]], message: str, k: int):
			if not message:
				return history
			return _chatbot_response(history, message, k)

		send.click(on_send, inputs=[chatbot, msg, topk], outputs=chatbot)
		msg.submit(on_send, inputs=[chatbot, msg, topk], outputs=chatbot)

		def on_rebuild():
			res = rebuild_index()
			return f"<span style='color: green;'>Index {res['status']}. Chunks: {res['num_chunks']}</span>"

		status = gr.Markdown(startup_msg)
		rebuild_btn.click(on_rebuild, outputs=status)

	return demo
