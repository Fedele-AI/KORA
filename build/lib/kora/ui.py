import gradio as gr
from typing import List, Dict, Any

from .rag import answer_question, rebuild_index


def _chatbot_response(history: List[List[str]], message: str, top_k: int, model: str) -> List[List[str]]:
	res = answer_question(query=message, top_k=top_k, model=model)
	answer = res["answer"]
	context = res.get("context", [])
	cite_lines = []
	for item in context:
		cite_lines.append(f"- {item['source']} (score {item['score']:.3f})")
	cite_text = "\n".join(cite_lines)
	full_answer = answer
	if cite_text:
		full_answer += f"\n\nContext sources:\n{cite_text}"
	history = history + [[message, full_answer]]
	return history


def build_interface() -> gr.Blocks:
	with gr.Blocks(title="KORA - Knowledge Oriented Retrieval Assistant") as demo:
		gr.Markdown("""
		**KORA** uses Docling + FAISS to retrieve from files in `RAG/` and queries Ollama `granite3.3:2b`.
		""")
		with gr.Row():
			topk = gr.Slider(label="topK", minimum=1, maximum=10, value=4, step=1)
			model = gr.Textbox(label="Ollama model", value="granite3.3:2b")
			rebuild_btn = gr.Button("Rebuild Index")
		
		chatbot = gr.Chatbot(height=500)
		msg = gr.Textbox(label="Your question")
		send = gr.Button("Send")

		def on_send(history: List[List[str]], message: str, k: int, m: str):
			if not message:
				return history
			return _chatbot_response(history, message, k, m)

		send.click(on_send, inputs=[chatbot, msg, topk, model], outputs=chatbot)
		msg.submit(on_send, inputs=[chatbot, msg, topk, model], outputs=chatbot)

		def on_rebuild():
			res = rebuild_index()
			return f"Index {res['status']}. Chunks: {res['num_chunks']}"

		status = gr.Markdown("Ready.")
		rebuild_btn.click(on_rebuild, outputs=status)

	return demo
