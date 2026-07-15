import os
import psutil
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Configuration ---
REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
CONTEXT_SIZE = 1024

def load_model():
    """Baixa (se necessário) e carrega o modelo GGUF na memória."""
    print(f"[*] Verificando cache ou baixando {FILENAME}...")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print("[*] Carregando modelo via llama.cpp (Edge CPU Mode)...")

        llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_threads=2,
            n_batch=128,
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"[!] Erro ao carregar o modelo: {e}")
        return None

llm = load_model()

def get_memory_usage():
    """Monitora o consumo de RAM (RSS) em tempo real."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    return f" **RAM Usage:** `{ram_mb:.2f} MB` |  **Model:** `Phi-3-Mini (Q4)` |  **Compute:** `CPU Edge`"

def generate_response(user_message, chat_history):
    """Gera a resposta com streaming nativo, usando o formato 'messages' do Gradio."""
    if not llm:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "Erro crítico: Modelo não foi carregado."})
        yield chat_history, get_memory_usage()
        return

    # Adiciona a mensagem do usuário e um placeholder vazio pro assistente
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    yield chat_history, get_memory_usage()

    # Prepara o prompt no formato ChatML exigido pelo Phi-3
    # (usa o histórico inteiro, exceto o placeholder vazio recém-adicionado)
    # Sanitiza: o Gradio pode devolver 'content' como lista/estrutura em vez de
    # string pura (ex: mensagens já renderizadas com markdown), e o llama-cpp-python
    # quebra se 'content' não for string. Forçamos a conversão aqui.
    messages = []
    for m in chat_history[:-1]:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        messages.append({"role": m["role"], "content": str(content)})

    try:
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.1,
            stream=True
        )

        response_text = ""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                response_text += delta["content"]
                chat_history[-1]["content"] = response_text
                yield chat_history, get_memory_usage()

    except Exception as e:
        chat_history[-1]["content"] = f"**[Erro na Inferência]:** {str(e)}"
        yield chat_history, get_memory_usage()

# --- Gradio UI (Full-Stack) ---
with gr.Blocks() as interface:
    gr.Markdown("# Edge AI Chatbot (100% Local & Private)")

    hardware_monitor = gr.Markdown(value=get_memory_usage())

    chatbot = gr.Chatbot(height=550)

    with gr.Row():
        msg_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here...",
            scale=9
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    submit_event = msg_input.submit(
        fn=generate_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, hardware_monitor]
    )
    submit_btn.click(
        fn=generate_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, hardware_monitor]
    )

    submit_event.then(lambda: "", None, [msg_input])
    submit_btn.click(lambda: "", None, [msg_input])

if __name__ == "__main__":
    if llm:
        print("\n Iniciando servidor Gradio local...")
        interface.launch(ssr_mode=False)
    else:
        print("Falha na inicialização. Verifique sua conexão para o download do modelo.")