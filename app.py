import os
import time
import psutil
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Configuration ---
# Trocado de Qwen2.5-1.5B para Qwen2.5-0.5B-Instruct: metade do tamanho,
# priorizando velocidade máxima. A especialização de nicho (próxima etapa)
# deve compensar boa parte da perda de conhecimento geral de um modelo tão pequeno.
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
CONTEXT_SIZE = 1024
MAX_RESPONSE_TOKENS = 300  # reduzido de 500: corta a cauda longa de tempo total

# Prompt de especialização: nicho = dúvidas técnicas sobre o stack
# Gradio, Hugging Face Spaces, CrewAI, MCP e execução local com llama.cpp/GGUF.
# Inclui exemplos curtos (few-shot fixo no prompt, NÃO é RAG) para reforçar o
# escopo e o padrão de recusa educada para perguntas fora do nicho.
NICHE_SYSTEM_PROMPT = (
    "Você é um assistente técnico especializado APENAS em: Gradio, Hugging "
    "Face Spaces, CrewAI, Model Context Protocol (MCP) e execução de modelos "
    "locais com llama.cpp/GGUF. Responda SEMPRE em português do Brasil, de "
    "forma direta e objetiva.\n\n"
    "Se a pergunta não for sobre esses temas, recuse educadamente e explique "
    "que só pode ajudar com esses tópicos. Exemplo:\n"
    "Pergunta: 'Qual a capital da França?'\n"
    "Resposta: 'Isso foge do meu escopo - eu ajudo só com dúvidas sobre "
    "Gradio, Hugging Face Spaces, CrewAI, MCP e llama.cpp. Posso ajudar com "
    "algo nessas áreas?'"
)

def load_model():
    """Baixa (se necessário) e carrega o modelo GGUF na memória."""
    print(f"[*] Verificando cache ou baixando {FILENAME}...")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print("[*] Carregando modelo via llama.cpp (Edge CPU Mode)...")

        llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_threads=2,         # Codespace tem só 2 cores (nproc=2) - manter em 2
            n_threads_batch=2,   # garante que o prefill do prompt também usa os 2 cores
            n_batch=128,         # 256 piorou o throughput (overhead sem ganho com só 2 threads)
            use_mlock=True,      # trava o modelo em RAM, evita swap e picos de latência
            flash_attn=True,     # reduz movimentação de dados durante a atenção (1.3x-2x em prompts longos)
            verbose=False
        )

        # Warm-up: primeira inferência costuma ser bem mais lenta (cache/mmap
        # "frios"). Roda uma geração descartável de 1 token aqui, na
        # inicialização, para não distorcer o TTFT da primeira pergunta real.
        print("[*] Aquecendo o modelo (warm-up)...")
        list(llm.create_chat_completion(
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=1,
            stream=False,
        ))
        print("[*] Warm-up concluído.")

        return llm
    except Exception as e:
        print(f"[!] Erro ao carregar o modelo: {e}")
        return None

llm = load_model()

def get_memory_usage(ttft=None, total_time=None, tokens=None):
    """Monitora o consumo de RAM (RSS) e, opcionalmente, métricas de tempo/velocidade."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    base = f" **RAM Usage:** `{ram_mb:.2f} MB` |  **Model:** `Qwen2.5-0.5B (Q4)` |  **Compute:** `CPU Edge`"

    if ttft is not None:
        base += f" |  **1º token:** `{ttft:.2f}s`"
    if total_time is not None:
        base += f" |  **Tempo total:** `{total_time:.2f}s`"
    if tokens is not None and total_time and total_time > 0:
        base += f" |  **Vel.:** `{tokens/total_time:.1f} tok/s`"

    return base

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

    # Prepara o prompt no formato de chat exigido pelo modelo
    # (usa o histórico inteiro, exceto o placeholder vazio recém-adicionado)
    # Sanitiza: o Gradio pode devolver 'content' como lista/estrutura em vez de
    # string pura (ex: mensagens já renderizadas com markdown), e o llama-cpp-python
    # quebra se 'content' não for string. Forçamos a conversão aqui.
    # Alguns chat templates de GGUF não tratam bem a role "system" (ex: o Phi-3
    # usado antes a ignorava). Por segurança, mantemos o workaround de embutir
    # a instrução em mensagens do usuário em vez de usar role="system".
    # Reforçamos o prompt de nicho em TODA mensagem do usuário (não só na
    # primeira): um modelo tão pequeno tende a "esquecer" a instrução ao
    # longo da conversa, e o texto mais recente pesa mais na geração.
    messages = []
    for m in chat_history[:-1]:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        content = str(content)
        if m["role"] == "user":
            content = f"{NICHE_SYSTEM_PROMPT}\n\nPergunta do usuário: {content}"
        messages.append({"role": m["role"], "content": content})

    try:
        start_time = time.time()
        first_token_time = None
        token_count = 0

        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.1,
            stream=True
        )

        response_text = ""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                if first_token_time is None:
                    first_token_time = time.time() - start_time

                response_text += delta["content"]
                token_count += 1
                chat_history[-1]["content"] = response_text

                elapsed = time.time() - start_time
                yield chat_history, get_memory_usage(ttft=first_token_time, total_time=elapsed, tokens=token_count)

        # Marca final com o tempo total definitivo
        total_time = time.time() - start_time
        yield chat_history, get_memory_usage(ttft=first_token_time, total_time=total_time, tokens=token_count)

    except Exception as e:
        chat_history[-1]["content"] = f"**[Erro na Inferência]:** {str(e)}"
        yield chat_history, get_memory_usage()

# --- Efeito visual: fade suave no campo de texto ao enviar ---
FADE_CSS = """
#msg_input textarea {
    transition: opacity 0.25s ease;
}
"""

FADE_JS = """
() => {
    const ta = document.querySelector('#msg_input textarea');
    if (ta) {
        ta.style.opacity = '0';
    }
}
"""

# --- Gradio UI (Full-Stack) ---
with gr.Blocks(css=FADE_CSS) as interface:
    gr.Markdown("# Assistente Técnico: Gradio, HF Spaces, CrewAI & MCP (100% Local)")

    hardware_monitor = gr.Markdown(value=get_memory_usage())

    chatbot = gr.Chatbot(height=550)

    with gr.Row():
        msg_input = gr.Textbox(
            show_label=False,
            placeholder="Pergunte sobre Gradio, Hugging Face Spaces, CrewAI, MCP ou llama.cpp...",
            scale=9,
            elem_id="msg_input"
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

    # Efeito de fade: listener independente, sem 'fn', só JS puro.
    # Fica separado da chamada acima para não sobrescrever os inputs de generate_response.
    msg_input.submit(fn=None, inputs=None, outputs=None, js=FADE_JS)
    submit_btn.click(fn=None, inputs=None, outputs=None, js=FADE_JS)

    # Limpa o texto e restaura a opacidade (fade-in) após o envio
    submit_event.then(lambda: "", None, [msg_input]).then(
        lambda: None, None, None,
        js="() => { const ta = document.querySelector('#msg_input textarea'); if (ta) ta.style.opacity = '1'; }"
    )
    submit_btn.click(lambda: "", None, [msg_input]).then(
        lambda: None, None, None,
        js="() => { const ta = document.querySelector('#msg_input textarea'); if (ta) ta.style.opacity = '1'; }"
    )

if __name__ == "__main__":
    if llm:
        print("\n Iniciando servidor Gradio local...")
        interface.launch(ssr_mode=False)
    else:
        print("Falha na inicialização. Verifique sua conexão para o download do modelo.")