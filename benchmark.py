"""
Benchmark automatizado para o chatbot-gradio local.

Roda uma bateria de perguntas de teste diretamente no modelo (sem precisar
da interface Gradio), medindo tempo até o primeiro token (TTFT), tempo
total, velocidade de geração (tokens/s) e uso de RAM. Salva tudo em CSV
para análise posterior.

Uso:
    python benchmark.py
    python benchmark.py --output resultados.csv
    python benchmark.py --sweep-batch          # testa vários valores de n_batch
"""

import argparse
import csv
import os
import time

import psutil
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Configuração (mesma do app.py) ---
# Trocado de Qwen2.5-1.5B para Qwen2.5-0.5B-Instruct: metade do tamanho,
# priorizando velocidade máxima. A especialização de nicho (próxima etapa)
# deve compensar boa parte da perda de conhecimento geral de um modelo tão pequeno.
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
CONTEXT_SIZE = 1024
MAX_RESPONSE_TOKENS = 300  # reduzido de 500: corta a cauda longa de tempo total

SYSTEM_PROMPT = (
    "Você é um assistente virtual que responde SEMPRE em português do Brasil, "
    "de forma clara e direta. Preste muita atenção ao histórico da conversa: "
    "use informações que o usuário já compartilhou (nome, contexto, "
    "preferências) em vez de ignorá-las."
)

# --- Bateria de testes ---
# Cada item é uma "conversa": uma lista de mensagens do usuário enviadas
# em sequência (permite testar memória de contexto multi-turn).
TEST_CASES = [
    {
        "id": "sanity_1",
        "category": "Sanidade básica",
        "turns": ["Olá, quem é você e o que você pode fazer?"],
    },
    {
        "id": "sanity_2",
        "category": "Sanidade básica",
        "turns": ["Me explique o que é machine learning em 3 frases."],
    },
    {
        "id": "memory_1",
        "category": "Memória de contexto",
        "turns": [
            "Meu nome é Chefia e eu trabalho com IA.",
            "Qual é o meu nome e minha área de trabalho?",
        ],
    },
    {
        "id": "long_response",
        "category": "Streaming / resposta longa",
        "turns": [
            "Liste 10 passos para treinar um modelo de deep learning do zero."
        ],
    },
    {
        "id": "context_limit",
        "category": "Limite de contexto (n_ctx=1024)",
        "turns": [
            "Aqui vai um texto para você resumir: A inteligência artificial "
            "tem transformado diversos setores da economia global. Empresas de "
            "tecnologia investem bilhões de dólares em pesquisa e "
            "desenvolvimento de modelos cada vez mais sofisticados. Ao mesmo "
            "tempo, surgem debates sobre regulamentação, ética e o impacto no "
            "mercado de trabalho. Governos ao redor do mundo tentam equilibrar "
            "inovação com segurança, enquanto startups buscam aplicar essas "
            "tecnologias em soluções práticas para problemas reais, desde "
            "diagnóstico médico até otimização de cadeias produtivas. "
            "Resuma esse texto em uma frase."
        ],
    },
    {
        "id": "reasoning",
        "category": "Raciocínio",
        "turns": [
            "Se hoje é quarta-feira e faltam 10 dias para uma reunião, em que "
            "dia da semana ela vai cair?"
        ],
    },
    {
        "id": "code",
        "category": "Código",
        "turns": ["Escreva uma função em Python que verifica se um número é primo."],
    },
    {
        "id": "lang_pt",
        "category": "Português vs inglês (PT)",
        "turns": ["Explique rapidamente o que é uma rede neural convolucional."],
    },
    {
        "id": "lang_en",
        "category": "Português vs inglês (EN)",
        "turns": ["Briefly explain what a convolutional neural network is."],
    },
    {
        "id": "edge_empty",
        "category": "Edge case",
        "turns": ["..."],
    },
]


def load_model(n_batch=128):
    print(f"[*] Verificando cache ou baixando {FILENAME}...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"[*] Carregando modelo via llama.cpp (Edge CPU Mode, n_batch={n_batch})...")
    llm = Llama(
        model_path=model_path,
        n_ctx=CONTEXT_SIZE,
        n_threads=2,         # Codespace tem só 2 cores (nproc=2) - manter em 2
        n_threads_batch=2,   # garante que o prefill do prompt também usa os 2 cores
        n_batch=n_batch,
        use_mlock=True,      # trava o modelo em RAM, evita swap e picos de latência
        flash_attn=True,     # reduz movimentação de dados durante a atenção (1.3x-2x em prompts longos)
        verbose=False,
    )

    print("[*] Aquecendo o modelo (warm-up)...")
    list(llm.create_chat_completion(
        messages=[{"role": "user", "content": "oi"}],
        max_tokens=1,
        stream=False,
    ))
    print("[*] Warm-up concluído.\n")

    return llm


def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_turn(llm, messages, max_tokens=MAX_RESPONSE_TOKENS, temperature=0.1):
    """Executa uma chamada de streaming e mede as métricas de performance."""
    start_time = time.time()
    first_token_time = None
    token_count = 0
    response_text = ""

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            response_text += content
            token_count += 1

    total_time = time.time() - start_time
    ttft = first_token_time if first_token_time is not None else total_time
    tokens_per_sec = token_count / total_time if total_time > 0 else 0.0

    return {
        "response": response_text,
        "ttft": ttft,
        "total_time": total_time,
        "tokens": token_count,
        "tokens_per_sec": tokens_per_sec,
    }


def run_benchmark(llm, test_cases):
    records = []

    for case in test_cases:
        # O chat template deste GGUF ignora a role "system"; embutimos a
        # instrução na primeira mensagem do usuário como workaround.
        history = []
        for turn_idx, user_message in enumerate(case["turns"], start=1):
            if turn_idx == 1:
                user_message_for_model = f"{SYSTEM_PROMPT}\n\n{user_message}"
            else:
                user_message_for_model = user_message
            history.append({"role": "user", "content": user_message_for_model})
            ram_before = get_ram_mb()

            print(f"\n[{case['id']} | turno {turn_idx}] Pergunta: {user_message[:80]}")
            result = run_turn(llm, history)

            ram_after = get_ram_mb()
            history.append({"role": "assistant", "content": result["response"]})

            record = {
                "case_id": case["id"],
                "category": case["category"],
                "turn": turn_idx,
                "prompt": user_message,
                "response_preview": result["response"][:200].replace("\n", " "),
                "ttft_s": round(result["ttft"], 3),
                "total_time_s": round(result["total_time"], 3),
                "tokens": result["tokens"],
                "tokens_per_sec": round(result["tokens_per_sec"], 2),
                "ram_before_mb": round(ram_before, 2),
                "ram_after_mb": round(ram_after, 2),
            }
            records.append(record)

            print(
                f"  -> TTFT: {record['ttft_s']}s | Total: {record['total_time_s']}s | "
                f"{record['tokens']} tokens | {record['tokens_per_sec']} tok/s | "
                f"RAM: {record['ram_after_mb']} MB"
            )

    return records


def run_batch_sweep(batch_values, prompt=None):
    """Recarrega o modelo com diferentes valores de n_batch e mede o impacto
    no TTFT/velocidade, usando um único prompt representativo (o mais longo
    da bateria, que estressa o prefill)."""
    if prompt is None:
        # usa o prompt mais longo (context_limit) como referência, já que é
        # o que mais evidencia o custo de prefill/batch
        prompt = next(c["turns"][0] for c in TEST_CASES if c["id"] == "context_limit")

    sweep_records = []
    for n_batch in batch_values:
        print(f"\n{'='*70}\nTestando n_batch={n_batch}\n{'='*70}")
        llm = load_model(n_batch=n_batch)
        messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}]
        result = run_turn(llm, messages)
        sweep_records.append({
            "n_batch": n_batch,
            "ttft_s": round(result["ttft"], 3),
            "total_time_s": round(result["total_time"], 3),
            "tokens": result["tokens"],
            "tokens_per_sec": round(result["tokens_per_sec"], 2),
        })
        print(
            f"  -> TTFT: {sweep_records[-1]['ttft_s']}s | "
            f"Total: {sweep_records[-1]['total_time_s']}s | "
            f"{sweep_records[-1]['tokens_per_sec']} tok/s"
        )
        del llm  # libera RAM antes de recarregar com o próximo valor

    return sweep_records


def save_csv(records, output_path):
    if not records:
        print("Nenhum resultado para salvar.")
        return

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[*] Resultados salvos em: {output_path}")


def print_summary(records):
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    total_time = sum(r["total_time_s"] for r in records)
    avg_ttft = sum(r["ttft_s"] for r in records) / len(records)
    avg_tps = sum(r["tokens_per_sec"] for r in records) / len(records)
    print(f"Total de perguntas testadas: {len(records)}")
    print(f"Tempo total do benchmark:    {total_time:.2f}s")
    print(f"TTFT médio:                  {avg_ttft:.2f}s")
    print(f"Velocidade média:            {avg_tps:.2f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark do chatbot local")
    parser.add_argument(
        "--output", default="benchmark_results.csv", help="Arquivo CSV de saída"
    )
    parser.add_argument(
        "--sweep-batch",
        action="store_true",
        help="Em vez da bateria completa, testa vários valores de n_batch "
             "(64, 128, 192, 256) no prompt mais longo, para achar o melhor.",
    )
    args = parser.parse_args()

    if args.sweep_batch:
        print("\n[*] Iniciando sweep de n_batch...\n")
        sweep_results = run_batch_sweep([64, 128, 192, 256])
        sweep_output = args.output.replace(".csv", "_batch_sweep.csv")
        save_csv(sweep_results, sweep_output)
        raise SystemExit(0)

    llm = load_model()
    if not llm:
        print("[!] Falha ao carregar o modelo. Abortando.")
        raise SystemExit(1)

    print("\n[*] Iniciando bateria de testes...\n")
    records = run_benchmark(llm, TEST_CASES)

    save_csv(records, args.output)
    print_summary(records)