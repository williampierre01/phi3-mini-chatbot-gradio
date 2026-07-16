# local-stack-assistant (Qwen2.5-0.5B-Instruct)

Assistente técnico local, especializado em dúvidas sobre **Gradio, Hugging Face Spaces, CrewAI, Model Context Protocol (MCP) e execução de modelos locais com llama.cpp/GGUF**.

Roda 100% offline, em CPU, sem depender de nenhuma API paga — seus dados nunca saem da máquina.

> O nome do modelo no título é proposital: como o modelo é trocado com frequência durante a fase de otimização/experimentação, ele fica registrado aqui para facilitar a organização entre versões. Veja o [Histórico de performance](#histórico-de-performance-e-decisões) para o racional de cada troca.

---

## O que é

Um chatbot rodando localmente via [llama.cpp](https://github.com/ggml-org/llama.cpp) (através do `llama-cpp-python`) com interface em [Gradio](https://www.gradio.app/), especializado por prompt engineering (sem RAG, por enquanto) para responder apenas sobre o stack de ferramentas acima. Perguntas fora desse escopo são recusadas educadamente pelo próprio modelo.

**Por que um modelo tão pequeno?** Prioriza velocidade e baixo consumo de recursos (roda em ~700MB de RAM, em CPU de 2 núcleos) em troca de conhecimento geral mais limitado. A especialização de nicho existe justamente para compensar essa limitação, concentrando a "atenção" do modelo em um domínio estreito.

## Stack técnica

- **Modelo**: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (quantização Q4_K_M, formato GGUF)
- **Inferência**: `llama-cpp-python` (com `flash_attn` e `use_mlock` habilitados)
- **Interface**: Gradio (`Blocks`, streaming nativo via `yield`)
- **Download do modelo**: `huggingface_hub` (baixado automaticamente no primeiro uso e cacheado)

## Como rodar

```bash
make install   # instala dependências (requirements.txt)
make run       # baixa o modelo (se necessário) e inicia o servidor Gradio
make clean     # limpa o cache do modelo baixado (~/.cache/huggingface/hub)
```

Na primeira execução, o download do modelo (~490MB) pode levar alguns segundos a minutos, dependendo da conexão.

## Escopo do assistente

O assistente foi instruído (via system prompt embutido no código, já que o chat template deste GGUF não trata a role `"system"` corretamente) a responder **apenas** sobre:

- Gradio
- Hugging Face Spaces
- CrewAI
- Model Context Protocol (MCP)
- Execução local de modelos com llama.cpp/GGUF

Perguntas fora desse escopo devem ser recusadas com uma explicação curta, redirecionando para os temas suportados.

⚠️ **Limitação conhecida**: como a especialização hoje é só via prompt (nenhum fine-tuning ainda), o cumprimento do escopo não é garantido em 100% dos casos — modelos desse tamanho (0.5B parâmetros) às vezes não seguem instruções de forma consistente ao longo de conversas mais longas. Isso é medido no `off_topic` do `benchmark.py` (veja abaixo) e é o principal motivador do fine-tuning planejado no roadmap.

## Benchmark

O projeto inclui um script de benchmark automatizado (`benchmark.py`) que roda uma bateria de perguntas de teste direto no modelo (sem precisar da UI), medindo:

- Tempo até o primeiro token (TTFT)
- Tempo total de resposta
- Velocidade de geração (tokens/s)
- Uso de RAM
- **Teste crítico de recusa fora de escopo** (`off_topic`)

```bash
python benchmark.py --output resultados.csv          # bateria completa
python benchmark.py --sweep-batch                    # testa vários valores de n_batch
```

Os resultados são salvos em CSV para análise e comparação entre versões/configurações.

## Histórico de performance e decisões

O projeto passou por várias rodadas de otimização, guiadas por dados reais de benchmark (não só teoria). Resumo da evolução:

| Versão | Modelo | Tempo total (bateria) | Velocidade | RAM | Principal mudança |
|---|---|---|---|---|---|
| v1 | Phi-3-mini (3.8B, Q4) | 519s | 4.30 tok/s | ~2.9 GB | Baseline inicial |
| v3 | Phi-3-mini (3.8B, Q4) | 369s | 4.31 tok/s | ~2.9 GB | System prompt corrigido (workaround de role) |
| v4 | Qwen2.5-1.5B-Instruct | 117s | 9.97 tok/s | ~1.8 GB | Troca de modelo (maior alavanca de ganho) |
| v5 | Qwen2.5-0.5B-Instruct | 60.5s | 17.74 tok/s | ~690 MB | Troca de modelo + `flash_attn` |

**Decisões descartadas** (testadas e revertidas por não compensarem):
- `n_batch` acima de 128 — piorou o throughput (overhead sem ganho, já que o ambiente só tem 2 núcleos de CPU)
- `n_threads` acima de 2 — não há ganho além do número de núcleos físicos disponíveis
- Quantização do KV-cache — ganho marginal dado o `n_ctx` pequeno (1024), não vale o risco de quebrar o carregamento

**Trade-off aceito conscientemente**: modelos menores (1.5B → 0.5B) perdem capacidade de raciocínio (ex: erra contas simples de dias da semana) e de geração de código livre. Esse projeto aceita essa perda porque o escopo é intencionalmente restrito a um nicho técnico bem definido, e não a uso geral.

## Roadmap

- [ ] Validar taxa de recusa fora de escopo (`off_topic`) em uso real
- [ ] Coletar dados reais de conversas para construir dataset de fine-tuning
- [ ] Fine-tuning (LoRA/QLoRA) no domínio, para reduzir dependência de prompt engineering puro
- [ ] Avaliar se um roteador/classificador (modelo pequeno decide se escala para um modelo maior) complementa bem o assistente especializado

## Estrutura do projeto

```
.
├── app.py            # Interface Gradio + lógica de inferência
├── benchmark.py       # Benchmark automatizado (performance + qualidade)
├── Makefile           # Comandos de instalação/execução/limpeza
└── requirements.txt    # Dependências Python
```

## Hardware de referência

Testado em ambiente Codespace com **2 núcleos de CPU** (`nproc = 2`) e sem GPU. Os parâmetros de `n_threads`, `n_batch` etc. em `app.py`/`benchmark.py` estão calibrados para esse hardware — vale reavaliar (rodar `python benchmark.py --sweep-batch`) se migrar para outro ambiente.