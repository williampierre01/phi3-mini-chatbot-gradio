# 🛡️ Edge AI Chatbot (Phi-3 100% Local)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Llama.cpp](https://img.shields.io/badge/llama.cpp-edge-orange.svg)
![Privacy](https://img.shields.io/badge/Privacy-First-green.svg)

## 📌 O Desafio
APIs em nuvem oferecem inferência rápida, mas impõem gargalos arquiteturais graves em cenários do mundo real: dependência de conexão constante, custos recorrentes imprevisíveis e riscos de conformidade em privacidade de dados. Como implementar um assistente de IA conversacional viável em sistemas *air-gapped* ou hardware legado com recursos limitados?

## 💡 A Solução
Este projeto implementa um chatbot com inferência 100% local, desenhado para ambientes de borda (Edge). Utiliza o modelo **Phi-3-Mini (4k instruct)** quantizado em 4-bit (formato GGUF), garantindo um balanço excepcional entre raciocínio lógico e baixo consumo de memória.

A orquestração do modelo é feita através do `llama-cpp-python` para maximizar o uso das threads da CPU local, eliminando a necessidade de GPUs dedicadas. A aplicação é servida através de uma interface reativa construída com **Gradio**.

## ⚙️ Métricas e Hardware Alvo
* **RAM Footprint:** Estabilizado em ~2.5 GB a 3.0 GB (Monitoramento em tempo real integrado na UI).
* **Infraestrutura:** Projetado para rodar em laptops padrão, mini PCs industriais ou edge devices (ex: Raspberry Pi 5).
* **Privacidade:** Zero telemetria ou chamadas de rede externas durante a inferência. Todo o processamento ocorre localmente.

## 🚀 Como Executar

A orquestração de setup e execução foi automatizada via `Makefile` para garantir uma experiência de deploy rápido.

1. Clone este repositório:
```bash
git clone https://github.com/williampierre01/phi3-mini-chatbot-gradio.git
cd phi3-mini-chatbot-gradio