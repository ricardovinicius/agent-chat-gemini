# 🤖 Agente Conversacional de PDFs com RAG e LLM

Este projeto implementa um agente conversacional capaz de "ler" e responder perguntas sobre o conteúdo de documentos PDF. Ele utiliza a arquitetura RAG (Retrieval Augmented Generation) com Modelos de Linguagem Amplos (LLMs) para fornecer respostas contextuais e precisas, baseadas exclusivamente nas informações presentes no documento fornecido.

## ✨ Funcionalidades

* **Upload de Documentos PDF:** Interface amigável para enviar arquivos PDF.
* **Processamento Inteligente:** Extração de texto, divisão em chunks semânticos e geração de embeddings.
* **Armazenamento Vetorial:** Indexação dos chunks para busca rápida por similaridade (por padrão, opera em memória para cada sessão de PDF, mas pode ser configurado para persistência).
* **Interface de Chat Interativa:** Converse com o agente sobre o conteúdo do PDF carregado.
* **Respostas Contextualizadas (RAG):** O LLM responde perguntas utilizando apenas o contexto extraído do PDF.
* **Suporte Flexível a LLMs:**
    * Integrado com LLMs locais via **Ollama** (ex: Qwen, Mistral, Llama).
    * Facilmente configurável para usar LLMs via API (ex: Google Gemini, OpenAI GPT).
* **Modelo de Embedding Configurável:** Suporte para modelos de embedding locais (via HuggingFace Sentence Transformers) ou via API.

## 🏛️ Arquitetura Simplificada

A aplicação segue um fluxo RAG:

1.  **Interface do Usuário (Streamlit):** O usuário faz upload do PDF e interage via chat.
2.  **Processador de PDF (`PDFProcessor`):** Carrega o PDF, extrai o texto e o divide em chunks.
3.  **Geração de Embeddings (`LLMHandler`):** Transforma os chunks de texto em vetores numéricos (embeddings).
4.  **Banco de Dados Vetorial (`VectorStore` - ChromaDB):** Armazena os chunks e seus embeddings. Para cada PDF processado, um novo banco vetorial em memória é criado por padrão, garantindo que as conversas sejam isoladas por documento e sessão.
5.  **Pipeline RAG (`RAGPipeline`):**
    * Quando o usuário faz uma pergunta:
        * A pergunta é convertida em um embedding.
        * O `Retriever` busca os chunks mais relevantes no banco vetorial.
        * Os chunks relevantes (contexto) e a pergunta são enviados para o LLM (via `LLMHandler`).
    * O LLM gera uma resposta baseada no contexto fornecido.
6.  **LLM (`LLMHandler`):** Modelo de Linguagem Amplo que gera as respostas.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Interface:** Streamlit
* **Orquestração RAG e LLM:** LangChain
* **LLMs Locais:** Ollama (permite rodar modelos como Qwen, Mistral, Llama 3, etc.)
* **LLMs via API (Opcional):** Google Gemini, OpenAI GPT
* **Embeddings:** HuggingFace Sentence Transformers (padrão, local), Google Gemini Embeddings, OpenAI Embeddings
* **Banco de Dados Vetorial:** ChromaDB (operando em memória por padrão para cada PDF)
* **Processamento de PDF:** PyMuPDF

## ⚙️ Pré-requisitos

* Python (versão 3.9 ou superior recomendada)
* Git
* **Opcional (para LLMs locais):** Ollama instalado e configurado.
    * Certifique-se de ter baixado os modelos que deseja usar (ex: `ollama pull qwen:1.8b`, `ollama pull mistral`).
* **Opcional (para aceleração de GPU com Ollama):** Drivers NVIDIA atualizados e CUDA (para GPUs NVIDIA) ou configuração apropriada para GPUs AMD/Apple Silicon.

## 🚀 Configuração e Instalação

1.  **Clone o Repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO_AQUI>
    cd nome-do-diretorio-do-projeto
    ```

2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv .venv
    ```
    * No Linux/macOS:
        ```bash
        source .venv/bin/activate
        ```
    * No Windows (PowerShell):
        ```bash
        .\.venv\Scripts\Activate.ps1
        ```
    * No Windows (CMD):
        ```bash
        .\.venv\Scripts\activate.bat
        ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Se você encontrar um erro relacionado ao `protobuf` durante a instalação ou execução, tente:*
    ```bash
    pip install protobuf==3.20.3
    ```

4.  **Configure as Variáveis de Ambiente (Opcional):**
    * Se você planeja usar LLMs ou modelos de embedding via API (Google Gemini, OpenAI), crie um arquivo chamado `.env` na raiz do projeto.
    * Copie o conteúdo de `.env.example` (se você criar um) para `.env` e preencha suas chaves de API:
      ```env
      # .env
      GOOGLE_API_KEY="sua_chave_api_google_aqui"
      OPENAI_API_KEY="sua_chave_api_openai_aqui"
      ```
    * A configuração padrão no `core/rag_pipeline.py` utiliza embeddings locais (HuggingFace) e tenta usar Google Gemini para o LLM. Se `GOOGLE_API_KEY` não estiver definida, a chamada ao LLM falhará. Você pode ajustar `DEFAULT_RAG_CONFIG` em `core/rag_pipeline.py` para usar Ollama como provedor de LLM por padrão, se preferir.

## ▶️ Executando a Aplicação

1.  **Inicie o Servidor Ollama (se estiver usando LLMs locais e ele não estiver rodando como serviço):**
    Abra um terminal e execute:
    ```bash
    ollama serve
    ```
    * Certifique-se de que o modelo que você pretende usar com Ollama (ex: `qwen:1.8b`) já foi baixado (`ollama pull nome_do_modelo`).

2.  **Execute a Aplicação Streamlit:**
    No terminal (com o ambiente virtual ativado), na pasta raiz do projeto, execute:
    ```bash
    streamlit run app.py
    ```
    A aplicação deverá abrir automaticamente no seu navegador web.

## 📖 Como Usar

1.  Acesse a interface no navegador (geralmente `http://localhost:8501`).
2.  Na barra lateral esquerda, clique em "Escolha um arquivo PDF" e selecione o documento PDF que deseja analisar.
3.  Aguarde o processamento do PDF. Você verá mensagens de status.
4.  Uma vez que o PDF esteja processado, a caixa de chat será habilitada. Digite sua pergunta sobre o conteúdo do documento e pressione Enter.
5.  O agente responderá com base nas informações encontradas no PDF.

## 📁 Estrutura do Projeto (Visão Geral)
meu_agente_pdf/
├── app.py                   # Código da interface Streamlit
├── core/                    # Lógica principal da aplicação
│   ├── init.py
│   ├── pdf_processor.py     # Carregamento e divisão de PDFs
│   ├── llm_handler.py       # Gerenciamento de LLMs e modelos de embedding
│   ├── vector_store.py      # Gerenciamento do banco de dados vetorial (ChromaDB)
│   └── rag_pipeline.py      # Orquestração do pipeline RAG
├── data/                    # Dados gerados ou temporários
│   ├── pdf_uploads_temp/    # Armazenamento temporário de PDFs enviados
│   └── vector_stores_persistent/ # (Opcional) Para vector stores persistentes, se configurado
├── .env                     # (Local, não versionado) Chaves de API e outras configurações secretas
├── requirements.txt         # Lista de dependências Python
└── README.md                # Este arquivo
