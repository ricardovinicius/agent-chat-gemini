# app.py
import streamlit as st
import os
import time # Para simular processamento e dar feedback visual

# Importar sua RAGPipeline
from core.rag_pipeline import RAGPipeline #, DEFAULT_RAG_CONFIG (se quiser expor configs)

# Diretório para salvar temporariamente os PDFs enviados
TEMP_PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdf_uploads_temp")
if not os.path.exists(TEMP_PDF_DIR):
    os.makedirs(TEMP_PDF_DIR)

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Agente Conversacional de PDFs",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Conversacional de PDFs")
st.caption("Faça upload de um PDF e converse sobre o seu conteúdo.")

# --- Funções Auxiliares ---
def save_uploaded_file(uploaded_file) -> str:
    """Salva o arquivo enviado em um local temporário e retorna o caminho."""
    file_path = os.path.join(TEMP_PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Inicialização do Estado da Sessão ---
# Usamos st.session_state para manter o estado entre as interações do usuário
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Por favor, faça o upload de um PDF para começarmos."}]

if "rag_pipeline_instance" not in st.session_state:
    st.session_state.rag_pipeline_instance = None # Armazenará nossa instância RAGPipeline

if "current_pdf_path" not in st.session_state:
    st.session_state.current_pdf_path = None # Caminho do PDF processado atualmente

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False # Flag para indicar se um PDF foi processado


# --- Lógica da Barra Lateral para Upload de PDF ---
with st.sidebar:
    st.header("Upload do seu PDF")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    if uploaded_file is not None:
        # Verifica se é um novo PDF ou o mesmo que já foi processado
        temp_pdf_path = os.path.join(TEMP_PDF_DIR, uploaded_file.name) # Caminho temporário para comparação

        if st.session_state.current_pdf_path != temp_pdf_path or not st.session_state.pdf_processed:
            st.info(f"Processando o arquivo: {uploaded_file.name}...")
            saved_pdf_path = save_uploaded_file(uploaded_file) # Salva o arquivo de fato
            st.session_state.current_pdf_path = saved_pdf_path # Atualiza o caminho do PDF atual

            # CONFIGURAÇÃO DA RAG PIPELINE (pode ser personalizada aqui se necessário)
            # Usaremos a configuração padrão da RAGPipeline, que já é in-memory para o vector store
            # Exemplo de como passar uma config customizada, se quisesse persistir:
            custom_config = {
                "embedding_config": {"provider": "huggingface"},
                "llm_config": {"provider": "ollama",
                               "model_name": "gemma3:1b",
                               "temperature": 0.3}
            }
            # st.session_state.rag_pipeline_instance = RAGPipeline(config=custom_config)

            try:
                st.session_state.rag_pipeline_instance = RAGPipeline(custom_config) # Usa config padrão (in-memory)
                with st.spinner("Analisando o PDF e preparando o agente... Isso pode levar um momento."):
                    setup_success = st.session_state.rag_pipeline_instance.setup_for_pdf(saved_pdf_path)

                if setup_success:
                    st.session_state.pdf_processed = True
                    st.success(f"PDF '{uploaded_file.name}' processado com sucesso! Pronto para suas perguntas.")
                    # Limpa mensagens antigas e adiciona mensagem de boas-vindas para o novo PDF
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"Olá! Converse comigo sobre o conteúdo do PDF '{uploaded_file.name}'."}
                    ]
                else:
                    st.session_state.pdf_processed = False
                    st.session_state.rag_pipeline_instance = None
                    st.error("Falha ao processar o PDF. Verifique o console para mais detalhes.")
            except Exception as e:
                st.session_state.pdf_processed = False
                st.session_state.rag_pipeline_instance = None
                st.error(f"Ocorreu um erro inesperado durante o processamento: {e}")
                # import traceback # Para debug mais detalhado no console
                # traceback.print_exc()

        elif st.session_state.pdf_processed:
            st.info(f"PDF '{os.path.basename(st.session_state.current_pdf_path)}' já está carregado.")

    else: # Nenhum arquivo carregado
        if st.session_state.pdf_processed: # Se havia um PDF e ele foi removido do uploader
            st.session_state.messages = [{"role": "assistant", "content": "Olá! Por favor, faça o upload de um PDF para começarmos."}]
            st.session_state.rag_pipeline_instance = None
            st.session_state.current_pdf_path = None
            st.session_state.pdf_processed = False


# --- Exibição do Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input do Usuário (Caixa de Chat) ---
if prompt := st.chat_input("Faça sua pergunta sobre o PDF..." if st.session_state.pdf_processed else "Faça upload de um PDF para começar."):
    if not st.session_state.pdf_processed or not st.session_state.rag_pipeline_instance:
        st.warning("Por favor, faça o upload e processe um PDF primeiro.")
        st.stop() # Interrompe a execução se nenhum PDF estiver pronto

    # Adicionar mensagem do usuário ao histórico do chat e exibir
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Obter e exibir resposta do assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Usado para streaming (se implementado) ou para exibir "Pensando..."
        full_response = ""
        with st.spinner("Pensando... 🤔"):
            try:
                assistant_response = st.session_state.rag_pipeline_instance.ask(prompt)
                if assistant_response:
                    full_response = assistant_response
                else:
                    full_response = "Desculpe, não consegui obter uma resposta."
            except Exception as e:
                full_response = f"Ocorreu um erro ao tentar obter a resposta: {e}"
                # import traceback
                # traceback.print_exc() # Para debug no console

        message_placeholder.markdown(full_response)

    # Adicionar resposta do assistente ao histórico do chat
    st.session_state.messages.append({"role": "assistant", "content": full_response})