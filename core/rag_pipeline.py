# core/rag_pipeline.py
import os
from typing import Dict, Any, Optional

# Importar dos seus módulos locais
from core.pdf_processor import load_and_chunk_pdf
from core.llm_handler import get_embeddings_model, get_llm
from core.vector_store import get_vector_store, add_documents_to_store

# Importar do LangChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever # Para type hinting

# Configurações padrão para o RAG Pipeline
DEFAULT_RAG_CONFIG = {
    "pdf_processor_config": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
    "embedding_config": {
        "provider": "huggingface", # ou "google", "openai"
        "model_name": None, # Será usado o padrão do provedor ou pode ser especificado
    },
    "vector_store_config": {
        # O persist_directory será baseado no nome do arquivo PDF para isolamento
        "mode": "in_memory",
        "collection_name_prefix": "rag_collection_",
    },
    "llm_config": {
        "provider": "google", # ou "openai"
        "model_name": "gemini-1.5-flash-latest", # Modelo rápido e eficiente para RAG
        "temperature": 0.3, # Baixa temperatura para respostas mais factuais e baseadas no contexto
    },
    "retriever_config": {
        "search_type": "similarity", # "similarity", "mmr", "similarity_score_threshold"
        "search_kwargs": {"k": 10}, # Recuperar os 10 chunks mais relevantes
    }
}

RAG_PROMPT_TEMPLATE = """
Você é um assistente de IA especializado em responder perguntas com base em um contexto fornecido.
Use APENAS as seguintes informações de contexto para responder à pergunta.
Se a resposta não estiver contida no contexto, diga educadamente que você não encontrou a informação no documento.
NÃO invente respostas nem use conhecimento externo. Seja conciso e direto.

Contexto:
{context}

Pergunta:
{question}

Resposta útil:
"""

class RAGPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_RAG_CONFIG, **(config if config else {})}
        self.embeddings_model = None
        self.vector_db = None
        self.llm = None
        self.rag_chain = None
        self.retriever: Optional[BaseRetriever] = None
        print("RAGPipeline inicializado com a configuração.")

    def _get_pdf_specific_persist_directory(self, pdf_file_path: str) -> str:
        """Cria um diretório de persistência específico para o PDF."""
        pdf_filename = os.path.splitext(os.path.basename(pdf_file_path))[0]
        safe_filename = "".join(c if c.isalnum() else "_" for c in pdf_filename) # Torna o nome seguro para diretório
        # Assume que este script está em core/, então .. vai para a raiz do projeto
        base_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "vector_stores")
        persist_dir = os.path.join(base_data_dir, safe_filename)
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
            print(f"Criado diretório de persistência: {persist_dir}")
        return persist_dir

    def setup_for_pdf(self, pdf_file_path: str):
        """
        Configura todo o pipeline RAG para um arquivo PDF específico.
        Isso inclui processar o PDF, criar embeddings, popular o vector store e montar a cadeia RAG.
        """
        print(f"\n--- Configurando pipeline RAG para: {pdf_file_path} ---")

        # 1. Processar PDF: Carregar e dividir em chunks
        print("Etapa 1: Processando PDF...")
        chunks = load_and_chunk_pdf(pdf_file_path)
        if not chunks:
            print(f"Nenhum chunk gerado para {pdf_file_path}. Interrompendo setup.")
            self.rag_chain = None # Garante que não haja uma cadeia antiga
            return False

        # 2. Obter modelo de embeddings
        print("Etapa 2: Carregando modelo de embeddings...")
        emb_conf = self.config["embedding_config"]
        self.embeddings_model = get_embeddings_model(
            provider=emb_conf.get("provider", "huggingface"),
            model_name=emb_conf.get("model_name")
        )

        # 3. Configurar Vector Store
        print("Etapa 3: Configurando Vector Store...")
        vs_conf = self.config["vector_store_config"]
        vs_mode = vs_conf.get("mode",
                              "in_memory")  # Pega o modo, padrão para in_memory se não especificado

        persist_dir_path = None
        is_persistent_store = False
        if vs_mode == "persistent":
            base_persist_dir = vs_conf.get("persist_directory_base",
                                           os.path.join("data",
                                                        "vector_stores_persistent_default"))
            persist_dir_path = self._get_pdf_specific_persist_directory_path(
                pdf_file_path, base_persist_dir)
            is_persistent_store = True
            print(
                f"Modo do Vector Store: Persistente. Diretório: {persist_dir_path}")
            # Opcional: Limpar o diretório persistente antes de adicionar para evitar duplicatas de execuções anteriores
            # if os.path.exists(persist_dir_path) and is_persistent_store:
            #     print(f"Limpando diretório persistente anterior: {persist_dir_path}")
            #     import shutil
            #     shutil.rmtree(persist_dir_path)
        else:
            print("Modo do Vector Store: In-Memory.")

        # Nome da coleção (usado tanto para in-memory quanto persistente)
        pdf_filename_safe = \
        os.path.splitext(os.path.basename(pdf_file_path))[0]
        pdf_filename_safe = "".join(
            c if c.isalnum() else "_" for c in pdf_filename_safe)
        collection_name = f"{vs_conf['collection_name_prefix']}{pdf_filename_safe}"

        self.vector_db = get_vector_store(
            embeddings_model=self.embeddings_model,
            collection_name=collection_name,
            persist_directory=persist_dir_path
            # Será None para in-memory
        )
        # Adicionar documentos. Se for in-memory, será uma nova store a cada setup_for_pdf.
        # Se for persistente, e o diretório não for limpo, pode adicionar duplicatas.
        # A lógica de limpeza de diretório (comentada acima) pode ser útil para persistente.
        print(
            f"Adicionando {len(chunks)} chunks ao vector store para a coleção '{collection_name}'...")
        add_documents_to_store(self.vector_db, chunks,
                               is_persistent=is_persistent_store)

        # 4. Criar Retriever
        print("Etapa 4: Criando Retriever...")
        self.retriever = self.vector_db.as_retriever(
            search_type=self.config["retriever_config"]["search_type"],
            search_kwargs=self.config["retriever_config"]["search_kwargs"]
        )

        # 5. Obter LLM
        print("Etapa 5: Carregando LLM...")
        llm_conf = self.config["llm_config"]
        self.llm = get_llm(
            provider=llm_conf["provider"],
            model_name=llm_conf["model_name"],
            temperature=llm_conf["temperature"]
        )

        # 6. Definir Prompt Template
        print("Etapa 6: Configurando Prompt Template...")
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # 7. Construir Cadeia RAG usando LCEL
        # A cadeia fará o seguinte:
        # - Recebe a pergunta do usuário.
        # - Usa o retriever para buscar chunks de contexto relevantes.
        # - Formata o prompt com a pergunta e o contexto.
        # - Envia o prompt para o LLM.
        # - Retorna a resposta do LLM.
        print("Etapa 7: Construindo a cadeia RAG...")

        # Função para formatar os documentos recuperados (contexto)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            RunnableParallel(
                context=(RunnablePassthrough() | self.retriever | format_docs), # Pega a pergunta, passa para o retriever, formata
                question=RunnablePassthrough() # Passa a pergunta original adiante
            )
            | rag_prompt          # Preenche o prompt com 'context' e 'question'
            | self.llm            # Envia para o LLM
            | StrOutputParser()   # Analisa a saída do LLM para string
        )
        print("--- Pipeline RAG configurado com sucesso! ---")
        return True

    def ask(self, question: str) -> Optional[str]:
        """
        Faz uma pergunta à cadeia RAG configurada.

        Args:
            question (str): A pergunta do usuário.

        Returns:
            Optional[str]: A resposta do LLM, ou None se a cadeia RAG não estiver configurada.
        """
        if not self.rag_chain:
            print("Erro: A cadeia RAG não foi configurada. Chame setup_for_pdf() primeiro.")
            return None
        if not question:
            print("Erro: Nenhuma pergunta fornecida.")
            return "Por favor, forneça uma pergunta."

        print(f"\nProcessando pergunta: {question}")
        try:
            # O input para a cadeia LCEL com RunnableParallel no início é a pergunta.
            # Se a chave "question" for usada no RunnableParallel, o input deve ser um dict: {"question": question}
            # No nosso caso, RunnablePassthrough() no início da ramificação 'context' e 'question'
            # significa que o input (a pergunta) é passado para ambas as ramificações.
            response = self.rag_chain.invoke(question)
            return response
        except Exception as e:
            print(f"Erro ao invocar a cadeia RAG: {e}")
            # import traceback
            # traceback.print_exc()
            return "Ocorreu um erro ao processar sua pergunta."


# --- Bloco de Teste (para executar este arquivo diretamente) ---
if __name__ == '__main__':
    print("--- Iniciando Teste do RAGPipeline ---")

    # Configurar o caminho para o PDF de teste
    # Se este script está em 'core/', o caminho relativo para a pasta data é '../data/'
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    test_pdf_filename = "exemplo.pdf"  # Use o mesmo PDF de teste do pdf_processor.py
    test_pdf_path = os.path.join(project_root, "data", "pdf_uploads", test_pdf_filename)

    print(f"Usando PDF de teste: {test_pdf_path}")

    if not os.path.exists(test_pdf_path):
        print(f"AVISO: Arquivo PDF de teste '{test_pdf_path}' não encontrado.")
        print("Por favor, certifique-se que o PDF de teste do 'pdf_processor.py' existe ou crie um novo.")

    if os.path.exists(test_pdf_path):
        custom_config = {
             "embedding_config": {"provider": "huggingface"}, # requer GOOGLE_API_KEY
             "llm_config": {"provider": "ollama", "model_name": "qwen3:1.7b", "temperature": 0.3}
        }

        # Para este teste, vamos usar a configuração padrão (HuggingFace embeddings, Google LLM)
        # Se você não tiver GOOGLE_API_KEY configurada, a parte do LLM falhará.
        # Se tiver, certifique-se que está no .env
        # rag_pipeline_instance = RAGPipeline() # Usa DEFAULT_RAG_CONFIG
        rag_pipeline_instance = RAGPipeline(config=custom_config) # Para usar config customizada

        setup_success = rag_pipeline_instance.setup_for_pdf(test_pdf_path)

        if setup_success:
            test_questions = [
                "Qual o nome do estudante?",
                "Qual a data de nascimento do estudante?",
                "Qual a idade do estudante (hoje é 25/05/2025)"
            ]

            for q in test_questions:
                print("-" * 50)
                answer = rag_pipeline_instance.ask(q)
                print(f"P: {q}")
                print(f"R: {answer}")
            print("-" * 50)
        else:
            print("Não foi possível configurar o pipeline RAG. Teste interrompido.")
    else:
        print(f"PDF de teste '{test_pdf_path}' ainda não encontrado. Não foi possível executar o teste do RAGPipeline.")

    print("\n--- Teste do RAGPipeline Concluído ---")