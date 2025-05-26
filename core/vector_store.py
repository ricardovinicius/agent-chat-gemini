# core/vector_store.py
from typing import Optional

from langchain_chroma import Chroma
# from langchain_core.embeddings import Embeddings # Para type hinting, se desejar

# Diretório onde o ChromaDB persistirá os dados.
# Relativo à raiz do projeto se este script for chamado de lá.
DEFAULT_PERSIST_DIRECTORY = "data/vector_db"

def get_vector_store(
    embeddings_model, # Espera um objeto de embedding do LangChain
    collection_name: str = "rag_documents_collection", # Nome da coleção dentro do ChromaDB
    persist_directory: Optional[str] = None # Tornar opcional
):
    """
    Cria ou carrega um banco de dados vetorial ChromaDB.
    Se persist_directory for None, opera em modo in-memory.

    Args:
        embeddings_model: O modelo de embedding a ser usado pelo Chroma.
        collection_name (str): Nome da coleção para os documentos.
        persist_directory (Optional[str]): O diretório onde o ChromaDB armazenará os dados.
                                           Se None, o ChromaDB operará em memória.
    Returns:
        Um objeto Chroma vector store.
    """
    if persist_directory:
        if not os.path.exists(persist_directory):
            try:
                os.makedirs(persist_directory)
                print(f"Criado diretório de persistência: {os.path.abspath(persist_directory)}")
            except OSError as e:
                print(f"Erro ao criar diretório de persistência '{persist_directory}': {e}")
                # Poderia levantar o erro ou tentar continuar em memória como fallback
                print("Tentando operar em modo in-memory como fallback.")
                persist_directory = None # Fallback para in-memory

        print(f"Vector store ChromaDB será configurado para persistência.")
        print(f"Coleção: '{collection_name}'. Diretório: '{os.path.abspath(persist_directory if persist_directory else '')}'")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_model,
            persist_directory=persist_directory
        )
    else:
        print(f"Vector store ChromaDB operando em modo IN-MEMORY.")
        print(f"Coleção: '{collection_name}'. Os dados serão perdidos ao final da sessão.")
        # Para in-memory, não passamos persist_directory
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_model
            # persist_directory é omitido para modo in-memory
        )
    return vector_store

def add_documents_to_store(
    vector_store: Chroma,
    documents: list, # Lista de objetos Document do LangChain (chunks)
    is_persistent: bool = False # Indica se o store é persistente
):
    """
    Adiciona documentos (chunks) ao Chroma vector store.
    Os embeddings serão gerados automaticamente pelo Chroma usando o `embeddings_model` fornecido na sua criação.

    Args:
        vector_store (Chroma): A instância do ChromaDB.
        documents (list): Uma lista de documentos (chunks) do LangChain.
        is_persistent (bool): True se o vector_store foi configurado para persistência.
    """
    if not documents:
        print("Nenhum documento para adicionar ao vector store.")
        return

    print(f"Adicionando {len(documents)} documentos ao vector store (Coleção: {vector_store._collection.name})...") # Acessando nome da coleção
    vector_store.add_documents(documents)

    if is_persistent:
        # Forçar a persistência (o Chroma geralmente persiste automaticamente para algumas operações,
        # mas é bom ser explícito após uma grande adição de documentos se o cliente for persistente)
        print(f"Persistindo vector store (Coleção: {vector_store._collection.name})...")
    else:
        print("Documentos adicionados ao vector store in-memory.")


# Exemplo de como você pode testar este módulo isoladamente
if __name__ == '__main__':
    # Este teste depende dos módulos pdf_processor e llm_handler
    from pdf_processor import load_and_chunk_pdf
    from llm_handler import get_embeddings_model
    import os

    # Caminho para o PDF de teste (ajuste conforme necessário)
    # Se rodar de dentro de 'core', o caminho é '../data/...'
    test_pdf_path = "../data/pdf_uploads/exemplo.pdf"

    # Certifique-se que o PDF de teste existe
    if not os.path.exists(test_pdf_path):
        # Tenta criar o diretório se não existir
        pdf_dir = os.path.dirname(test_pdf_path)
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            print(f"Criado diretório: {pdf_dir}")
        print(f"Por favor, adicione um arquivo PDF de teste em '{test_pdf_path}' para executar o teste.")
    else:
        print("Iniciando teste do vector_store...")
        try:
            # 1. Carregar e dividir o PDF
            print("\n--- Etapa 1: Processando PDF ---")
            chunks = load_and_chunk_pdf(test_pdf_path)

            if chunks:
                # 2. Obter o modelo de embedding
                print("\n--- Etapa 2: Carregando Modelo de Embedding ---")
                # Usaremos HuggingFace por padrão para este teste
                embeddings = get_embeddings_model(provider="huggingface")

                # 3. Obter/Criar o Vector Store
                print("\n--- Etapa 3: Configurando Vector Store ---")
                # Usaremos um diretório de persistência diferente para o teste para não sujar o principal
                test_persist_dir = "../data/vector_db_test"
                if not os.path.exists(test_persist_dir):
                    os.makedirs(test_persist_dir)
                vector_db = get_vector_store(embeddings_model=embeddings, persist_directory=test_persist_dir, collection_name="test_collection")

                # 4. Adicionar documentos ao store
                print("\n--- Etapa 4: Adicionando Documentos ao Store ---")
                add_documents_to_store(vector_db, chunks)

                # 5. Testar a contagem de itens no store (opcional, mas útil)
                # O Chroma não tem um .count() direto e fácil sem carregar tudo para alguns clientes.
                # Mas podemos tentar uma busca simples para ver se algo foi inserido.
                print("\n--- Etapa 5: Verificando o Store ---")
                retriever = vector_db.as_retriever(search_kwargs={"k": 1})
                sample_query_text = chunks[0].page_content[:50] # Pega um pedaço do primeiro chunk como query
                retrieved_docs = retriever.invoke(sample_query_text)

                if retrieved_docs:
                    print(f"Busca de teste retornou {len(retrieved_docs)} documento(s). O primeiro é:")
                    print(retrieved_docs[0].page_content[:200] + "...")
                    print(f"Metadados: {retrieved_docs[0].metadata}")
                else:
                    print("Busca de teste não retornou documentos. Verifique a ingestão.")

                print("\nTeste do vector_store concluído com sucesso!")
                print(f"Dados do teste persistidos em: {os.path.abspath(test_persist_dir)}")
                print("Você pode apagar esta pasta ('vector_db_test') após o teste se desejar.")

            else:
                print("Nenhum chunk gerado do PDF. Verifique o processamento do PDF.")

        except Exception as e:
            print(f"Erro durante o teste do vector_store: {e}")