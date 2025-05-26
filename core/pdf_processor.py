from langchain_community.document_loaders import PyMuPDFLoader # Ou PyPDFLoader se preferir
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Carrega um arquivo PDF e divide seu conteúdo em chunks.

    Args:
        pdf_file_path (str): O caminho para o arquivo PDF.
        chunk_size (int): O tamanho máximo de cada chunk (em caracteres).
        chunk_overlap (int): A sobreposição de caracteres entre chunks adjacentes.

    Returns:
        list: Uma lista de documentos (chunks) do LangChain.
    """
    try:
        # 1. Carregar o PDF
        # Usaremos PyMuPDFLoader que geralmente é rápido e bom com formatação.
        # Alternativamente, PyPDFLoader: from langchain_community.document_loaders import PyPDFLoader
        loader = PyMuPDFLoader(pdf_file_path)
        documents = loader.load()

        if not documents:
            print(f"Nenhum documento carregado do PDF: {pdf_file_path}")
            return []

        # 2. Dividir os documentos em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Útil para referenciar a origem do chunk
        )
        chunks = text_splitter.split_documents(documents)

        print(f"PDF '{pdf_file_path}' carregado e dividido em {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        print(f"Erro ao processar o PDF {pdf_file_path}: {e}")
        return []

# Exemplo de como você pode testar este módulo isoladamente (opcional, coloque em um script de teste)
if __name__ == '__main__':
    # Crie uma pasta 'data/pdf_uploads/' na raiz do projeto e coloque um PDF de teste lá.
    # Exemplo: 'data/pdf_uploads/meu_documento_teste.pdf'
    # Certifique-se que o caminho para o PDF de teste está correto.
    test_pdf_path = "../data/pdf_uploads/exemplo.pdf" # Ajuste este caminho!
                                                     # Se rodar de dentro da pasta 'core', o caminho é '../data/...'
                                                     # Se rodar da raiz do projeto, o caminho é 'data/...'

    # Para testar, crie um PDF simples chamado 'exemplo.pdf' na pasta 'data/pdf_uploads'
    # ou use um PDF existente e ajuste o caminho.
    # Exemplo: Crie um arquivo de texto simples, salve como PDF.

    # Verifique se o diretório data/pdf_uploads existe
    import os
    if not os.path.exists(os.path.dirname(test_pdf_path)):
        os.makedirs(os.path.dirname(test_pdf_path))
        print(f"Criado diretório: {os.path.dirname(test_pdf_path)}")
        print(f"Por favor, adicione um arquivo PDF de teste em {test_pdf_path} para executar o teste.")

    if os.path.exists(test_pdf_path):
        chunks = load_and_chunk_pdf(test_pdf_path)
        if chunks:
            print(f"\nPrimeiro chunk ({len(chunks[0].page_content)} caracteres):")
            print(chunks[0].page_content)
            print(f"\nMetadados do primeiro chunk: {chunks[0].metadata}")
        else:
            print("Nenhum chunk foi gerado.")
    else:
        print(f"Arquivo de teste não encontrado em {test_pdf_path}. Crie um PDF de teste para prosseguir.")