# core/llm_handler.py
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- Modelos de Embedding ---
def get_embeddings_model(provider: str = "huggingface", model_name: str = None):
    """
    Configura e retorna o modelo de embedding especificado.

    Args:
        provider (str): O provedor do modelo de embedding ('huggingface', 'openai', 'google').
        model_name (str): O nome específico do modelo (opcional para alguns provedores).

    Returns:
        Um objeto de modelo de embedding do LangChain.
    """
    if provider == "huggingface":
        # Gratuito, roda localmente. Bom para começar.
        # 'all-MiniLM-L6-v2' é um modelo popular, eficiente e com bom desempenho.
        hf_model_name = model_name if model_name else "all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(
            model_name=hf_model_name,
            model_kwargs={'device': 'cuda'}
        )
    else:
        raise ValueError(f"Provedor de embedding desconhecido: {provider}")

# --- Modelos de Linguagem (LLMs) ---
def get_llm(provider: str = "google", model_name: str = None, temperature: float = 0.3):
    """
    Configura e retorna o modelo de linguagem (LLM) especificado.

    Args:
        provider (str): O provedor do LLM ('google', 'openai').
        model_name (str): O nome específico do modelo.
        temperature (float): Parâmetro de criatividade/aleatoriedade do LLM. (0.0 a 1.0+)

    Returns:
        Um objeto LLM do LangChain.
    """
    api_key = None
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chave de API do Google (GOOGLE_API_KEY) não encontrada no arquivo .env")
        llm_model_name = model_name if model_name else "gemini-1.5-flash-latest"
        print(f"Usando Google Gemini LLM: {llm_model_name}")
        return ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=api_key, temperature=temperature)
    elif provider == "ollama":
        llm_model_name = model_name if model_name else "llama3.2"
        print(f"Usando Ollama: {llm_model_name}")
        return ChatOllama(model=llm_model_name, temperature=temperature)
    else:
        raise ValueError(f"Provedor de LLM desconhecido: {provider}")

# Exemplo de como você pode testar este módulo isoladamente
if __name__ == '__main__':
    try:
        print("Testando Hugging Face Embeddings (local):")
        hf_embeddings = get_embeddings_model(provider="huggingface")
        sample_text = "Olá, mundo! Este é um texto de exemplo."
        embedding_vector = hf_embeddings.embed_query(sample_text)
        print(f"Embedding gerado com sucesso. Dimensões: {len(embedding_vector)}")

        print("\nTestando Google Gemini LLM (requer API Key):")
        try:
            llm = get_llm(provider="google",
                          temperature=0.1)  # Use baixa temperatura para respostas mais factuais
            response = llm.invoke(
                "Olá! Faça um resumo conciso sobre a importância da IA generativa.")
            print(f"Resposta do LLM (Google): {response.content}")
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Erro ao testar LLM (Google): {e}")
    except Exception as e:
        print(f"Erro durante o teste dos embeddings: {e}")