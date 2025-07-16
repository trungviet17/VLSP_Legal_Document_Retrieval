import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.envconfig import EnvConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class ModelFactory:

    @staticmethod 
    def get_openai_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=EnvConfig.OPENAI_API_KEY
        )
    

    @staticmethod
    def get_openai_embeddings(model_name: str = "text-embedding-3-small"):
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=EnvConfig.OPENAI_API_KEY
        )


    @staticmethod
    def get_google_llm(model_name: str = "gemini-1.5-flash"):
    
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            max_output_tokens=1000,
            google_api_key=EnvConfig.GEMINI_API_KEY
        )
    

    @staticmethod
    def get_google_embeddings(model_name: str = "textembedding-gecko@001"):
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=EnvConfig.GEMINI_API_KEY
        )
    

    @staticmethod 
    def get_transformer_embeddings(model_name: str = "all-MiniLM-L6-v2", cache_dir: str = EnvConfig.CACHE_DIR):
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder = cache_dir, 
            model_kwargs={"device": "cpu"},
        )
    


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", type: str = "transformer", cache_dir: str = EnvConfig.CACHE_DIR):
    
    if type == "openai":
        return ModelFactory.get_openai_embeddings(model_name)
    elif type == "google":
        return ModelFactory.get_google_embeddings(model_name)
    elif type == "transformer":
        return ModelFactory.get_transformer_embeddings(model_name, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported embedding model type: {type}")