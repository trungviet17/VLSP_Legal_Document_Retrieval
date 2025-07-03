import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.envconfig import EnvConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class ModelFactory:

    @staticmethod 
    def get_openai_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=EnvConfig.OPENAI_API_KEY
        )
    

    @staticmethod
    def get_openai_embeddings(model_name: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=EnvConfig.OPENAI_API_KEY
        )


    @staticmethod
    def get_google_llm(model_name: str = "gemini-1.5-flash"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            max_output_tokens=1000,
            google_api_key=EnvConfig.GEMINI_API_KEY
        )
    

    @staticmethod
    def get_google_embeddings(model_name: str = "textembedding-gecko@001"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=EnvConfig.GEMINI_API_KEY
        )