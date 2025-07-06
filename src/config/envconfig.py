from dotenv import load_dotenv
from dataclasses import dataclass 
import os 

load_dotenv()

@dataclass
class EnvConfig: 

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    CACHE_DIR = os.getenv("CACHE_DIR", default="./cache_dir")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")