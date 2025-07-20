from pathlib import Path
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field

CONFIGS = {
    # API Keys and Endpoints
    "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY", None),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", None),
    "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY", None),

    # SSE Protocol Settings
    "DEFAULT_ENCODING": "utf-8",
    "DEFAULT_ENCODING_ERROR_HANDLER": "strict",
    "DEFAULT_HTTP_TIMEOUT": 5,
    "DEFAULT_SSE_READ_TIMEOUT": 60 * 5,

}

class LLMSettings(BaseModel):
    model: str = Field(..., description="The name of the LLM model.")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(8192, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )



