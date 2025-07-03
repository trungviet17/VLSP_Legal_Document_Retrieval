from langchain_core.output_parsers import BaseOutputParser
import re 
import json 
from typing import Dict, Any


class CustomOutputParser(BaseOutputParser):

    def parse(self, text: str) -> Dict:

        try:

            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
            
            data = json.loads(text)

            return data 

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}") from e