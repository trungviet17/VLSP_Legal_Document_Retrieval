import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from pyvi import ViTokenizer
from tqdm import tqdm


class BaseChunker: 

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.max_tokens, 
            chunk_overlap = 0,
            separators= ["\n", ".", "!", "?", ";"]
        ) 

    def process_corpus(self, corpus: List[str], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]: 
        processed = []
        for doc, meta in tqdm(zip(corpus, metadata), total = len(corpus), desc="Processing documents"):
            chunks = self._chunk_(doc)
            chunks = [ViTokenizer.tokenize(chunk) for chunk in chunks]  
            for chunk in chunks:
                processed.append({
                    "text": chunk,
                    "metadata": meta
                })
        return processed
      


    def _chunk_(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)

    
