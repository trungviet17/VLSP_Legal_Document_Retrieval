import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

import bisect
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pyvi import ViTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.model import get_embedding_model
from src.core.chunking.baseline_chunking import BaseChunker

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from transformers import AutoTokenizer


CHUNKING_STRATEGIES = ['semantic', 'fixed', 'sentences']

class LateChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_type: str = "transformer", embedding_cache_dir: str = None, chunking_strategy: str = 'fixed'):
        super().__init__(max_tokens=max_tokens, chunk_overlap=chunk_overlap)
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError("Unsupported chunking strategy: ", chunking_strategy)
        self.chunking_strategy = chunking_strategy
        if embedding_cache_dir is None:
            from config.envconfig import EnvConfig
            embedding_cache_dir = EnvConfig.CACHE_DIR

        self.embedding_model = get_embedding_model(
            model_name=embedding_model_name,
            type=embedding_type,
            cache_dir=embedding_cache_dir
        )

        # Use a fast tokenizer that supports offset mapping
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
        except:
            # Fallback to a tokenizer that supports offset mapping
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", use_fast=True)
            print("Warning: Using fallback tokenizer. Consider using a Vietnamese tokenizer that supports offset mapping.")

        # Initialize semantic splitter when needed
        if chunking_strategy == 'semantic':
            self.splitter = SemanticSplitterNodeParser(
                embed_model=self.embedding_model,
                show_progress=False,
            )
    
    def process_corpus(self, corpus, metadata):
        return super().process_corpus(corpus, metadata)

    def get_text_chunks_from_tokens(self, text: str, token_spans: List[Tuple[int, int]]) -> List[str]:
        """
        Convert token spans to text chunks using alternative method when offset mapping is not available
        """
        try:
            # Try to use offset mapping if available
            tokens = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            token_offsets = tokens.offset_mapping
            
            text_chunks = []
            for start_idx, end_idx in token_spans:
                if start_idx < len(token_offsets) and end_idx <= len(token_offsets):
                    char_start = token_offsets[start_idx][0]
                    char_end = token_offsets[end_idx - 1][1]
                    chunk_text = text[char_start:char_end]
                    text_chunks.append(chunk_text)
            return text_chunks
            
        except NotImplementedError:
            # Fallback method: decode tokens directly
            tokens = self.tokenizer(text, add_special_tokens=False)
            input_ids = tokens.input_ids
            
            text_chunks = []
            for start_idx, end_idx in token_spans:
                if start_idx < len(input_ids) and end_idx <= len(input_ids):
                    chunk_tokens = input_ids[start_idx:end_idx]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    text_chunks.append(chunk_text)
            return text_chunks

    def chunk_semantically(
        self,
        text: str,
    ) -> List[Tuple[int, int]]:
        # Initialize splitter if not already done
        if not hasattr(self, 'splitter'):
            self.splitter = SemanticSplitterNodeParser(
                embed_model=self.embedding_model,
                show_progress=False,
            )
            
        # Get semantic nodes
        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        ]

        try:
            # Try to use offset mapping if available
            tokens = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            token_offsets = tokens.offset_mapping

            chunk_spans = []
            for char_start, char_end in nodes:
                # Convert char indices to token indices
                start_chunk_index = bisect.bisect_left(
                    [offset[0] for offset in token_offsets], char_start
                )
                end_chunk_index = bisect.bisect_right(
                    [offset[1] for offset in token_offsets], char_end
                )

                if start_chunk_index < len(token_offsets) and end_chunk_index <= len(token_offsets):
                    chunk_spans.append((start_chunk_index, end_chunk_index))
                else:
                    break

            return chunk_spans
            
        except NotImplementedError:
            # Fallback: use character-based chunking with approximate token conversion
            tokens = self.tokenizer(text, add_special_tokens=False)
            total_tokens = len(tokens.input_ids)
            text_length = len(text)
            
            chunk_spans = []
            for char_start, char_end in nodes:
                # Approximate token positions based on character positions
                start_token = int((char_start / text_length) * total_tokens)
                end_token = int((char_end / text_length) * total_tokens)
                
                # Ensure bounds are valid
                start_token = max(0, min(start_token, total_tokens - 1))
                end_token = max(start_token + 1, min(end_token, total_tokens))
                
                chunk_spans.append((start_token, end_token))
            
            return chunk_spans
    
    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: int,
    ) -> List[Tuple[int, int]]:
        try:
            tokens = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            token_offsets = tokens.offset_mapping
        except NotImplementedError:
            # Fallback: just use token count
            tokens = self.tokenizer(text, add_special_tokens=False)
            token_offsets = None

        # Use input_ids length for chunking
        total_tokens = len(tokens.input_ids)
        chunk_spans = []
        
        for i in range(0, total_tokens, chunk_size):
            chunk_end = min(i + chunk_size, total_tokens)
            if chunk_end - i > 0:
                chunk_spans.append((i, chunk_end))

        return chunk_spans
    
    def chunk_by_sentences(
        self,
        text: str,
        n_sentences: int,
    ) -> List[Tuple[int, int]]:
        try:
            tokens = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            token_offsets = tokens.offset_mapping
        except NotImplementedError:
            tokens = self.tokenizer(text, add_special_tokens=False)
            token_offsets = None
            
        input_ids = tokens.input_ids

        chunk_spans = []
        chunk_start = 0
        count_chunks = 0
        
        for i in range(len(input_ids)):
            # Decode token to check for sentence boundaries
            token_text = self.tokenizer.decode([input_ids[i]], skip_special_tokens=True)
            
            # Check for sentence boundaries (double newlines or end of text)
            if '\n\n' in token_text and (
                (len(input_ids) == i + 1) or 
                (token_offsets is None or i + 1 < len(token_offsets))
            ):
                count_chunks += 1
                if count_chunks == n_sentences:
                    chunk_spans.append((chunk_start, i + 1))
                    chunk_start = i + 1
                    count_chunks = 0
                    
        # Add remaining tokens as final chunk
        if len(input_ids) - chunk_start > 1:
            chunk_spans.append((chunk_start, len(input_ids)))
            
        return chunk_spans
    
    def chunk(
        self,
        text: str,
        chunking_strategy: str = None,
        chunk_size: Optional[int] = None,
        n_sentences: Optional[int] = None,
    ):
        chunking_strategy = chunking_strategy or self.chunking_strategy
        if chunking_strategy == "semantic":
            return self.chunk_semantically(text)
        elif chunking_strategy == "fixed":
            chunk_size = chunk_size or self.max_tokens
            if chunk_size < 4:
                raise ValueError("Chunk size must be >= 4.")
            return self.chunk_by_tokens(text, chunk_size)
        elif chunking_strategy == "sentences":
            n_sentences = n_sentences or 3
            return self.chunk_by_sentences(text, n_sentences)
        else:
            raise ValueError("Unsupported chunking strategy")
        
    def _chunk_(self, text):
        """
        Main chunking method that returns actual text chunks instead of token indices
        """
        chunk_spans = self.chunk(text, self.chunking_strategy, self.max_tokens, 3)
        
        # Convert token spans to text chunks using the helper method
        text_chunks = self.get_text_chunks_from_tokens(text, chunk_spans)
        
        return text_chunks


if __name__ == "__main__": 
    text = "1. Căn cứ quy định tại Luật các tổ chức tín dụng, Thông tư này và quy định của pháp luật có liên quan, tổ chức tín dụng ban hành quy định nội bộ về giao dịch tiền gửi tiết kiệm của tổ chức tín dụng phù hợp với mô hình quản lý, đặc điểm, điều kiện kinh doanh, đảm bảo việc thực hiện giao dịch tiền gửi tiết kiệm chính xác, an toàn tài sản cho người gửi tiền và an toàn hoạt động cho tổ chức tín dụng.\n\n2. Quy định nội bộ phải quy định rõ trách nhiệm và nghĩa vụ của từng bộ phận, cá nhân có liên quan đến việc thực hiện giao dịch tiền gửi tiết kiệm và phải bao gồm tối thiểu các quy định sau:\t\ta) Nhận tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận tiền, ghi sổ kế toán việc nhận tiền gửi tiết kiệm; điền đầy đủ các nội dung quy định tại khoản 2 Điều 7 vào Thẻ tiết kiệm; giao Thẻ tiết kiệm cho người gửi tiền;\t\tb) Chi trả tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận Thẻ tiết kiệm; ghi sổ kế toán; chi trả gốc, lãi tiền gửi tiết kiệm;\t\tc) Sử dụng tiền gửi tiết kiệm làm tài sản bảo đảm;\t\td) Chuyển giao quyền sở hữu tiền gửi tiết kiệm;\t\tđ) Xử lý các trường hợp rủi ro theo quy định tại Điều 16 Thông tư này;\t\te) Thiết kế, in ấn, nhập xuất, bảo quản, kiểm kê, quản lý Thẻ tiết kiệm;\t\tg) Biện pháp để người gửi tiền tra cứu khoản tiền gửi tiết kiệm và biện pháp tổ chức tín dụng thông báo cho người gửi tiền khi có thay đổi đối với khoản tiền gửi tiết kiệm theo quy định tại Điều 11 Thông tư này;\t\th) Nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử (áp dụng đối với tổ chức tín dụng thực hiện nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử).\n\n Chào tôi là Chốn liền, tôi là một người yêu thích công nghệ và đang tìm kiếm những giải pháp mới để cải thiện cuộc sống hàng ngày của mình. Tôi tin rằng công nghệ có thể giúp chúng ta tiết kiệm thời gian, nâng cao hiệu quả công việc và mang lại nhiều tiện ích cho cuộc sống. Tôi rất mong muốn được chia sẻ những ý tưởng và kinh nghiệm của mình với cộng đồng để cùng nhau phát triển và áp dụng công nghệ vào cuộc sống một cách hiệu quả nhất."

    # Use ViTokenizer for Vietnamese text preprocessing (optional)
    tokenized_text = ViTokenizer.tokenize(text)
    print(f"ViTokenizer result sample: {tokenized_text[:100]}...")
    
    # Count tokens using the actual tokenizer
    chunker = LateChunker(max_tokens=128, chunk_overlap=50, chunking_strategy='fixed')
    tokens = chunker.tokenizer(text, add_special_tokens=False)
    print(f"Total tokens (PhoBERT): {len(tokens.input_ids)}")

    # Get chunks
    chunks = chunker._chunk_(text)

    print(f"\nTotal chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        chunk_tokens = chunker.tokenizer(chunk, add_special_tokens=False)
        print(f"Chunk {i+1} (tokens: {len(chunk_tokens.input_ids)}): {chunk[:100]}...")
        print("=" * 50 + "\n")