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

from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

class LateChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_type: str = "transformer", embedding_cache_dir: str = None):
        super().__init__(max_tokens=max_tokens, chunk_overlap=chunk_overlap)

        if embedding_cache_dir is None:
            from config.envconfig import EnvConfig
            embedding_cache_dir = EnvConfig.CACHE_DIR

        self.embedding_model = get_embedding_model(
            model_name=embedding_model_name,
            type=embedding_type,
            cache_dir=embedding_cache_dir
        )

        self.transformer_layer = self.embedding_model._first_module()
        self.pooling_layer = self.embedding_model._last_module()

    def process_corpus(self, corpus, metadata):
        return super().process_corpus(corpus, metadata)
        
    def _chunk_(self, text):
        """
        Main chunking method that returns actual text chunks instead of token indices
        """
        sentences = text.split('\n\n')
        sentences = [s.strip() for s in sentences if s.strip()]

        # Step 1: Tokenize the entire text
        tokens = self.embedding_model.tokenizer(text, return_tensors='pt', padding=False, truncation=False)

        # Step 2: Get token embeddings
        with torch.no_grad():
            outputs = self.transformer_layer({'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']})
            token_embeddings = outputs['token_embeddings']

        # Step 3: Use pooling layer for chunks
        sentence_embeddings = []
        current_token_idx = 1  # skip CLS token

        for chunk in sentences:
            chunk_tokens = self.embedding_model.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
            chunk_length = chunk_tokens['input_ids'].shape[1] - 2  # Remove CLS and SEP tokens
            
            chunk_embeddings = token_embeddings[:, current_token_idx:current_token_idx+chunk_length]
            chunk_attention_mask = chunk_tokens['attention_mask'][:, 1: -1] # Remove CLS and SEP tokens
            
            sentence_embedding = torch.mean(chunk_embeddings, dim=1)  # Mean pooling
            sentence_embedding = sentence_embedding.squeeze(0)  # Remove batch dimension
            
            # # Use pooling layer
            # features = {}
            # features['token_embeddings'] = chunk_embeddings # Add batch dimension
            # features['attention_mask'] = chunk_attention_mask
            # features['sentence_embedding'] = torch.mean(chunk_embeddings, dim=1)  # Mean pooling
            # sentence_embedding = pooling_layer(features)['sentence_embedding']
            # sentence_embedding = sentence_embedding.squeeze(0)  # Remove batch dimension

            sentence_embeddings.append(sentence_embedding)
            current_token_idx += chunk_length

        sentence_embeddings = torch.stack(sentence_embeddings)

        return sentences, sentence_embeddings


if __name__ == "__main__": 
    text = "1. Căn cứ quy định tại Luật các tổ chức tín dụng, Thông tư này và quy định của pháp luật có liên quan, tổ chức tín dụng ban hành quy định nội bộ về giao dịch tiền gửi tiết kiệm của tổ chức tín dụng phù hợp với mô hình quản lý, đặc điểm, điều kiện kinh doanh, đảm bảo việc thực hiện giao dịch tiền gửi tiết kiệm chính xác, an toàn tài sản cho người gửi tiền và an toàn hoạt động cho tổ chức tín dụng.\n\n2. Quy định nội bộ phải quy định rõ trách nhiệm và nghĩa vụ của từng bộ phận, cá nhân có liên quan đến việc thực hiện giao dịch tiền gửi tiết kiệm và phải bao gồm tối thiểu các quy định sau:\t\ta) Nhận tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận tiền, ghi sổ kế toán việc nhận tiền gửi tiết kiệm; điền đầy đủ các nội dung quy định tại khoản 2 Điều 7 vào Thẻ tiết kiệm; giao Thẻ tiết kiệm cho người gửi tiền;\t\tb) Chi trả tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận Thẻ tiết kiệm; ghi sổ kế toán; chi trả gốc, lãi tiền gửi tiết kiệm;\t\tc) Sử dụng tiền gửi tiết kiệm làm tài sản bảo đảm;\t\td) Chuyển giao quyền sở hữu tiền gửi tiết kiệm;\t\tđ) Xử lý các trường hợp rủi ro theo quy định tại Điều 16 Thông tư này;\t\te) Thiết kế, in ấn, nhập xuất, bảo quản, kiểm kê, quản lý Thẻ tiết kiệm;\t\tg) Biện pháp để người gửi tiền tra cứu khoản tiền gửi tiết kiệm và biện pháp tổ chức tín dụng thông báo cho người gửi tiền khi có thay đổi đối với khoản tiền gửi tiết kiệm theo quy định tại Điều 11 Thông tư này;\t\th) Nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử (áp dụng đối với tổ chức tín dụng thực hiện nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử).\n\n Chào tôi là Chốn liền, tôi là một người yêu thích công nghệ và đang tìm kiếm những giải pháp mới để cải thiện cuộc sống hàng ngày của mình. Tôi tin rằng công nghệ có thể giúp chúng ta tiết kiệm thời gian, nâng cao hiệu quả công việc và mang lại nhiều tiện ích cho cuộc sống. Tôi rất mong muốn được chia sẻ những ý tưởng và kinh nghiệm của mình với cộng đồng để cùng nhau phát triển và áp dụng công nghệ vào cuộc sống một cách hiệu quả nhất."

    # Use ViTokenizer for Vietnamese text preprocessing (optional)
    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split() 

    print(f"Total tokens: {len(tokens)}")
    
    # Count tokens using the actual tokenizer
    chunker = LateChunker(max_tokens=128, chunk_overlap=50)

    # Get chunks
    chunk_embeddings, chunks = chunker._chunk_(text)

    print(f"\nTotal chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        chunk_tokens = chunker.tokenizer(chunk, add_special_tokens=False)
        print(f"Chunk {i+1} (tokens: {len(chunk_tokens.input_ids)}): {chunk[:100]}...")
        print("=" * 50 + "\n")