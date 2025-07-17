import sys
import os
import re
import numpy as np
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from utils.model import get_embedding_model
from baseline_chunking import BaseChunker
from nltk import sent_tokenize


class ClusteringChunker(BaseChunker):
    def __init__(self, max_tokens: int=512, chunk_overlap: int=50,
                 model_name: str='truro7/vn-law-embedding', lambda_pos: float=0.5,
                 clustering_method: str='agglomerative', eps: float=0.5, min_samples: int=2):
        
        super().__init__(max_tokens, chunk_overlap)
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.model = get_embedding_model(model_name=model_name, type='transformer')
        self.lambda_pos = lambda_pos
        self.clustering_method = clustering_method
        self.eps = eps
        self.min_samples = min_samples

    def _chunk_(self, text: str) -> list[str]:
        
        sentences = re.split(r'(?:\n|\t|\n\t|\t\t)+', text.strip())
        
        n = len(sentences)
        if n == 0:
            return []
        
        embeddings = self.model.embed_documents(sentences)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                d_pos = abs(i - j)/n
                d_cos = 1 - cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                distance_matrix[i][j] = self.lambda_pos * d_pos + (1 - self.lambda_pos) * d_cos
        
        if self.clustering_method == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=self.eps, 
                linkage='average').fit(distance_matrix)
            
        elif self.clustering_method == 'dbscan':
            distance_matrix = np.clip(distance_matrix, 0, None)
            clustering = DBSCAN(
                eps=self.eps, 
                min_samples=self.min_samples, 
                metric='precomputed').fit(distance_matrix)
        else: 
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        labels = clustering.labels_
        cluster_map = {}
        for idx, label in enumerate(labels):
            if label == -1:
                label = f"noise_{idx}"
            cluster_map.setdefault(label, []).append(sentences[idx])
        
        # Split chunk if surpass max_tokens
        chunks = [''.join(group) for group in cluster_map.values()]
        final_chunks = []
       
        for i, chunk in enumerate(chunks):
            if len(chunk) > self.max_tokens:
                sub_chunks = super()._chunk_(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks
    

    def process_corpus(self, corpus, metadata):
        return super().process_corpus(corpus, metadata)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Clustering-based text chunking")
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens per chunk')
    parser.add_argument('--chunk_overlap', type=int, default=50, help='Number of overlapping tokens between chunks')
    parser.add_argument('--clustering_method', type=str, default='agglomerative', choices=['agglomerative', 'dbscan'], help='Clustering method to use')
    args = parser.parse_args()

    text = "1. Căn cứ quy định tại Luật các tổ chức tín dụng, Thông tư này và quy định của pháp luật có liên quan, tổ chức tín dụng ban hành quy định nội bộ về giao dịch tiền gửi tiết kiệm của tổ chức tín dụng phù hợp với mô hình quản lý, đặc điểm, điều kiện kinh doanh, đảm bảo việc thực hiện giao dịch tiền gửi tiết kiệm chính xác, an toàn tài sản cho người gửi tiền và an toàn hoạt động cho tổ chức tín dụng.\n\n" \
    "2. Quy định nội bộ phải quy định rõ trách nhiệm và nghĩa vụ của từng bộ phận, cá nhân có liên quan đến việc thực hiện giao dịch tiền gửi tiết kiệm và phải bao gồm tối thiểu các quy định sau:\t\t \
        a) Nhận tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận tiền, ghi sổ kế toán việc nhận tiền gửi tiết kiệm; điền đầy đủ các nội dung quy định tại khoản 2 Điều 7 vào Thẻ tiết kiệm; giao Thẻ tiết kiệm cho người gửi tiền;\t\t \
            b) Chi trả tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận Thẻ tiết kiệm; ghi sổ kế toán; chi trả gốc, lãi tiền gửi tiết kiệm;\t\t \
                c) Sử dụng tiền gửi tiết kiệm làm tài sản bảo đảm;\t\t \
                    d) Chuyển giao quyền sở hữu tiền gửi tiết kiệm;\t\t \
                        đ) Xử lý các trường hợp rủi ro theo quy định tại Điều 16 Thông tư này;\t\t " \
                        "e) Thiết kế, in ấn, nhập xuất, bảo quản, kiểm kê, quản lý Thẻ tiết kiệm;\t\t \
                            g) Biện pháp để người gửi tiền tra cứu khoản tiền gửi tiết kiệm và biện pháp tổ chức tín dụng thông báo cho người gửi tiền khi có thay đổi đối với khoản tiền gửi tiết kiệm theo quy định tại Điều 11 Thông tư này;\t\t \
                                h) Nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử (áp dụng đối với tổ chức tín dụng thực hiện nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử)"

    chunker = ClusteringChunker(max_tokens=args.max_tokens, chunk_overlap=args.chunk_overlap, clustering_method=args.clustering_method)
    chunks = chunker._chunk_(text)

    
    for i, chunk in enumerate(chunks):        
        print(f"Chunk {i+1} with len {len(chunk.split())}: {chunk}\n")