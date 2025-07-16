import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from typing import List, Dict, Any
from pyvi import ViTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.model import get_embedding_model
from src.core.chunking.baseline_chunking import BaseChunker

class MaxMinChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_type: str = "transformer", embedding_cache_dir: str = None,
                 fixed_threshold=0.75, c=0.9, init_constant=1.5):
        super().__init__(max_tokens=max_tokens, chunk_overlap=chunk_overlap)
        self.fixed_threshold = fixed_threshold
        self.c = c
        self.init_constant = init_constant
        if embedding_cache_dir is None:
            from config.envconfig import EnvConfig
            embedding_cache_dir = EnvConfig.CACHE_DIR

        self.embedding_model = get_embedding_model(
            model_name=embedding_model_name,
            type=embedding_type,
            cache_dir=embedding_cache_dir
        )

    def process_corpus(self, corpus: List[str], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for doc, meta in tqdm(zip(corpus, metadata), total=len(corpus), desc="Processing documents"):
            chunks = self._chunk_(doc)
            for chunk in chunks:
                processed.append({
                    "text": chunk,
                    "metadata": meta
                })
        return processed
    
    @staticmethod
    def process_sentences(sentences, embeddings, fixed_threshold=0.6, c=0.9, init_constant=1.5):
        """
        Process sentences into paragraphs based on semantic similarity.

        Args:
        - sentences (list of str): List of sentences to process.
        - embeddings (np.array): Sentence embeddings of shape (n_sentences, embedding_dim).
        - fixed_threshold (float): Fixed similarity threshold for joining sentences.
        - c (float): Coefficient for adjusting the similarity threshold.
        - init_constant (float): Initial constant for similarity comparison when cluster size is 1.

        Returns:
        - list of list of str: List of paragraphs, where each paragraph is a list of sentences.
        """
        
        def sigmoid(x):
            """Sigmoid function for adjusting threshold based on cluster size."""
            return 1 / (1 + np.exp(-x))

        paragraphs = []
        current_paragraph = [sentences[0]]
        cluster_start, cluster_end = 0, 1
        pairwise_min = -float('inf')

        for i in range(1, len(sentences)):
            cluster_embeddings = embeddings[cluster_start:cluster_end]

            if cluster_end - cluster_start > 1:
                new_sentence_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]

                # Adjust threshold based on cluster size and similarity
                adjusted_threshold = pairwise_min * c * sigmoid((cluster_end - cluster_start) - 1)
                new_sentence_similarity = np.max(new_sentence_similarities)
                
                # Use the minimum of the minimum similarities and the pairwise_min
                pairwise_min = min(np.min(new_sentence_similarities), pairwise_min)
            else:
                adjusted_threshold = 0
                # Use an initial constant when there's only one sentence in the cluster
                pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
                new_sentence_similarity = init_constant * pairwise_min

            # Decide whether to add the sentence to the current paragraph or start a new one
            if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
                current_paragraph.append(sentences[i])
                cluster_end += 1
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [sentences[i]]
                cluster_start, cluster_end = i, i + 1
                pairwise_min = -float('inf')

        # Append the last paragraph
        paragraphs.append(current_paragraph)
        return paragraphs


    def _chunk_(self, text: str) -> list[str]:
        # Tách câu
        from nltk.tokenize import sent_tokenize
        try:
            sentences = sent_tokenize(text, language='vietnamese')
        except Exception:
            # fallback nếu không có tiếng Việt
            sentences = text.split('\n\n')
            sentences = [s.strip() for s in sentences if s.strip()]

        # Lấy embedding cho từng câu
        embeddings = self.embedding_model.embed_documents(sentences)
        # Gom nhóm các câu thành đoạn (paragraphs)
        paragraphs = self.process_sentences(
            sentences, np.array(embeddings),
            fixed_threshold=self.fixed_threshold,
            c=self.c,
            init_constant=self.init_constant
        )
        # Gộp các câu trong mỗi đoạn thành 1 chunk
        chunks = [' '.join(paragraph) for paragraph in paragraphs]
        # Cắt chunk nếu quá dài (dùng logic của BaseChunker)
        final_chunks = []
        for chunk in chunks:
            final_chunks.extend(super()._chunk_(chunk))
        return final_chunks
    

if __name__ == "__main__": 


    text = "1. Căn cứ quy định tại Luật các tổ chức tín dụng, Thông tư này và quy định của pháp luật có liên quan, tổ chức tín dụng ban hành quy định nội bộ về giao dịch tiền gửi tiết kiệm của tổ chức tín dụng phù hợp với mô hình quản lý, đặc điểm, điều kiện kinh doanh, đảm bảo việc thực hiện giao dịch tiền gửi tiết kiệm chính xác, an toàn tài sản cho người gửi tiền và an toàn hoạt động cho tổ chức tín dụng.\n\n2. Quy định nội bộ phải quy định rõ trách nhiệm và nghĩa vụ của từng bộ phận, cá nhân có liên quan đến việc thực hiện giao dịch tiền gửi tiết kiệm và phải bao gồm tối thiểu các quy định sau:\t\ta) Nhận tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận tiền, ghi sổ kế toán việc nhận tiền gửi tiết kiệm; điền đầy đủ các nội dung quy định tại khoản 2 Điều 7 vào Thẻ tiết kiệm; giao Thẻ tiết kiệm cho người gửi tiền;\t\tb) Chi trả tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận Thẻ tiết kiệm; ghi sổ kế toán; chi trả gốc, lãi tiền gửi tiết kiệm;\t\tc) Sử dụng tiền gửi tiết kiệm làm tài sản bảo đảm;\t\td) Chuyển giao quyền sở hữu tiền gửi tiết kiệm;\t\tđ) Xử lý các trường hợp rủi ro theo quy định tại Điều 16 Thông tư này;\t\te) Thiết kế, in ấn, nhập xuất, bảo quản, kiểm kê, quản lý Thẻ tiết kiệm;\t\tg) Biện pháp để người gửi tiền tra cứu khoản tiền gửi tiết kiệm và biện pháp tổ chức tín dụng thông báo cho người gửi tiền khi có thay đổi đối với khoản tiền gửi tiết kiệm theo quy định tại Điều 11 Thông tư này;\t\th) Nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử (áp dụng đối với tổ chức tín dụng thực hiện nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử).\n\n Chào tôi là Chốn liền, tôi là một người yêu thích công nghệ và đang tìm kiếm những giải pháp mới để cải thiện cuộc sống hàng ngày của mình. Tôi tin rằng công nghệ có thể giúp chúng ta tiết kiệm thời gian, nâng cao hiệu quả công việc và mang lại nhiều tiện ích cho cuộc sống. Tôi rất mong muốn được chia sẻ những ý tưởng và kinh nghiệm của mình với cộng đồng để cùng nhau phát triển và áp dụng công nghệ vào cuộc sống một cách hiệu quả nhất."

    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split() 

    print(f"Total tokens: {len(tokens)}")


    chunker = MaxMinChunker(max_tokens=512, chunk_overlap=50) # Cần thiết lập ngưỡng để chia đoạn phù hợp

    chunks = chunker._chunk_(text)


    for i, chunk in enumerate(chunks):
        
        print(f"Chunk {i+1} with len {len(chunk)}: {chunk}\n")
        print("=" * 50 + "\n")