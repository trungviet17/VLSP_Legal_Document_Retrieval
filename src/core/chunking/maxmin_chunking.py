import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from typing import List, Dict, Any
from pyvi import ViTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.model import get_embedding_model
from core.chunking.baseline_chunking import BaseChunker
from config.envconfig import EnvConfig
from nltk.tokenize import sent_tokenize



class MaxMinChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_type: str = "transformer", embedding_cache_dir: str = None,
                 fixed_threshold=0.75, c=0.9, init_constant=1.5):
        super().__init__(max_tokens=max_tokens, chunk_overlap=chunk_overlap)

        self.fixed_threshold = fixed_threshold
        self.c = c

        self.init_constant = init_constant
        if embedding_cache_dir is None:
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
    

    def process_sentences(sentences, embeddings, fixed_threshold=0.6, c=0.9, init_constant=1.5):
        
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

        # Tokenize the text into sentences
        try:
            sentences = sent_tokenize(text, language='vietnamese')
        except Exception:
            sentences = text.split('\n\n')
            sentences = [s.strip() for s in sentences if s.strip()]

        embeddings = self.embedding_model.embed_documents(sentences)
        
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


    text = "1.\r\nTrong thời hạn 10 ngày kể từ ngày phát hiện hành vi vi phạm, người có thẩm\r\nquyền xử phạt của cơ quan Công an nơi phát hiện vi phạm hành chính thực hiện\r\nnhư sau:a)\r\nXác định thông tin về phương tiện giao thông, chủ phương tiện, tổ chức, cá nhân\r\ncó liên quan đến vi phạm hành chính thông qua cơ quan đăng ký xe, Cơ sở dữ liệu\r\nQuốc gia về dân cư, cơ quan, tổ chức khác có liên quan;b)\r\nTrường hợp chủ phương tiện, tổ chức, cá nhân có liên quan đến vi phạm hành\r\nchính không cư trú, đóng trụ sở tại địa bàn cấp huyện nơi cơ quan Công an đã\r\nphát hiện vi phạm hành chính, nếu xác định vi phạm hành chính đó thuộc thẩm\r\nquyền xử phạt của Trưởng Công an xã, phường, thị trấn thì chuyển kết quả thu\r\nthập được bằng phương tiện, thiết bị kỹ thuật nghiệp vụ đến Công an xã, phường,\r\nthị trấn nơi chủ phương tiện, tổ chức, cá nhân có liên quan đến vi phạm hành\r\nchính cư trú, đóng trụ sở (theo mẫu số 03 ban hành\r\nkèm theo Thông tư này) để giải quyết, xử lý vụ việc vi phạm (khi được trang bị\r\nhệ thống mạng kết nối gửi bằng phương thức điện tử).Trường\r\nhợp vi phạm hành chính không thuộc thẩm quyền xử phạt của Trưởng Công an xã,\r\nphường, thị trấn hoặc thuộc thẩm quyền xử phạt của Trưởng Công an xã, phường,\r\nthị trấn nhưng Công an xã, phường, thị trấn chưa được trang bị hệ thống mạng\r\nkết nối thì chuyển kết quả thu thập được bằng phương tiện, thiết bị kỹ thuật\r\nnghiệp vụ đến Công an cấp huyện nơi chủ phương tiện, tổ chức, cá nhân có liên\r\nquan đến vi phạm hành chính cư trú, đóng trụ sở (theo mẫu\r\nsố 03 ban hành kèm theo Thông tư này) để giải quyết, xử lý vụ việc vi phạm;c)\r\nGửi thông báo (theo mẫu số 02 ban hành kèm theo Thông\r\ntư này) yêu cầu chủ phương tiện, tổ chức, cá nhân có liên quan đến vi phạm hành\r\nchính đến trụ sở cơ quan Công an nơi phát hiện vi phạm hành chính hoặc đến trụ\r\nsở Công an xã, phường, thị trấn, Công an cấp huyện nơi cư trú, đóng trụ sở để\r\ngiải quyết vụ việc vi phạm hành chính nếu việc đi lại gặp khó khăn và không có điều\r\nkiện trực tiếp đến trụ sở cơ quan Công an nơi phát hiện vi phạm hành chính theo\r\nquy định tại khoản 2 Điều 15 Nghị định số 135/2021/NĐ-CP.\r\nViệc gửi thông báo vi phạm được thực hiện bằng văn bản giấy hoặc bằng phương\r\nthức điện tử (khi\r\nđáp ứng điều kiện về cơ sở hạ tầng, kỹ thuật, thông tin).\n\n2.\r\nKhi chủ phương tiện, tổ chức, cá nhân có liên quan đến vi phạm hành chính đến\r\ncơ quan Công an để giải quyết vụ việc vi phạm thì người có thẩm quyền xử phạt\r\nvi phạm hành chính của cơ quan Công an nơi phát hiện vi phạm hoặc Trưởng Công\r\nan xã, phường, thị trấn, Trưởng Công an cấp huyện tiến hành giải quyết, xử lý\r\nvụ việc vi phạm theo quy định tại điểm c, điểm d khoản 1 Điều 15\r\nNghị định số 135/2021/NĐ-CP.\n\n3.\r\nTrường hợp vụ việc vi phạm do Công an xã, phường, thị trấn, Công an cấp huyện\r\ngiải quyết, xử lý thì phải thông báo ngay (trên Hệ thống cơ sở dữ liệu xử lý vi\r\nphạm hành chính) kết quả giải quyết, xử lý vụ việc cho cơ quan Công an nơi phát\r\nhiện vi phạm. Đồng thời, cập nhật trạng thái đã giải quyết, xử lý vụ việc trên\r\nTrang thông tin điện tử của Cục Cảnh sát giao thông và gửi ngay thông báo kết\r\nthúc cảnh báo phương tiện giao thông vi phạm cho cơ quan Đăng kiểm, gỡ bỏ trạng\r\nthái đã gửi thông báo cảnh báo cho cơ quan Đăng kiểm trên Hệ thống cơ sở dữ\r\nliệu xử lý vi phạm hành chính (nếu đã có thông tin cảnh báo từ cơ quan Công an\r\nnơi phát hiện vi phạm đối với vụ việc quy định tại khoản 5 Điều này).\n\n4.\r\nTrường hợp vụ việc vi phạm do cơ quan Công an nơi phát hiện vi phạm giải quyết,\r\nxử lý thì phải thông báo ngay (trên Hệ thống cơ sở dữ liệu xử lý vi phạm hành\r\nchính) kết quả giải quyết vụ việc cho Công an xã, phường, thị trấn hoặc Công an\r\ncấp huyện đã nhận kết quả thu thập được bằng phương tiện, thiết bị kỹ thuật\r\nnghiệp vụ. Đồng thời, cập nhật trạng thái đã giải quyết, xử lý vụ việc trên\r\nTrang thông tin điện tử của Cục Cảnh sát giao thông và gửi ngay thông báo kết\r\nthúc cảnh báo phương tiện giao thông vi phạm cho cơ quan Đăng kiểm, gỡ bỏ trạng\r\nthái đã gửi thông báo cảnh báo cho cơ quan Đăng kiểm trên Hệ thống cơ sở dữ\r\nliệu xử lý vi phạm hành chính đối với vụ việc quy định tại khoản 5 Điều này.\n\n5.\r\nQuá thời hạn 20 ngày, kể từ ngày gửi thông báo vi phạm, chủ phương tiện, tổ\r\nchức, cá nhân có liên quan đến vi phạm hành chính không đến trụ sở cơ quan Công\r\nan nơi phát hiện vi phạm để giải quyết vụ việc hoặc cơ quan Công an nơi phát\r\nhiện vi phạm chưa nhận được thông báo kết quả giải quyết, xử lý vụ việc của\r\nCông an xã, phường, thị trấn, Công an cấp huyện đã nhận kết quả thu thập được\r\nbằng phương tiện, thiết bị kỹ thuật nghiệp vụ thì người có thẩm quyền xử phạt\r\nvi phạm hành chính của cơ quan Công an nơi phát hiện vi phạm thực hiện như sau:a)\r\nCập nhật thông tin của phương tiện giao thông vi phạm (loại phương tiện; biển\r\nsố, màu biển số; thời gian, địa điểm vi phạm, hành vi vi phạm; đơn vị phát hiện\r\nvi phạm; đơn vị giải quyết vụ việc, số điện thoại liên hệ) trên Trang thông tin\r\nđiện tử của Cục Cảnh sát giao thông để chủ phương tiện, tổ chức, cá nhân có\r\nliên quan đến vi phạm hành chính biết, liên hệ giải quyết theo quy định;b)\r\nGửi thông báo cảnh báo phương tiện giao thông vi phạm cho cơ quan Đăng kiểm\r\n(đối với phương tiện giao thông có quy định phải kiểm định); cập nhật trạng\r\nthái đã gửi thông báo cảnh báo cho cơ quan Đăng kiểm trên Hệ thống cơ sở dữ\r\nliệu xử lý vi phạm hành chính. Đối với phương tiện giao thông là xe mô tô, xe\r\ngắn máy, xe máy điện, tiếp tục gửi thông báo đến Công an xã, phường, thị trấn\r\nnơi chủ phương tiện, tổ chức, cá nhân có liên quan đến vi phạm hành chính cư\r\ntrú, đóng trụ sở (theo mẫu số 04 ban hành kèm theo\r\nThông tư này). Công an xã, phường, thị trấn có trách nhiệm chuyển thông báo đến\r\ncho chủ phương tiện, tổ chức, cá nhân có liên quan đến vi phạm hành chính và yêu\r\ncầu họ thực hiện theo thông báo vi phạm; kết quả làm việc, thông báo lại cho cơ\r\nquan Công an đã ra thông báo vi phạm (theo mẫu số 04\r\nban hành kèm theo Thông tư này).\n\n6. Việc chuyển kết\r\nquả thu thập được bằng phương tiện, thiết bị kỹ thuật nghiệp vụ, thông báo kết\r\nquả giải quyết vụ việc vi phạm được thực hiện bằng phương thức điện tử."

    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split() 

    print(f"Total tokens: {len(tokens)}")


    chunker = MaxMinChunker(max_tokens=512, chunk_overlap=50) # Cần thiết lập ngưỡng để chia đoạn phù hợp

    chunks = chunker._chunk_(text)


    for i, chunk in enumerate(chunks):
        
        print(f"Chunk {i+1} with len {len(chunk)}: {chunk}\n")
        print("=" * 50 + "\n")