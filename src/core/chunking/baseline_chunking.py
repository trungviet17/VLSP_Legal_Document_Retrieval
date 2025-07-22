import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from typing import List, Dict, Any
from pyvi import ViTokenizer
from tqdm import tqdm
from utils.model import get_embedding_model

class BaseChunker: 

    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", embedding_type: str = "transformer", embedding_cache_dir: str = None,) :
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap 

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
        for doc, meta in tqdm(zip(corpus, metadata), total = len(corpus), desc="Processing documents"):
            chunks, vectors = self._chunk_(doc)  
            for chunk, vector in zip(chunks, vectors):
                processed.append({
                    "text": chunk,
                    "metadata": meta,
                    "vector": vector
                })
        return processed
      


    def _chunk_(self, text: str) -> list[str]:

        tokenized_text = ViTokenizer.tokenize(text)
        tokens = tokenized_text.split() 

        chunks = []
        start = 0 

        while start < len(tokens): 
            end = min(start + self.max_tokens, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start = end - self.chunk_overlap 
            if start < 0: 
                start = 0 

        vectors = []
        for chunk in chunks:
            vector = self.embedding_model.embed_documents(chunk)
            vectors.append(vector)

        return chunks, vectors





if __name__ == "__main__": 


    text = "1. Ngân hàng thương mại có đủ năng lực thực hiện bảo lãnh nhà ở hình thành trong tương lai khi: \t\ta) Trong giấy phép thành lập và hoạt động hoặc tại văn bản sửa đổi, bổ sung giấy phép thành lập và hoạt động của ngân hàng thương mại có quy định nội dung hoạt động bảo lãnh ngân hàng;\t\tb) Không bị cấm, hạn chế, đình chỉ, tạm đình chỉ thực hiện bảo lãnh nhà ở hình thành trong tương lai.\n\n2. Ngân hàng Nhà nước công bố công khai danh sách ngân hàng thương mại có đủ năng lực thực hiện bảo lãnh nhà ở hình thành trong tương lai trong từng thời kỳ trên Cổng thông tin điện tử của Ngân hàng Nhà nước. \n\n3. Ngân hàng thương mại xem xét, quyết định cấp bảo lãnh cho chủ đầu tư khi:\t\ta) Chủ đầu tư đáp ứng đủ các yêu cầu quy định tại Điều 11 Thông tư này (trừ trường hợp ngân hàng thương mại bảo lãnh cho chủ đầu tư trên cơ sở bảo lãnh đối ứng);\t\tb) Dự án của chủ đầu tư đáp ứng đủ các điều kiện của bất động sản hình thành trong tương lai được đưa vào kinh doanh theo quy định tại Điều 55 Luật Kinh doanh bất động sản và quy định của pháp luật có liên quan.\n\n4. Trình tự thực hiện bảo lãnh nhà ở hình thành trong tương lai: \t\ta) Căn cứ đề nghị của chủ đầu tư hoặc bên bảo lãnh đối ứng, ngân hàng thương mại xem xét, thẩm định và quyết định cấp bảo lãnh cho chủ đầu tư;\t\tb) Ngân hàng thương mại và chủ đầu tư ký hợp đồng bảo lãnh nhà ở hình thành trong tương lai theo quy định tại Điều 56 Luật Kinh doanh bất động sản và quy định tại khoản 13 Điều 3, Điều 15 Thông tư này;\t\tc) Sau khi ký hợp đồng mua, thuê mua nhà ở, trong đó có quy định nghĩa vụ tài chính của chủ đầu tư, chủ đầu tư gửi hợp đồng mua, thuê mua nhà ở cho ngân hàng thương mại để đề nghị ngân hàng thương mại phát hành thư bảo lãnh cho bên mua; \t\td) Ngân hàng thương mại căn cứ hợp đồng mua, thuê mua nhà ở và hợp đồng bảo lãnh nhà ở hình thành trong tương lai để phát hành thư bảo lãnh và gửi cho từng bên mua hoặc gửi chủ đầu tư để cung cấp thư bảo lãnh cho bên mua theo thỏa thuận.\n\n5. Thời hạn hiệu lực và nội dung của hợp đồng bảo lãnh nhà ở hình thành trong tương lai:\t\ta) Hợp đồng bảo lãnh nhà ở hình thành trong tương lai có hiệu lực kể từ thời điểm ký cho đến khi nghĩa vụ bảo lãnh của toàn bộ các thư bảo lãnh cho bên mua hết hiệu lực theo quy định tại Điều 23 Thông tư này và mọi nghĩa vụ của chủ đầu tư đối với ngân hàng thương mại theo hợp đồng bảo lãnh nhà ở hình thành trong tương lai đã hoàn thành;\t\tb) Ngoài các nội dung theo quy định tại khoản 2 Điều 15 Thông tư này (trừ nội dung tại điểm h và điểm i trong trường hợp bảo lãnh trên cơ sở bảo lãnh đối ứng), hợp đồng bảo lãnh nhà ở hình thành trong tương lai còn phải có các nội dung sau:(\t\ti) Ngân hàng thương mại có nghĩa vụ phát hành thư bảo lãnh cho bên mua khi nhận được hợp đồng mua, thuê mua nhà ở do chủ đầu tư gửi đến trước thời hạn giao, nhận nhà theo cam kết quy định tại hợp đồng mua, thuê mua nhà ở;(i\t\ti) Ngân hàng thương mại và chủ đầu tư thỏa thuận cụ thể về việc ngân hàng thương mại hoặc chủ đầu tư có nghĩa vụ gửi thư bảo lãnh cho bên mua sau khi ngân hàng thương mại phát hành thư bảo lãnh;(ii\t\ti) Nghĩa vụ tài chính của chủ đầu tư;(i\t\tv) Hồ sơ bên mua gửi cho ngân hàng thương mại yêu cầu thực hiện nghĩa vụ bảo lãnh phải kèm theo thư bảo lãnh do ngân hàng thương mại phát hành cho bên mua. \n\n6. Thời hạn hiệu lực và nội dung của thư bảo lãnh:\t\ta) Thư bảo lãnh có hiệu lực kể từ thời điểm phát hành cho đến thời điểm ít nhất sau 30 ngày kể từ thời hạn giao, nhận nhà theo cam kết tại hợp đồng mua, thuê mua nhà ở, trừ trường hợp nghĩa vụ bảo lãnh chấm dứt theo quy định tại Điều 23 Thông tư này. Trường hợp ngân hàng thương mại và chủ đầu tư chấm dứt hợp đồng bảo lãnh nhà ở hình thành trong tương lai trước thời hạn, các thư bảo lãnh đã phát hành cho các bên mua trước đó vẫn có hiệu lực cho đến khi nghĩa vụ bảo lãnh chấm dứt; \t\tb) Ngoài các nội dung theo quy định tại khoản 1 Điều 16 Thông tư này, thư bảo lãnh còn phải có nội dung nêu rõ nghĩa vụ tài chính của chủ đầu tư được ngân hàng thương mại bảo lãnh.\n\n7. Số tiền bảo lãnh cho một dự án nhà ở hình thành trong tương lai tối đa bằng tổng số tiền chủ đầu tư được phép nhận ứng trước của bên mua theo quy định tại Điều 57 Luật Kinh doanh bất động sản và các khoản tiền khác (nếu có) theo hợp đồng mua, thuê mua nhà ở.\n\n8. Số dư bảo lãnh trong bảo lãnh nhà ở hình thành trong tương lai:\t\ta) Số dư bảo lãnh đối với chủ đầu tư hoặc bên bảo lãnh đối ứng được xác định chính bằng số tiền thuộc nghĩa vụ tài chính của chủ đầu tư. Số dư bảo lãnh giảm dần khi nghĩa vụ bảo lãnh đối với bên mua chấm dứt theo quy định tại Điều 23 Thông tư này; \t\tb) Thời điểm ghi nhận số dư bảo lãnh là thời điểm chủ đầu tư thông báo với ngân hàng thương mại số tiền đã nhận ứng trước của các bên mua kể từ thời điểm thư bảo lãnh có hiệu lực quy định tại điểm c Khoản này;\t\tc) Ngân hàng thương mại và chủ đầu tư thỏa thuận về thời gian thông báo và cập nhật số tiền đã nhận ứng trước của các bên mua từ thời điểm thư bảo lãnh có hiệu lực trong tháng nhưng không muộn hơn ngày làm việc cuối cùng của tháng để làm cơ sở xác định số dư bảo lãnh. Chủ đầu tư chịu trách nhiệm trước pháp luật về việc thông báo chính xác số tiền và thời điểm đã nhận ứng trước của các bên mua cho ngân hàng thương mại.\n\n9. Ngân hàng thương mại có quyền và nghĩa vụ sau:\t\ta) Ngân hàng thương mại có quyền:(\t\ti) Từ chối phát hành thư bảo lãnh cho bên mua nếu hợp đồng mua, thuê mua nhà ở chưa phù hợp với quy định của pháp luật có liên quan hoặc sau khi đã chấm dứt hợp đồng bảo lãnh nhà ở hình thành trong tương lai với chủ đầu tư;(i\t\ti) Từ chối thực hiện nghĩa vụ bảo lãnh đối với số tiền không thuộc nghĩa vụ tài chính của chủ đầu tư hoặc số tiền bên mua nộp vượt quá tỷ lệ quy định tại Điều 57 Luật Kinh doanh bất động sản hoặc bên mua không xuất trình được thư bảo lãnh mà ngân hàng thương mại đã phát hành cho người thụ hưởng là bên mua.\t\tb) Ngân hàng thương mại có nghĩa vụ:(\t\ti) Phát hành thư bảo lãnh và gửi cho chủ đầu tư hoặc bên mua (theo thỏa thuận) khi nhận được hợp đồng mua, thuê mua nhà ở hợp lệ trước thời hạn giao, nhận nhà dự kiến quy định tại hợp đồng mua, thuê mua nhà ở;(i\t\ti) Trường hợp ngân hàng thương mại và chủ đầu tư chấm dứt hợp đồng bảo lãnh nhà ở hình thành trong tương lai trước thời hạn, chậm nhất vào ngày làm việc tiếp theo, ngân hàng thương mại phải thông báo công khai trên trang thông tin điện tử của ngân hàng thương mại và thông báo bằng văn bản cho cơ quan quản lý nhà ở cấp tỉnh thuộc địa bàn nơi có dự án nhà ở của chủ đầu tư, trong đó nêu rõ nội dung ngân hàng thương mại không tiếp tục phát hành thư bảo lãnh cho bên mua ký hợp đồng mua, thuê mua nhà ở với chủ đầu tư sau thời điểm ngân hàng thương mại chấm dứt hợp đồng bảo lãnh nhà ở hình thành trong tương lai với chủ đầu tư. Đối với các thư bảo lãnh đã phát hành cho bên mua trước đó, ngân hàng thương mại tiếp tục thực hiện cam kết cho đến khi nghĩa vụ bảo lãnh chấm dứt;(ii\t\ti) Thực hiện nghĩa vụ bảo lãnh với số tiền trả thay tương ứng với nghĩa vụ tài chính của chủ đầu tư được xác định căn cứ theo hồ sơ yêu cầu thực hiện nghĩa vụ bảo lãnh do bên mua cung cấp phù hợp với điều kiện thực hiện nghĩa vụ bảo lãnh quy định tại thư bảo lãnh.\n\n10. Chủ đầu tư có quyền và nghĩa vụ sau:\t\ta) Chủ đầu tư có quyền:Đề nghị ngân hàng thương mại phát hành thư bảo lãnh cho tất cả bên mua thuộc dự án nhà ở hình thành trong tương lai được ngân hàng bảo lãnh trong thời hạn hợp đồng bảo lãnh nhà ở hình thành trong tương lai có hiệu lực. \t\tb) Chủ đầu tư có nghĩa vụ:(\t\ti) Gửi thư bảo lãnh do ngân hàng thương mại phát hành cho bên mua sau khi nhận được từ ngân hàng thương mại (theo thỏa thuận);(i\t\ti) Trường hợp ngân hàng thương mại và chủ đầu tư chấm dứt hợp đồng bảo lãnh nhà ở hình thành trong tương lai trước thời hạn, chậm nhất vào ngày làm việc tiếp theo, chủ đầu tư phải thông báo công khai trên trang thông tin điện tử của chủ đầu tư (nếu có) và thông báo bằng văn bản cho cơ quan quản lý nhà ở cấp tỉnh thuộc địa bàn nơi có dự án nhà ở của chủ đầu tư;(ii\t\ti) Thông báo chính xác cho ngân hàng thương mại số tiền đã nhận ứng trước của từng bên mua kể từ thời điểm thư bảo lãnh có hiệu lực.\n\n11. Bên mua có quyền:\t\ta) Được nhận thư bảo lãnh do ngân hàng thương mại phát hành từ ngân hàng thương mại hoặc chủ đầu tư gửi đến trong thời hạn hợp đồng bảo lãnh nhà ở hình thành trong tương lai có hiệu lực và trước thời hạn giao, nhận nhà dự kiến quy định tại hợp đồng mua, thuê mua nhà ở; \t\tb) Yêu cầu ngân hàng thương mại thực hiện nghĩa vụ bảo lãnh đối với nghĩa vụ tài chính của chủ đầu tư trên cơ sở xuất trình thư bảo lãnh kèm theo hồ sơ phù hợp với thư bảo lãnh (nếu có).\n\n12. Ngoài các quy định tại Điều này, các nội dung khác về việc bảo lãnh nhà ở hình thành trong tương lai thực hiện theo quy định tương ứng tại Thông tư này."

    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split() 

    print(f"Total tokens: {len(tokens)}")


    chunker = BaseChunker(max_tokens=512, chunk_overlap=50)

    chunks = chunker._chunk_(text)


    for i, chunk in enumerate(chunks):
        
        print(f"Chunk {i+1} with len {len(chunk)}: {chunk}\n")
     

    
