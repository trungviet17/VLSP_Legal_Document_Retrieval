import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from typing import List, Dict, Any
from pyvi import ViTokenizer
from tqdm import tqdm


class BaseChunker: 

    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50) :
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap 

    def process_corpus(self, corpus: List[str], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]: 
        processed = []
        for doc, meta in tqdm(zip(corpus, metadata), total = len(corpus), desc="Processing documents"):
            chunks = self._chunk_(doc)  
            for chunk in chunks:
                processed.append({
                    "text": chunk,
                    "metadata": meta
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

        return chunks 





if __name__ == "__main__": 


    text = "1. Căn cứ quy định tại Luật các tổ chức tín dụng, Thông tư này và quy định của pháp luật có liên quan, tổ chức tín dụng ban hành quy định nội bộ về giao dịch tiền gửi tiết kiệm của tổ chức tín dụng phù hợp với mô hình quản lý, đặc điểm, điều kiện kinh doanh, đảm bảo việc thực hiện giao dịch tiền gửi tiết kiệm chính xác, an toàn tài sản cho người gửi tiền và an toàn hoạt động cho tổ chức tín dụng.\n\n2. Quy định nội bộ phải quy định rõ trách nhiệm và nghĩa vụ của từng bộ phận, cá nhân có liên quan đến việc thực hiện giao dịch tiền gửi tiết kiệm và phải bao gồm tối thiểu các quy định sau:\t\ta) Nhận tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận tiền, ghi sổ kế toán việc nhận tiền gửi tiết kiệm; điền đầy đủ các nội dung quy định tại khoản 2 Điều 7 vào Thẻ tiết kiệm; giao Thẻ tiết kiệm cho người gửi tiền;\t\tb) Chi trả tiền gửi tiết kiệm, trong đó phải có tối thiểu các nội dung: nhận Thẻ tiết kiệm; ghi sổ kế toán; chi trả gốc, lãi tiền gửi tiết kiệm;\t\tc) Sử dụng tiền gửi tiết kiệm làm tài sản bảo đảm;\t\td) Chuyển giao quyền sở hữu tiền gửi tiết kiệm;\t\tđ) Xử lý các trường hợp rủi ro theo quy định tại Điều 16 Thông tư này;\t\te) Thiết kế, in ấn, nhập xuất, bảo quản, kiểm kê, quản lý Thẻ tiết kiệm;\t\tg) Biện pháp để người gửi tiền tra cứu khoản tiền gửi tiết kiệm và biện pháp tổ chức tín dụng thông báo cho người gửi tiền khi có thay đổi đối với khoản tiền gửi tiết kiệm theo quy định tại Điều 11 Thông tư này;\t\th) Nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử (áp dụng đối với tổ chức tín dụng thực hiện nhận, chi trả tiền gửi tiết kiệm bằng phương tiện điện tử)"

    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split() 

    print(f"Total tokens: {len(tokens)}")


    chunker = BaseChunker(max_tokens=512, chunk_overlap=50)

    chunks = chunker._chunk_(text)


    for i, chunk in enumerate(chunks):
        
        print(f"Chunk {i+1} with len {len(chunk)}: {chunk}\n")
     

    
