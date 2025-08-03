import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import List
from pyvi import ViTokenizer


from core.chunking.baseline_chunking import BaseChunker
import re



class PuncChunker(BaseChunker): 

    def __init__(self, max_tokens: int = 512, chunk_overlap: int = 50, separator: str = "\n\n") -> None: 
        super().__init__(max_tokens, chunk_overlap)
        self.separator = separator


    def _split_long_chunk(self, chunk: str): 

        half_chunk_size = self.max_tokens // 2
        sentences = re.split(r'(\t|\.)', chunk)
        chunks = []
        start = 0

        while start < len(sentences):
            token_count = 0
            end = start
            while end < len(sentences) and token_count <= self.max_tokens + half_chunk_size:
                unit_length = len(sentences[end].split())
                if token_count + unit_length > self.max_tokens + half_chunk_size:
                    break
                token_count += unit_length
                end += 1
            if end == start:
                end = start + 1
            chunks.append(' '.join(sentences[start:end]))
            start = end

        return [c.strip() for c in chunks if c.strip()]


    def _chunk_(self, text: str) -> List[str]:

        tokenized_text = ViTokenizer.tokenize(text)

        chunks = []
        raw_chunks = tokenized_text.split(self.separator)
        raw_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
        half_chunk_size = self.max_tokens // 2

        idx = 0
        while idx < len(raw_chunks):
            if len(raw_chunks[idx].split()) >= half_chunk_size and len(raw_chunks[idx].split()) <= self.max_tokens + half_chunk_size:
                chunks.append(raw_chunks[idx])
                idx += 1
            elif len(raw_chunks[idx].split()) > self.max_tokens + half_chunk_size:
                # handle too long chunk 
                split_chunks = self._split_long_chunk(raw_chunks[idx])
                chunks.extend(split_chunks)
                idx += 1 

            else: 
                # handle merging with previous or next chunk
                prev_len = len(raw_chunks[idx - 1].split()) if idx > 0 else float('inf')
                next_len = len(raw_chunks[idx + 1].split()) if idx + 1 < len(raw_chunks) else float('inf')

                if prev_len <= next_len and idx > 0:
                    # Merge with previous chunk
                    merged = raw_chunks[idx - 1] + self.separator + raw_chunks[idx]
                    if len(merged.split()) > self.max_tokens + self.chunk_overlap:
                        # Split if merged chunk is too long
                        split_chunks = self._split_long_chunk(merged)
                        chunks.extend(split_chunks)
                    else:
                        # Replace previous chunk with merged
                        chunks[-1] = merged
                    idx += 1
                elif idx + 1 < len(raw_chunks):
                    # Merge with next chunk
                    merged = raw_chunks[idx] + self.separator + raw_chunks[idx + 1]
                    if len(merged.split()) > self.max_tokens + self.chunk_overlap:
                        split_chunks = self._split_long_chunk(merged)
                        chunks.extend(split_chunks)
                    else:
                        chunks.append(merged)
                    idx += 2
                else:
                    chunks.append(raw_chunks[idx])
                    idx += 1
                
        return [c.strip() for c in chunks if c.strip()]
               

    
if __name__ ==  "__main__": 


    text = """
    1. Xếp lương khi nâng ngạch công chức, viên chức :a. Trường hợp chưa hưởng phụ cấp thâm niên vượt khung ở ngạch cũ thì căn cứ vào hệ số lương đang hưởng ở ngạch cũ để xếp vào hệ số lương bằng hoặc cao hơn gần nhất ở ngạch mới. Thời gian hưởng lương ở ngạch mới được tính kể từ ngày ký quyết định bổ nhiệm vào ngạch mới. Thời gian xét nâng bậc lương lần sau ở ngạch mới được tính như sau: Nếu chênh lệch giữa hệ số lương được xếp ở ngạch mới so với hệ số lương đang hưởng ở ngạch cũ bằng hoặc lớn hơn chênh lệch hệ số lương giữa 2 bậc lương liền kề ở ngạch cũ, thì được tính kể từ ngày ký quyết định bổ nhiệm vào ngạch mới; nếu nhỏ hơn chênh lệch hệ số lương giữa 2 bậc lương liền kề ở ngạch cũ, thì được tính kể từ ngày xếp hệ số lương đang hưởng ở ngạch cũ.b. Trường hợp đang hưởng phụ cấp thâm niên vượt khung ở ngạch cũ, thì căn cứ vào tổng hệ số lương cộng phụ cấp thâm niên vượt khung đang hưởng ở ngạch cũ để xếp vào hệ số lương bằng hoặc cao hơn gần nhất ở ngạch mới. Thời gian hưởng lương ở ngạch mới và thời gian xét nâng bậc lương lần sau ở ngạch mới được tính kể từ ngày ký quyết định bổ nhiệm vào ngạch mới.Ví dụ 1: Bà Trần Thị A đang hưởng 6% phụ cấp thâm niên vượt khung ở ngạch chuyên viên (mã số 01.003) kể từ ngày 01 tháng 4 năm 2007 (tổng hệ số lương 4,98 cộng 6%VK đang hưởng ở ngạch chuyên viên là 5,28). Bà A đạt kỳ thi nâng ngạch chuyên viên chính và đến ngày 01 tháng 02 năm 2008 được cơ quan có thẩm quyền ký quyết định bổ nhiệm vào ngạch chuyên viên chính (mã số 01.002), thì bà A được căn cứ vào tổng hệ số lương đang hưởng ở ngạch chuyên viên là 5,28 này để xếp vào hệ số lương cao hơn gần nhất là 5,42 bậc 4 ngạch chuyên viên chính. Thời gian hưởng lương ở ngạch chuyên viên chính và thời gian xét nâng bậc lương lần sau ở ngạch chuyên viên chính của bà A được tính kể từ ngày 01 tháng 02 năm 2008 (ngày ký quyết định bổ nhiệm vào ngạch chuyên viên chính).c. Trường hợp có tổng hệ số lương cộng phụ cấp thâm niên vượt khung đang hưởng ở ngạch cũ lớn hơn hệ số lương ở bậc cuối cùng trong ngạch mới, thì xếp vào hệ số lương ở bậc cuối cùng trong ngạch mới và được hưởng thêm hệ số chênh lệch bảo lưu cho bằng tổng hệ số lương cộng phụ cấp thâm nhiên vượt khung đang hưởng ở ngạch cũ. Thời gian hưởng lương ở ngạch mới (kể cả hệ số chênh lệch bảo lưu) và thời gian xét hưởng phụ cấp thâm niên vượt khung ở ngạch mới được tính kể từ ngày ký quyết định bổ nhiệm vào ngạch mới.Hệ số chênh lệch bảo lưu tại điểm c này (tính tròn số sau dấu phẩy 2 số) được hưởng trong suốt thời gian cán bộ, công chức, viên chức xếp lương ở ngạch mới. Sau đó, nếu cán bộ, công chức, viên chức tiếp tục được nâng ngạch hoặc chuyển ngạch khác, thì được cộng hệ số chênh lệch bảo lưu này vào hệ số lương (kể cả phụ cấp thâm nhiên vượt khung, nếu có) đang hưởng để xếp lương vào ngạch được bổ nhiệm khi nâng ngạch hoặc chuyển ngạch và thôi hưởng hệ số chênh lệch bảo lưu kể từ ngày hưởng lương ở ngạch mới.Ví dụ 2: Ông Nguyễn Văn B đang hưởng 15% phụ cấp thâm niên vượt khung ở ngạch kiểm ngân viên (mã số 07.047) kể từ ngày 01 tháng 02 năm 2007 (tổng hệ số lương 3,63 cộng 15%VK đang hưởng ở ngạch kiểm ngân viên là 4,17). Đến ngày 01 tháng 10 năm 2007, ông B đủ điều kiện và được cơ quan có thẩm quyền quyết định nâng lên ngạch cán sự (mã số 01.004). Do tổng hệ số lương 4,17 đang hưởng ở ngạch kiểm ngân viên lớn hơn hệ số lương 4,06 ở bậc cuối cùng trong ngạch cán sự, nên ông B được xếp vào hệ số lương 4,06 bậc 12 ngạch cán sự và được hưởng thêm hệ số chênh lệch bảo lưu 0,11 (4,17 - 4,06) kể từ ngày 01 tháng 10 năm 2007 (ngày bổ nhiệm vào ngạch cán sự). Đến ngày 01 tháng 10 năm 2009, sau đủ 2 năm và có đủ điều kiện, ông B được hưởng 5% phụ cấp thâm niên vượt khung ở ngạch cán sự và vẫn tiếp tục được hưởng hệ số chênh lệch bảo lưu 0,11.Đến ngày 01 tháng 3 năm 2010, ông B đủ điều kiện và được cơ quan có thẩm quyền quyết định nâng lên ngạch chuyên viên (mã số 01.003) thì ông B được căn cứ vào tổng hệ số lương cộng hệ số chênh lệch bảo lưu và 5% phụ cấp thâm niên vượt khung đang hưởng ở ngạch cán sự là 4,37 (4,06 + 0,11 + 5%VK của 4,06) để xếp vào hệ số lương cao hơn gần nhất ở ngạch chuyên viên là 4,65 bậc 8 và thôi hưởng hệ số chênh lệch bảo lưu 0,11 kể từ ngày 01 tháng 3 năm 2010 (ông B đang hưởng phụ cấp thâm niên vượt khung ở ngạch cán sự nên thời gian hưởng lương ở ngạch chuyên viên và thời gian xét nâng bậc lương lần sau ở ngạch chuyên viên được tính kể từ ngày ký quyết định bổ nhiệm vào ngạch chuyên viên).\n\n2. Xếp lương khi chuyển ngạch trong cùng loại công chức, viên chức:a. Trường hợp được bổ nhiệm vào ngạch mới trong cùng nhóm ngạch với ngạch cũ (ngạch cũ và ngạch mới có cùng hệ số bậc lương), thì xếp ngang bậc lương và % phụ cấp thâm niên vượt khung (nếu có) đang hưởng ở ngạch cũ (kể cả tính thời gian xét nâng bậc lương lần sau hoặc xét hưởng phụ cấp thâm niên vượt khung nếu có ở ngạch cũ) sang ngạch mới.b. Trường hợp được bổ nhiệm vào ngạch mới có hệ số lương cùng bậc cao hơn ngạch cũ (ví dụ từ ngạch thuộc A2.2 sang ngạch thuộc A2.1), thì thực hiện như cách xếp lương khi nâng ngạch công chức, viên chức hướng dẫn tại Khoản 1 mục II Thông tư này.c. Trường hợp được bổ nhiệm vào ngạch mới có hệ số lương cùng bậc thấp hơn ngạch cũ (ví dụ từ ngạch thuộc A2.1 sang ngạch thuộc A2.2), thì thực hiện như cách xếp lương hướng dẫn tại điểm a Khoản 2 này và được hưởng thêm hệ số chênh lệch bảo lưu cho bằng hệ số lương (kể cả phụ cấp thâm nhiên vượt khung, nếu có) đang hưởng ở ngạch cũ. Hệ số chênh lệch bảo lưu này được thực hiện như hướng dẫn tại điểm c Khoản 1 mục II Thông tư này.\n\n3. Xếp lương khi chuyển loại công chức, viên chức:Trường hợp công chức, viên chức đủ tiêu chuẩn và điều kiện được cấp có thẩm quyền quyết định chuyển loại công chức, viên chức từ loại A0 sang loại A1; từ loại B, loại C sang loại A (gồm A0 và A1) hoặc từ loại C sang loại B, thì thực hiện như cách xếp lương khi nâng ngạch công chức, viên chức hướng dẫn tại Khoản 1 mục II Thông tư này.\n\n4. Xếp lương đối với cán bộ, công chức, viên chức đang làm việc và đã có quyết định nâng ngạch, chuyển ngạch, thay đổi ngạch (do được bổ sung hoặc có thay đổi về phân loại công chức, viên chức) từ sau ngày có hướng dẫn chuyển xếp lương cũ sang lương mới theo Nghị định số 204/2004/NĐ-CP  (sau ngày 26 tháng 01 năm 2005) đến trước ngày Thông tư này có hiệu lực thi hành (trừ các trường hợp quy định tại các Khoản 6, 7, 8 và 10 Mục III Thông tư số 79/2005/TT-BNV  ngày 10 tháng 8 năm 2005 của Bộ Nội vụ và các trường hợp đang được hưởng bảo lưu hệ số phụ cấp chức vụ lãnh đạo):a. Nếu tính lại theo hướng dẫn tại Thông tư này mà được xếp bậc lương, tính thời gian xét nâng bậc lương lần sau hoặc xét hưởng phụ cấp thâm nhiên vượt khung (nếu có) ở ngạch mới có lợi hơn thì được điều chỉnh lại theo hướng dẫn tại Thông tư này.Riêng thời gian hưởng bậc lương mới (sau khi xếp lại lương theo quy định tại điểm a này) được tính thống nhất kể từ ngày ký quyết định xếp lại bậc lương và không tính truy lĩnh tiền lương, không tính đóng hưởng bảo hiểm xã hội, bảo hiểm y tế phần chênh lệch giữa kết quả chuyển xếp lại lương theo hướng dẫn tại Thông tư này so với quyết định của cơ quan có thẩm quyền từ sau ngày 26 tháng 01 năm 2005 đến trước ngày Thông tư này có hiệu lực thi hành.b. Nếu tính lại theo hướng dẫn tại Thông tư này mà không có lợi hơn thì không xếp lại lương đối với các trường hợp này.
    """
    print(f"Original text length: {len(text.split())} tokens")
    tokenized_text = ViTokenizer.tokenize(text)
    print(f"Total tokens: {len(tokenized_text.split())}")
    separator = "\n"
    chunker = PuncChunker(max_tokens=512, chunk_overlap=0, separator=separator)

    chunks = chunker._chunk_(text)
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}\n")
        print(f"Chunk {i+1} length: {len(chunk.split())} tokens\n")
    print(f"Total chunks: {len(chunks)}")
