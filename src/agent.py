from typing import Callable
from .store import EmbeddingStore

class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """
        Khởi tạo Agent.
        :param store: Thực thể của EmbeddingStore đã chứa dữ liệu.
        :param llm_fn: Một hàm nhận vào một chuỗi (prompt) và trả về một chuỗi (phản hồi từ LLM).
        """
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Thực hiện quy trình RAG để trả lời câu hỏi.
        """
        # 1. Tìm kiếm các đoạn văn bản (chunks) liên quan nhất từ Store
        search_results = self.store.search(query=question, top_k=top_k)
        
        if not search_results:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi này."

        # 2. Xây dựng Context từ các kết quả tìm kiếm
        # Chúng ta lấy nội dung 'content' từ mỗi record được trả về
        context_parts = []
        for i, res in enumerate(search_results, 1):
            content = res.get("content", "")
            context_parts.append(f"[Đoạn {i}]: {content}")
        
        context_text = "\n\n".join(context_parts)

        # 3. Tạo Prompt hoàn chỉnh cho LLM
        # Cấu trúc: Ngữ cảnh + Câu hỏi -> LLM
        prompt = f"""
Sử dụng thông tin dưới đây để trả lời câu hỏi của người dùng một cách chính xác nhất. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời.

NGỮ CẢNH:
{context_text}

CÂU HỎI: 
{question}

TRẢ LỜI:
""".strip()

        # 4. Gọi LLM và trả về kết quả
        return self.llm_fn(prompt)