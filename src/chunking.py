from __future__ import annotations

import math
import re




class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks

class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Tách câu dựa trên các dấu kết thúc phổ biến
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk)
        return chunks

class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Chiến lược tách đệ quy để giữ ngữ cảnh tốt nhất có thể."""
        if len(text) <= self.chunk_size or not separators:
            return [text]

        # Chọn separator ưu tiên cao nhất
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Tách văn bản
        final_chunks = []
        if separator == "": # Trường hợp cuối cùng: tách cứng theo độ dài
            for i in range(0, len(text), self.chunk_size):
                final_chunks.append(text[i : i + self.chunk_size])
            return final_chunks

        # Tách theo separator hiện tại
        parts = text.split(separator)
        current_chunk = ""

        for part in parts:
            # Nếu phần nhỏ này vẫn lớn hơn chunk_size, gọi đệ quy với separator tiếp theo
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                recursive_parts = self._split(part, remaining_separators)
                final_chunks.extend(recursive_parts)
            
            # Gom các phần nhỏ lại cho đến khi đạt ngưỡng chunk_size
            elif len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk = current_chunk + (separator if current_chunk else "") + part
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return [c for c in final_chunks if c]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính toán Cosine Similarity."""
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(x**2 for x in vec_a))
    mag_b = math.sqrt(sum(x**2 for x in vec_b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)

class ChunkingStrategyComparator:
    """So sánh kết quả của các chiến lược chunking."""
    
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        comparison = {}
        for name, strategy in strategies.items():
            result = strategy.chunk(text)
            comparison[name] = {
                "count": len(result),
                "avg_length": sum(len(c) for c in result) / len(result) if result else 0,
                "chunks": result[:2] # Trả về 2 mẫu đầu tiên để xem thử
            }
        return comparison

