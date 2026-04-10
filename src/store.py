from __future__ import annotations

import uuid
from typing import Any, Callable

# Giả định các import này đã tồn tại trong cấu trúc thư mục của bạn
# từ câu hỏi trước: .chunking có _dot, .embeddings có _mock_embed, .models có Document
from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.
    Tries to use ChromaDB if available; falls back to an in-memory store.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._use_chroma = False

        try:
            import chromadb
            from chromadb.config import Settings

            # Khởi tạo ephemeral client (chạy trong RAM)
            # Nếu muốn lưu xuống đĩa, bạn có thể đổi thành PersistentClient(path="./path")
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except (ImportError, Exception):
            print("ChromaDB not found or failed to init. Falling back to in-memory store.")
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Chuẩn hóa dữ liệu Document thành một record để lưu trữ."""
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata)
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
            
        return {
            "id": str(uuid.uuid4()) if not doc.id else doc.id,
            "embedding": embedding,
            "content": doc.content,
            "metadata": metadata
        }

    def _search_records(self, query_embedding: list[float], records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Tìm kiếm tương đồng trong bộ nhớ bằng Dot Product."""
        similarities = []
        for rec in records:
            # Tính độ tương đồng giữa query và từng chunk
            score = _dot(query_embedding, rec["embedding"])
            similarities.append((score, rec))

        # Sắp xếp theo score giảm dần
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Trả về top_k record kèm theo score
        results = []
        for score, rec in similarities[:top_k]:
            result = rec.copy()
            result["score"] = score
            results.append(result)
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """Nhúng (Embed) văn bản và lưu vào kho lưu trữ."""
        if self._use_chroma:
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                embeddings.append(record["embedding"])
                documents.append(record["content"])
                metadatas.append(record["metadata"])
            
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Tìm kiếm top_k tài liệu tương đồng nhất."""
        query_embedding = self._embedding_fn(query)

        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            # Format lại kết quả từ Chroma cho đồng nhất
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i] if "distances" in results else None
                })
            return formatted_results
        else:
            return self._search_records(query_embedding, self._store, top_k)

    def get_collection_size(self) -> int:
        """Trả về tổng số chunk đang lưu trữ."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Tìm kiếm kết hợp lọc metadata."""
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter # Chroma hỗ trợ lọc trực tiếp
            )
            # (Phần format kết quả tương tự hàm search bên trên)
            return [{"id": results["ids"][0][i], "content": results["documents"][0][i]} for i in range(len(results["ids"][0]))]
        
        else:
            # Logic lọc thủ công cho In-memory
            filtered_records = []
            for rec in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if rec["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(rec)
            
            query_embedding = self._embedding_fn(query)
            return self._search_records(query_embedding, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Xóa tất cả các chunk thuộc về một doc_id cụ thể."""
        initial_count = self.get_collection_size()
        
        if self._use_chroma:
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < initial_count
        else:
            # Lọc lại store, chỉ giữ lại những record không trùng doc_id
            self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_count