from abc import ABC, abstractmethod
from typing import List

class BaseEmbedding(ABC):
    """Lớp trừu tượng cho mô hình embedding."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Chuyển đổi danh sách văn bản thành danh sách embedding."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Chuyển đổi một văn bản thành embedding."""
        pass
    