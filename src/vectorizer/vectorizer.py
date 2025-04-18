from typing import List
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.embeddings.bm25_embedding import BM25Embedding
from uuid import uuid4
import os
from dotenv import load_dotenv

load_dotenv()

class Vectorizer:
    """Lớp tạo các embeddings cho văn bản."""
    
    def __init__(self, use_dense: bool = True):
        self.use_dense = use_dense
        try: 
            self.dense_model = OpenAIEmbedding(os.getenv("OPENAI_API_KEY")) if use_dense else None
        except Exception as e:
            raise ValueError(f"Không thể khởi tạo mô hình OpenAI: {e}") from e
        
    def vectorizer(self, docs: List[Document]) -> List[dict]:
        """Tạo các embeddings cho văn bản."""
        entities = []
        
        if self.use_dense:
            try: 
                dense_vectors = self.dense_model.embed_documents([doc.page_content for doc in docs])
            except Exception as e:
                dense_vectors = [None] * len(docs)
                raise ValueError(f"Không thể tạo embeddings cho văn bản: {e}") from e
        else:
            dense_vectors = [None] * len(docs)
            
        for doc, dense_vector in zip(docs, dense_vectors):
            vector_data = {
                "id": str(uuid4()),
                "text": doc.page_content,
                "dense_vector": dense_vector,
                "sparse_vector": None, # sparse vector sẽ được tạo bởi Milvus BM25 function
                "metadata": doc.metadata
            }
            entities.append(vector_data)
        
        return entities
   
        
    