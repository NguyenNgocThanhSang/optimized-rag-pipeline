import os 
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai import OpenAI
from .base import BaseEmbedding
load_dotenv()

class OpenAIEmbedding(BaseEmbedding):
    """Lớp cho mô hình embedding OpenAI"""
    
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-3-small"):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        
        if not api_key:
            raise ValueError("API key không được cung cấp.")
        
        try: 
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Không thể khởi tạo client OpenAI. Lỗi: {e}") from e
        
    def embed_documents(self, texts):
        try: 
            results = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
        
            return [embedding.embedding for embedding in results.data]
        except Exception as e:
            raise ValueError(f"Không gọi được API của OpenAI Embedding. Lỗi: {e}") from e
        
    def embed_query(self, text):
        try:
            results = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            return results.data[0].embedding
        except Exception as e:
            raise ValueError(f"Không gọi được API của OpenAI Embedding. Lỗi: {e}") from e
        
    
    

            
        