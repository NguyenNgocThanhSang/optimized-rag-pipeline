from google import genai
from google.genai import types
from .base import BaseEmbedding
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, api_key: str = None, model_name: str = "models/text-embedding-004"):
        """Khởi tạo lớp GeminiEmbedding với API key và tên mô hình."""
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        
        if not api_key:
            raise ValueError("API key không được cung cấp.")
        
        try: 
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Không thể khởi tạo client Gemini. Lỗi: {e}") from e    
            
        
    def embed_documents(self, texts):
        try: 
            results = self.client.models.embed_content(
                model=self.model_name,
                content=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )

            return [embedding.values for embedding in results.embeddings]
        except Exception as e:
            raise ValueError(f"Không gọi được API của Gemini Embedding. Lỗi: {e}") from e
        
    def embed_query(self, text):
        try:
            results = self.client.models.embed_content(
                model=self.model_name,
                content=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768
                )
            )

            return results.embeddings[0].values
        except Exception as e:
            raise ValueError(f"Không gọi được API của Gemini Embedding. Lỗi: {e}") from e
        
        
        
        
        