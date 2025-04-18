import os
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from google import genai
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, MilvusClient, Function, FunctionType
from uuid import uuid4
from dotenv import load_dotenv


load_dotenv()

class MilvusDatabase:
    def __init__(self, collection_name: str = 'hpt_rag_pipeline', vector_size: int = 1536):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.uri = os.getenv('ZILLIZ_CLOUD_URI')
        self.token = os.getenv('ZILLIZ_CLOUD_TOKEN')
        
        # Kiểm tra kết nối tới Milvus
        try: 
            self.client = MilvusClient(uri=self.uri, token=self.token)
        except Exception as e:
            raise ValueError(f"Không thể kết nối tới Milvus: {e}") from e
        
        # Kiểm tra xem collection đã tồn tại chưa
        if not self.client.has_collection(self.collection_name):
            # Nếu chưa tồn tại, tạo mới collection
            self._create_collection()
                
                
    def _create_collection(self):
        try:
            analyzer_params = {
                "tokenizer": "standard", # Mandatory: Specifies tokenizer
                "filter": ["lowercase"], # Optional: Built-in filter that converts text to lowercase
            }

            fields = [
                FieldSchema(name='id', dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=analyzer_params),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_size),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="number", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="issued_date", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="article", dtype=DataType.VARCHAR, max_length=255)
            ]
            schema = CollectionSchema(fields=fields, description="HPT RAG Pipeline Collection")
            
            bm25_function = Function(
                name='text_bm25_embedding',
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,  
                         
            )
            
            schema.add_function(bm25_function)
            
            index_params = self.client.prepare_index_params()
            
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="FLAT",
                metric_type="COSINE",
                params={},
            )
            
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
            
            self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
        except Exception as e:
            raise ValueError(f"Không thể tạo collection '{self.collection_name}': {e}") from e
        
    def insert(self, data: List[dict]) -> None:
        """Chèn dữ liệu vào collection."""
        try: 
            entities = []
            for item in data:
                # Chuẩn bị các trường metadata, xử lý các trường bị khuyết hoặc các trường có kiểu list
                metadata = item["metadata"]
                chapter = "|".join(metadata.get("chapter", ["", ""])) if metadata.get("chapter") else ""
                section = "|".join(metadata.get("section", ["", ""])) if metadata.get("section") else ""
                article = "|".join(metadata.get("article", ["", ""])) if metadata.get("article") else ""   
                
                entity = {
                    "id": item["id"],
                    "text": item.get("text", ""),
                    "dense_vector": item["dense_vector"],
                    # "sparse_vector": item["sparse_vector"],
                    "source": metadata.get("source", ""),
                    "type": metadata.get("type", ""),
                    "title": metadata.get("title", ""),
                    "number": metadata.get("number", ""),
                    "issued_date": metadata.get("issued_date", ""),
                    "chapter": chapter,
                    "section": section,
                    "article": article
                }
                entities.append(entity) 
            self.client.insert(collection_name=self.collection_name, data=entities)
        except Exception as e:
            raise ValueError(f"Không thể chèn dữ liệu vào collection '{self.collection_name}': {e}") from e
    

    

        
        
        
        
