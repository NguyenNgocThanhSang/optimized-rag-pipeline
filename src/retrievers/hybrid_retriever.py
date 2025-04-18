import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, RRFRanker
from langchain_core.documents import Document
from src.embeddings.openai_embedding import OpenAIEmbedding
from rich import print 
from src.rerankers.hybridcc_ranker import HybridCCRanker
from rich import traceback
traceback.install()

load_dotenv()

class HybridRetriever:
    def __init__(self, collection_name: str = "hpt_rag_pipeline", output_fields: Optional[List[str]] = None, cc_alpha: float = 0.5):
        self.collection_name = collection_name
        self.uri = os.getenv("ZILLIZ_CLOUD_URI")
        self.token = os.getenv("ZILLIZ_CLOUD_TOKEN")
        self.openai_client = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
        self.milvus_client = MilvusClient(
            uri=self.uri,
            token=self.token
        )
        self.ranker = RRFRanker(60)
        
        self.ccranker = HybridCCRanker(alpha=cc_alpha)
        
    def dense_search(self, query: str, top_k: int = 5) -> List[List[dict]]:
        dense_vector = self.openai_client.embed_query(query)
        
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            anns_field="dense_vector",
            data=[dense_vector],
            # limit=top_k,
            search_params={"metric_type": "COSINE"},
            output_fields=['text']
        )
        
        return res


    def sparse_search(self, query: str, top_k: int = 5) -> List[List[dict]]:
        search_params = {
            'params': {'drop_ratio_search': 0.2}, # Proportion of small vector values to ignore during the search
        }

        res = self.milvus_client.search(
            collection_name=self.collection_name, 
            data=[query],
            anns_field='sparse_vector',
            limit=3,
            search_params=search_params,
            output_fields=['text']
        )
        
        return res
        
    def hybrid_search(self, query: str, top_k: int = 5, search_k_dense: int = 20, search_k_sparse: int = 20):
        """
        Performs hybrid search by combining dense and sparse results
        and reranking them using the initialized HybridCCRanker.

        Args:
            query: The search query string.
            top_k: The final number of results to return after reranking.
            search_k_dense: The number of results to fetch from the initial dense search.
            search_k_sparse: The number of results to fetch from the initial sparse search.

        Returns:
            A list of reranked documents (dictionaries) sorted by the Hybrid CC score.
        """
        print(f"\nInitiating Hybrid Search (Dense k={search_k_dense}, Sparse k={search_k_sparse}, Final k={top_k})...")

        # 1. Perform Dense Search
        dense_results = self.dense_search(query, top_k=search_k_dense)
        num_dense = len(dense_results[0]) if dense_results and dense_results[0] else 0
        print(f"Retrieved {num_dense} results from dense search.")

        # 2. Perform Sparse Search
        sparse_results = self.sparse_search(query, top_k=search_k_sparse)
        num_sparse = len(sparse_results[0]) if sparse_results and sparse_results[0] else 0
        print(f"Retrieved {num_sparse} results from sparse search.")

        if num_dense == 0 and num_sparse == 0:
             print("No results found from either dense or sparse search.")
             return []

        # 3. Rerank using HybridCCRanker
        print("Reranking results using HybridCCRanker...")
        try:
            reranked_results = self.ccranker.rerank(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k
            )
            print(f"Reranking complete. Returning top {len(reranked_results)} results.")
            return reranked_results
        except Exception as e:
            print(f"Error during Hybrid CC reranking: {e}")
            # Fallback strategy: return dense results if reranking fails?
            print("Warning: Reranking failed. Returning raw dense results as fallback.")
            return (dense_results[0] if dense_results and dense_results[0] else [])[:top_k]

                

from src.retrievers.hybrid_retriever import HybridRetriever

def test_hybrid_search():
    query = "thủ tục xác định cấp độ an toàn thông tin"
    retriever = HybridRetriever(collection_name="hpt_rag_pipeline")

    print(f"\n🔎 Đang tìm kiếm với truy vấn: '{query}'\n")
    print(retriever.hybrid_search(query=query))
    
    



if __name__ == "__main__":
    test_hybrid_search()
