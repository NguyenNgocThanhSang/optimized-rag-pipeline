from typing import List, Dict, Any
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer, Analyzer
import json
import py_vncorenlp
from pathlib import Path

analyzer = Analyzer(name='en', tokenizer='white_space')

class BM25Embedding(BM25EmbeddingFunction):
    """Lớp BM25 Embedding sử dụng VNCoreNLP để phân tích cú pháp tiếng Việt."""
    
    def __init__(self, analyzer = analyzer, corpus = None, k1 = 1.5, b = 0.75, epsilon = 0.25, num_workers = 1):
        super().__init__(analyzer, corpus, k1, b, epsilon, num_workers)
        self.vncorenlp = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:/VnCoreNLP', max_heap_size='-Xmx2g')
        
    def vncore_tokenize(self, text: str) -> List[str]:
        """Phân tích cú pháp văn bản tiếng Việt bằng VNCoreNLP."""
        return self.vncorenlp.word_segment(text)
    
    def save(self, path: str):
        bm25_params = {}
        bm25_params["version"] = "v1"
        bm25_params["corpus_size"] = self.corpus_size
        bm25_params["avgdl"] = self.avgdl
        bm25_params["idf_word"] = [None for _ in range(len(self.idf))]
        bm25_params["idf_value"] = [None for _ in range(len(self.idf))]
        for word, values in self.idf.items():
            bm25_params["idf_word"][values[1]] = word
            bm25_params["idf_value"][values[1]] = values[0]

        bm25_params["k1"] = self.k1
        bm25_params["b"] = self.b
        bm25_params["epsilon"] = self.epsilon

        with Path(path).open("w") as json_file:
            json.dump(bm25_params, json_file, ensure_ascii=False, indent=4)
            
    
