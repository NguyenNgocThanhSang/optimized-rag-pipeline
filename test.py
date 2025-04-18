import os
from src.loaders.word_loader import DocumentLoader
from src.vectorizer.vectorizer import Vectorizer
from src.vectorstores.milvus import MilvusDatabase

# Cấu hình đường dẫn file DOCX cần upload
file_path = "documents/85_2016_ND-CP_317475.docx"

# Kiểm tra file tồn tại
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

try:
    # Bước 1: Load và tách chunk văn bản
    loader = DocumentLoader(file_path=file_path)
    docs = loader.load_and_split()

    # Bước 2: Tạo embeddings
    vectorizer = Vectorizer(use_dense=True)
    vectorized_docs = vectorizer.vectorizer(docs)

    # Bước 3: Kết nối và insert vào Milvus
    milvus_db = MilvusDatabase(collection_name="hpt_rag_pipeline", vector_size=1536)
    milvus_db.insert(vectorized_docs)

    print(f"✅ Upload thành công {file_path} ({len(vectorized_docs)} chunks) lên Milvus.")

except Exception as e:
    print(f"❌ Lỗi khi xử lý file {file_path}: {e}")
