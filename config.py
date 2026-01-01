import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

class Config:
    # 檔案路徑
    PDF_PATH = "fubon.pdf"          # 您的 PDF 檔名
    DB_PATH = "./db/chroma_db"      # 資料庫儲存資料夾
    COLLECTION_NAME = "fubon_hybrid_v1"

    # 模型與 API 設定
    API_KEY = os.getenv("GEMINI_API_KEY")
    GENERATOR_MODEL = "gemma-3-12b-it"
    JUDGE_MODEL = "gemma-3-27b-it"
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    
    # 檢查 API Key
    if not API_KEY:
        raise ValueError("❌ 找不到 GEMINI_API_KEY，請檢查 .env 檔案！")