import sys
from langchain_community.document_loaders import PDFPlumberLoader # 或沿用你原本的 pdfplumber 寫法
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config
import pdfplumber
from langchain_core.documents import Document

# 這裡沿用你原本優秀的 PDF 轉 Markdown 表格邏輯
def load_pdf_with_tables(file_path):
    print(f"📄 解析 PDF: {file_path}")
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_markdowns = []
            for table in tables:
                if not table: continue
                clean_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                if len(clean_table) > 0:
                    header = "| " + " | ".join(clean_table[0]) + " |"
                    separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
                    body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_table[1:]])
                    table_markdowns.append(f"\n{header}\n{separator}\n{body}\n")
            
            full_content = text
            if table_markdowns:
                full_content += "\n\n=== 表格結構 (Markdown) ===\n" + "\n".join(table_markdowns)
            docs.append(Document(page_content=full_content, metadata={"source": file_path, "page": i + 1}))
    return docs

def main():
    print("🚀 開始建立向量資料庫 (Ingestion)...")
    
    # 1. 讀取與處理
    raw_docs = load_pdf_with_tables(Config.PDF_PATH)
    
    # 2. 切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", "。", "！", "？", " ", ""])
    splits = text_splitter.split_documents(raw_docs)
    print(f"📦 切分完成：共 {len(splits)} 個區塊")

    # 3. Embedding
    print("🧠 載入 Embedding 模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 存入 Chroma (關鍵：設定 persist  _directory)
    print(f"💾 寫入資料庫至 {Config.DB_PATH} ...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name=Config.COLLECTION_NAME,
        persist_directory=Config.DB_PATH  # 設定儲存路徑
    )
    
    print("✅ 資料庫建立完成！請執行 streamlit run app.py")

if __name__ == "__main__":
    main()