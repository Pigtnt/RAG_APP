import sys
import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import Config

def load_pdf_with_tables(file_path):
    """
    [Logic] 特殊處理：PDF 表格轉 Markdown
    功能：除了讀取純文字，還將表格轉換為 Markdown 格式，
    優點：大幅提升 LLM 對結構化數據 (如費率表、資格表) 的理解能力。
    """
    print(f"📄 解析 PDF: {file_path}")
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            
            # 提取表格並轉為 Markdown
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
            
            # 組合內容：純文字 + 表格 Markdown
            full_content = text
            if table_markdowns:
                full_content += "\n\n=== 表格結構 (Markdown) ===\n" + "\n".join(table_markdowns)
            
            # 封裝為 Document 物件
            docs.append(Document(page_content=full_content, metadata={"source": file_path, "page": i + 1}))
    return docs

def main():
    print("🚀 開始建立向量資料庫 (Ingestion)...")
    
    # 1. 讀取與處理
    raw_docs = load_pdf_with_tables(Config.PDF_PATH)
    
    # 2. 切分 (Text Splitting)
    # [Param] chunk_size: 1000 字元。若表格很大，建議設大一點以免表格被切斷。
    # [Param] chunk_overlap: 200 字元。保留上下文重疊，避免語意斷裂。
    # [Param] separators: 切分優先級 (段落 -> 句子 -> 空格)。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", "。", "！", "？", " ", ""])
    splits = text_splitter.split_documents(raw_docs)
    print(f"📦 切分完成：共 {len(splits)} 個區塊")

    # 3. Embedding
    # [Core] 將文字轉為向量數值
    print("🧠 載入 Embedding 模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, # [Param] 若有 GPU 可改為 'cuda'
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 存入 Chroma (Vector Database)
    # [Core] 寫入硬碟 (Persist)，供後續查詢使用
    print(f"💾 寫入資料庫至 {Config.DB_PATH} ...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name=Config.COLLECTION_NAME, # [Param] 資料集名稱
        persist_directory=Config.DB_PATH        # [Param] 儲存路徑
    )
    
    print("✅ 資料庫建立完成！請執行 uv run streamlit run app.py")

if __name__ == "__main__":
    main()