import os
import sys
import time
import re
import pdfplumber  # 需安裝 pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 🔍 檢查 API Key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ 錯誤：找不到 GOOGLE_API_KEY！")
    sys.exit(1)

# ==========================================
# 🛠️ 設定區
# ==========================================
file_path = "fubon.pdf"
GENERATOR_MODEL = "gemma-3-12b-it"  # 球員
JUDGE_MODEL = "gemma-3-27b-it"  # 裁判

print(f"🚀 啟動 RAG 系統 (Level 5.5 - Hybrid Search + Markdown Tables)...")
print(f"📄 處理檔案: {file_path}")


# ==========================================
# 1. 資料處理 (關鍵改良：表格轉 Markdown)
# ==========================================
def pdf_to_markdown_with_plumber(file_path):
    """
    使用 pdfplumber 讀取 PDF，將偵測到的表格轉換為 Markdown 格式，
    並附加在頁面純文字之後。
    """
    print(f"   [系統提示] 解析 PDF 並轉換表格為 Markdown: {file_path} ...")
    docs = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. 提取純文字 (保留原本內容)
                text = page.extract_text() or ""

                # 2. 提取表格
                tables = page.extract_tables()
                table_markdowns = []

                for table in tables:
                    if not table: continue

                    # 清理表格資料 (處理 None)
                    clean_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]

                    # 轉為 Markdown
                    if len(clean_table) > 0:
                        # 處理 Header
                        header = "| " + " | ".join(clean_table[0]) + " |"
                        separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"

                        # 處理 Body
                        body_rows = []
                        for row in clean_table[1:]:
                            body_rows.append("| " + " | ".join(row) + " |")

                        body = "\n".join(body_rows)
                        md_table = f"\n{header}\n{separator}\n{body}\n"
                        table_markdowns.append(md_table)

                # 3. 組合內容：純文字 + 標示 + Markdown 表格
                # 這樣做的好處是：文字題查得到 text，表格題查得到 markdown
                full_content = text
                if table_markdowns:
                    full_content += "\n\n=== 偵測到的表格結構 (Markdown) ===\n" + "\n".join(table_markdowns)

                docs.append(Document(
                    page_content=full_content,
                    metadata={"source": file_path, "page": i + 1}
                ))
    except Exception as e:
        print(f"❌ PDF 解析錯誤: {e}")
        sys.exit(1)

    return docs


def create_retriever(file_path):
    # 1. 載入並轉換資料
    raw_docs = pdf_to_markdown_with_plumber(file_path)

    # 2. 切分 (關鍵：加大 Chunk Size 以容納 Markdown 表格)
    # 原本 400 太小，表格會被切斷。改為 1000。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(raw_docs)
    print(f"   [系統提示] 文件已切分為 {len(splits)} 個區塊 (Chunk Size: 1000)")

    # 3. Embedding 模型
    print("   [系統提示] 載入 Embedding 模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 建立 Chroma (向量檢索)
    print("   [系統提示] 建立 Chroma 向量索引...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="fubon_markdown_hybrid"  # 改個名字確保不混用舊資料
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 5. 建立 BM25 (關鍵字檢索)
    print("   [系統提示] 建立 BM25 關鍵字索引...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10

    # 6. 混合檢索
    print("   [系統提示] 啟動 Hybrid Ensemble...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever, vectorstore  # 回傳 vectorstore 以便最後清理


# 建立檢索器
retriever, vector_db = create_retriever(file_path)

# 載入 Re-ranker
print("   [系統提示] 載入 Cross-Encoder Re-ranker...")
reranker_model = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

# ==========================================
# 2. 模型與 Prompt 設定
# ==========================================
try:
    llm_generator = ChatGoogleGenerativeAI(
        model=GENERATOR_MODEL, temperature=0.1, google_api_key=GOOGLE_API_KEY
    )
    llm_judge = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL, temperature=0.0, google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"❌ 模型初始化失敗: {e}")
    sys.exit(1)

# Prompt Template (針對表格優化)
rag_template = """# Role
你是一位專業、邏輯嚴謹的「台北富邦銀行頂級卡權益審核專員」。你的任務是根據 <context> 準確回答客戶問題。

# Task
請閱讀 <context>，並針對 <question> 進行資格審核與回覆。
<context> 可能包含 Markdown 格式的表格，請仔細對照表格的欄位與數值。

# Constraints
1. **嚴格引用**：回答必須基於 <context> 內容，回答結尾請標註來源。
2. **表格對照**：若資料為 Markdown 表格，請確保「欄位」與「列」的對應關係正確 (例如：確認「卡別」對應的「門檻」)。
3. **數值比對**：若問題涉及金額、天數，請在思考過程中列出算式比對。
4. **排除條款**：特別檢查「一般消費定義」的排除項目。
5. **誠實回答**：若 <context> 未提及，回答「手冊中未提及」。

# Instruction (CoT)
在回答前，請務必先進行 <thinking> 步驟：
1. **識別變數**：用戶的卡別、身分、消費金額、時間點。
2. **查找條款**：在 <context> 中找到對應規則（優先查看 Markdown 表格）。
3. **邏輯判定**：
   - 資格檢查：用戶金額 vs 門檻？
   - 期限檢查：天數 vs 限制？
   - 排除檢查：是否在排除名單？
4. **生成回答**：根據判定結果回覆。

# Context
{context}

# Question
{question}

Answer:"""

rag_prompt = PromptTemplate.from_template(rag_template)


# ==========================================
# 3. 評分邏輯 (RAGAS - Lite)
# ==========================================
def calculate_ragas_score(question, answer, contexts):
    # Faithfulness
    f_prompt = PromptTemplate.from_template("""
    你是一位嚴格的 RAG 評測員。請檢查「AI 回答」是否包含「參考片段」中沒有的幻覺資訊。
    若參考片段中有 Markdown 表格，請確認 AI 是否正確讀取表格數據。
    【參考片段】：{contexts}
    【AI 回答】：{answer}
    請回傳 0.0 到 1.0 的分數。只回傳數字。
    """)

    # Relevance
    r_prompt = PromptTemplate.from_template("""
    你是一位嚴格的 RAG 評測員。請評分「AI 回答」是否精準回答了「用戶問題」。
    【用戶問題】：{question}
    【AI 回答】：{answer}
    請回傳 0.0 到 1.0 的分數。只回傳數字。
    """)

    try:
        f_chain = f_prompt | llm_judge | StrOutputParser()
        r_chain = r_prompt | llm_judge | StrOutputParser()

        f_str = f_chain.invoke({"contexts": contexts, "answer": answer}).strip()
        r_str = r_chain.invoke({"question": question, "answer": answer}).strip()

        f_match = re.findall(r"[-+]?\d*\.\d+|\d+", f_str)
        r_match = re.findall(r"[-+]?\d*\.\d+|\d+", r_str)

        return (float(f_match[0]) if f_match else 0.0, float(r_match[0]) if r_match else 0.0)
    except:
        return 0.0, 0.0


# ==========================================
# 4. 主執行流程
# ==========================================
def run_rag_with_evaluation(query):
    print(f"\n❓ 測試問題: {query}")
    print("-" * 50)

    # 1. Recall & Rerank
    initial_docs = retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker_model.predict(pairs)
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_3_docs = [doc for doc, score in scored_docs[:3]]

    context_text = "\n\n".join([doc.page_content for doc in top_3_docs])

    # Debug: 可以在這裡印出 context_text 檢查是否有 Markdown 表格
    # print(f"[DEBUG] Context Preview: {context_text[:200]}...")

    # 2. Generation
    print(f"🤖 球員 (12B) 生成回答中 (Streaming)...")
    print("-" * 20)
    chain = rag_prompt | llm_generator | StrOutputParser()
    full_answer = ""
    for chunk in chain.stream({"context": context_text, "question": query}):
        print(chunk, end="", flush=True)
        full_answer += chunk
    print()

    # 3. Evaluation
    print("-" * 50)
    print(f"⚖️ 裁判 (27B) 評分中...")
    f_score, r_score = calculate_ragas_score(query, full_answer, context_text)

    print(f"📊 評分報告: F={f_score:.2f}, R={r_score:.2f}")
    if f_score < 0.8:
        print("   ⚠️  警示：可能產生幻覺！")
    elif r_score < 0.5:
        print("   ⚠️  警示：答非所問！")
    else:
        print("   ✅ Pass：表現優良。")
    print("=" * 60)


# ==========================================
# 5. 測試題庫
# ==========================================
questions = [
    # 簡單題
    "尊御世界卡年費多少？",
    # 表格題 (關鍵測試 Q2)
    "我要預約連假期間的『國內機場接送』，最晚需要在幾個工作天前預約？",
    "請問道路救援服務專線的電話號碼是多少？",

    # 陷阱題 (表格 + 邏輯)
    "我上週剛買了機票，金額是 12,000 元，我是富邦世界卡的卡友（非理財會員），請問我可以預約免費機場接送嗎？",
    "我用尊御世界卡刷了機票，但是是在 7 個月前（約 210 天前）刷的，現在要出國可以用機場外圍停車嗎？",
    "我是富邦無限卡持卡人，我兒子今年 26 歲未婚，跟我一起出國，我幫他刷了全額機票，請問他有旅遊平安險的保障嗎？",

    # 表格題 (關鍵測試 Q7 - 這是最難的)
    "我持有富邦世界卡，上期帳單一般消費 18,000 元，請問我去『台灣聯通』停車可以免費停幾小時？",

    # 否定題
    "我為了湊免年費的門檻，去全聯福利中心買了很多東西，請問這些消費算在『一般消費』裡面嗎？",
    "我剛買的新手機被偷了，可以用信用卡的『全球購物保障』申請理賠嗎？",
    "我的車有改裝過，底盤比較低（離地 15 公分），車子拋錨了可以使用免費道路救援拖吊嗎？"
]

for q in questions:
    run_rag_with_evaluation(q)
    time.sleep(3)

# 清理
vector_db.delete_collection()
print("\n✅ 所有測試完成！")