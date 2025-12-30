# import os
# import time
# import re
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from sentence_transformers import CrossEncoder
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # ==========================================
# # 🛠️ 設定區：雙模型架構 (Player vs Judge)
# # ==========================================
# file_path = "fubon.pdf"
#
# # 🟢 球員：負責回答問題 (輕量級)
# # 注意：若 API 報錯，請嘗試加上 "models/" 前綴，如 "models/gemma-3-12b-it"
# GENERATOR_MODEL = "gemma-3-12b-it"
#
# # 🟢 裁判：負責評分 (重量級，邏輯更強)
# JUDGE_MODEL = "gemma-3-27b-it"
#
# print(f"🚀 啟動 RAG 系統 (雙模型協作版)...")
# print(f"🏃 球員模型 (Generator): {GENERATOR_MODEL}")
# print(f"👨‍⚖️ 裁判模型 (Judge):     {JUDGE_MODEL}")
# print(f"📄 處理檔案: {file_path}")
#
#
# # ==========================================
# # 1. 資料處理
# # ==========================================
# def load_and_split_pdf(file_path):
#     print("   [系統提示] PDFPlumber 解析中...")
#     loader = PDFPlumberLoader(file_path)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
#     splits = text_splitter.split_documents(docs)
#     return splits
#
#
# # ==========================================
# # 2. 載入檢索模型
# # ==========================================
# print("   [系統提示] 載入 Embedding & Cross-Encoder...")
# embedding_model = HuggingFaceEmbeddings(
#     model_name="intfloat/multilingual-e5-large",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# reranker_model = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
#
# # ==========================================
# # 3. 建立向量資料庫
# # ==========================================
# pdf_splits = load_and_split_pdf(file_path)
# vectorstore = Chroma.from_documents(
#     documents=pdf_splits,
#     embedding=embedding_model,
#     collection_name="fubon_dual_model_demo"  # 改名避免衝突
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#
# # ==========================================
# # 4. 設定 LLM (雙模型實例化)
# # ==========================================
#
# # 🟢 1. 球員 (Generator) - Gemma 3 12B
# llm_generator = ChatGoogleGenerativeAI(
#     model=GENERATOR_MODEL,
#     temperature=0.1,
#     safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
# )
#
# # 🟢 2. 裁判 (Judge) - Gemma 3 27B
# llm_judge = ChatGoogleGenerativeAI(
#     model=JUDGE_MODEL,
#     temperature=0.0,  # 裁判必須絕對客觀
#     safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
# )
#
# # RAG 回答 Prompt
# rag_template = """你是一位專業的銀行客服。請嚴格依據下方【相關片段】回答【客戶問題】。
# 若無相關資訊，請回答「手冊中未提及」。
#
# 【相關片段】：
# {context}
#
# 【客戶問題】：
# {question}
#
# 回答:"""
# rag_prompt = PromptTemplate.from_template(rag_template)
#
#
# # ==========================================
# # 🟢 RAGAS 評分邏輯 (使用 llm_judge / 27B)
# # ==========================================
# def calculate_ragas_score(question, answer, contexts):
#     # Faithfulness (無幻覺)
#     faithfulness_prompt = PromptTemplate.from_template("""
#     你是一位嚴格的 RAG 評測員。
#     請檢查「AI 回答」是否包含「參考片段」中沒有的幻覺資訊。
#
#     【參考片段】：
#     {contexts}
#
#     【AI 回答】：
#     {answer}
#
#     請回傳一個 0.0 到 1.0 的分數 (1.0 代表完全忠實於原文，無幻覺)。
#     只回傳數字，不要有其他文字。
#     """)
#
#     # Relevance (切題度)
#     relevance_prompt = PromptTemplate.from_template("""
#     你是一位嚴格的 RAG 評測員。
#     請評分「AI 回答」是否精準回答了「用戶問題」，且沒有答非所問。
#
#     【用戶問題】：
#     {question}
#
#     【AI 回答】：
#     {answer}
#
#     請回傳一個 0.0 到 1.0 的分數 (1.0 代表非常切題)。
#     只回傳數字，不要有其他文字。
#     """)
#
#     try:
#         # 🔴 關鍵：這裡使用 llm_judge (27B) 來執行評分
#         f_chain = faithfulness_prompt | llm_judge | StrOutputParser()
#         r_chain = relevance_prompt | llm_judge | StrOutputParser()
#
#         f_str = f_chain.invoke({"contexts": contexts, "answer": answer}).strip()
#         r_str = r_chain.invoke({"question": question, "answer": answer}).strip()
#
#         # 解析數字
#         f_match = re.findall(r"[-+]?\d*\.\d+|\d+", f_str)
#         r_match = re.findall(r"[-+]?\d*\.\d+|\d+", r_str)
#
#         f_score = float(f_match[0]) if f_match else 0.0
#         r_score = float(r_match[0]) if r_match else 0.0
#
#         return min(f_score, 1.0), min(r_score, 1.0)
#
#     except Exception as e:
#         print(f"   [評分系統錯誤] {e}")
#         return 0.0, 0.0
#
#
# # ==========================================
# # 主流程
# # ==========================================
# def run_rag_with_evaluation(query):
#     print(f"\n❓ 測試問題: {query}")
#     print("-" * 50)
#
#     # 1. Recall & Rerank
#     initial_docs = retriever.invoke(query)
#     pairs = [[query, doc.page_content] for doc in initial_docs]
#     scores = reranker_model.predict(pairs)
#     scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
#     top_3_docs = [doc for doc, score in scored_docs[:3]]
#
#     # 2. Generation (使用 llm_generator / 12B)
#     context_text = "\n\n".join([doc.page_content for doc in top_3_docs])
#     print(f"🤖 球員 (12B) 生成回答中...")
#
#     # 🔴 關鍵：這裡使用 llm_generator
#     chain = rag_prompt | llm_generator | StrOutputParser()
#     answer = chain.invoke({"context": context_text, "question": query})
#     print(f"💬 回答: {answer.strip()}")
#
#     # 3. Evaluation (使用 llm_judge / 27B)
#     print("-" * 50)
#     print(f"⚖️ 裁判 (27B) 評分中...")
#     f_score, r_score = calculate_ragas_score(query, answer, context_text)
#
#     print(f"📊 評分報告:")
#     print(f"   ➤ Faithfulness (無幻覺): {f_score:.2f} / 1.0")
#     print(f"   ➤ Relevance (切題度):   {r_score:.2f} / 1.0")
#
#     if f_score < 0.8:
#         print("   ⚠️  警示：球員可能產生幻覺！")
#     elif r_score < 0.5:
#         print("   ⚠️  警示：球員答非所問！")
#     else:
#         print("   ✅ Pass：表現優良。")
#     print("=" * 60)
#
#
# # ==========================================
# # 📝 10 題完整測試題庫 (簡單、陷阱、否定)
# # ==========================================
# questions = [
#     # --- 簡單題 ---
#     "請問富邦尊御世界卡的正卡年費是多少錢？附卡要年費嗎？",
#     "我要預約連假期間的『國內機場接送』，最晚需要在幾個工作天前預約？",
#     "請問道路救援服務專線的電話號碼是多少？",
#
#     # --- 陷阱題 (考驗 12B 的邏輯) ---
#     "我上週剛買了機票，金額是 12,000 元，我是富邦世界卡的卡友（非理財會員），請問我可以預約免費機場接送嗎？",
#     "我用尊御世界卡刷了機票，但是是在 7 個月前（約 210 天前）刷的，現在要出國可以用機場外圍停車嗎？",
#     "我是富邦無限卡持卡人，我兒子今年 26 歲未婚，跟我一起出國，我幫他刷了全額機票，請問他有旅遊平安險的保障嗎？",
#     "我持有富邦世界卡，上期帳單一般消費 18,000 元，請問我去『台灣聯通』停車可以免費停幾小時？",
#
#     # --- 否定題 (排除條款) ---
#     "我為了湊免年費的門檻，去全聯福利中心買了很多東西，請問這些消費算在『一般消費』裡面嗎？",
#     "我剛買的新手機被偷了，可以用信用卡的『全球購物保障』申請理賠嗎？",
#     "我的車有改裝過，底盤比較低（離地 15 公分），車子拋錨了可以使用免費道路救援拖吊嗎？"
# ]
#
# # 執行測試
# for q in questions:
#     run_rag_with_evaluation(q)
#     time.sleep(3)
#
# # 清理
# vectorstore.delete_collection()
# print("\n✅ 所有測試完成！")|
# import os
# import sys  # 用於中斷程式
# import time
# import re
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from sentence_transformers import CrossEncoder
# from dotenv import load_dotenv
#
# # 🟢 新增：混合檢索需要的模組
# # 混合檢索需要這兩個
# from langchain.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever
#
# load_dotenv()
#
# # 🔍 RAG 工程師的除錯檢查：確認 Key 是否存在
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
#
# if not GOOGLE_API_KEY:
#     print("❌ 錯誤：找不到 GOOGLE_API_KEY！")
#     print("   請確認目錄下是否有 .env 檔案，且內容包含 GOOGLE_API_KEY=AIza...")
#     sys.exit(1) # 直接停止，避免後面報錯
#
#
# # ==========================================
# # 🛠️ 設定區：雙模型架構 (Player vs Judge)
# # ==========================================
# file_path = "fubon.pdf"
#
# # 🟢 球員：負責回答問題 (輕量級)
# GENERATOR_MODEL = "gemma-3-12b-it"
#
# # 🟢 裁判：負責評分 (重量級，邏輯更強)
# JUDGE_MODEL = "gemma-3-27b-it"
#
# print(f"🚀 啟動 RAG 系統 (Level 5 - Hybrid Search 混合檢索版)...")
# print(f"🏃 球員模型 (Generator): {GENERATOR_MODEL}")
# print(f"👨‍⚖️ 裁判模型 (Judge):     {JUDGE_MODEL}")
# print(f"📄 處理檔案: {file_path}")
#
#
# # ==========================================
# # 1. 資料處理
# # ==========================================
# def load_and_split_pdf(file_path):
#     print("   [系統提示] PDFPlumber 解析中...")
#     loader = PDFPlumberLoader(file_path)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
#     splits = text_splitter.split_documents(docs)
#     return splits
#
#
# # ==========================================
# # 2. 載入檢索模型
# # ==========================================
# print("   [系統提示] 載入 Embedding & Cross-Encoder...")
# embedding_model = HuggingFaceEmbeddings(
#     model_name="intfloat/multilingual-e5-large",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# reranker_model = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
#
# # ==========================================
# # 3. 建立 Hybrid Retriever (關鍵修改區) 🟢
# # ==========================================
# pdf_splits = load_and_split_pdf(file_path)
#
# # --- A. 建立向量檢索 (Vector Search) ---
# # 擅長：語意理解 (例如知道「費用」跟「錢」有關)
# print("   [系統提示] 建立 Chroma 向量索引...")
# vectorstore = Chroma.from_documents(
#     documents=pdf_splits,
#     embedding=embedding_model,
#     collection_name="fubon_hybrid_final"  # 改名避免衝突
# )
# chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#
# # --- B. 建立關鍵字檢索 (Keyword Search - BM25) ---
# # 擅長：精確匹配 (例如精準抓到 "180天", "尊御世界卡", "20,000" 這些字)
# print("   [系統提示] 建立 BM25 關鍵字索引...")
# bm25_retriever = BM25Retriever.from_documents(pdf_splits)
# bm25_retriever.k = 10  # 讓它也抓 10 筆
#
# # --- C. 融合 (Ensemble) ---
# # 將兩者的結果結合，權重各半 (0.5/0.5)
# print("   [系統提示] 啟動 Hybrid Ensemble (混合檢索)...")
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[chroma_retriever, bm25_retriever],
#     weights=[0.5, 0.5]
# )
#
# # 將最終的檢索器設定為混合檢索器
# retriever = ensemble_retriever
#
# # ==========================================
# # 4. 設定 LLM (雙模型實例化)
# # ==========================================
# print("   [系統提示] 初始化 LLM 模型中...")
#
# try:
#     # ✅ 修改點：顯式傳入 google_api_key，避免觸發 DefaultCredentialsError
#     llm_generator = ChatGoogleGenerativeAI(
#         model=GENERATOR_MODEL,
#         temperature=0.1,
#         google_api_key=GOOGLE_API_KEY  # 強制指定 Key
#     )
#
#     llm_judge = ChatGoogleGenerativeAI(
#         model=JUDGE_MODEL,
#         temperature=0.0,
#         google_api_key=GOOGLE_API_KEY  # 強制指定 Key
#     )
# except Exception as e:
#     print(f"❌ 模型初始化失敗: {e}")
#     sys.exit(1)
#
#
# # RAG 回答 Prompt
# rag_template = """你是一位專業的銀行客服。請嚴格依據下方【相關片段】回答【客戶問題】。
# 若無相關資訊，請回答「手冊中未提及」。
#
# 【相關片段】：
# {context}
#
# 【客戶問題】：
# {question}
#
# 回答:"""
# rag_prompt = PromptTemplate.from_template(rag_template)
#
#
# # ==========================================
# # 🟢 RAGAS 評分邏輯 (使用 llm_judge / 27B)
# # ==========================================
# def calculate_ragas_score(question, answer, contexts):
#     # Faithfulness (無幻覺)
#     faithfulness_prompt = PromptTemplate.from_template("""
#     你是一位嚴格的 RAG 評測員。
#     請檢查「AI 回答」是否包含「參考片段」中沒有的幻覺資訊。
#
#     【參考片段】：
#     {contexts}
#
#     【AI 回答】：
#     {answer}
#
#     請回傳一個 0.0 到 1.0 的分數 (1.0 代表完全忠實於原文，無幻覺)。
#     只回傳數字，不要有其他文字。
#     """)
#
#     # Relevance (切題度)
#     relevance_prompt = PromptTemplate.from_template("""
#     你是一位嚴格的 RAG 評測員。
#     請評分「AI 回答」是否精準回答了「用戶問題」，且沒有答非所問。
#
#     【用戶問題】：
#     {question}
#
#     【AI 回答】：
#     {answer}
#
#     請回傳一個 0.0 到 1.0 的分數 (1.0 代表非常切題)。
#     只回傳數字，不要有其他文字。
#     """)
#
#     try:
#         # 🔴 關鍵：這裡使用 llm_judge (27B) 來執行評分
#         f_chain = faithfulness_prompt | llm_judge | StrOutputParser()
#         r_chain = relevance_prompt | llm_judge | StrOutputParser()
#
#         f_str = f_chain.invoke({"contexts": contexts, "answer": answer}).strip()
#         r_str = r_chain.invoke({"question": question, "answer": answer}).strip()
#
#         # 解析數字
#         f_match = re.findall(r"[-+]?\d*\.\d+|\d+", f_str)
#         r_match = re.findall(r"[-+]?\d*\.\d+|\d+", r_str)
#
#         f_score = float(f_match[0]) if f_match else 0.0
#         r_score = float(r_match[0]) if r_match else 0.0
#
#         return min(f_score, 1.0), min(r_score, 1.0)
#
#     except Exception as e:
#         print(f"   [評分系統錯誤] {e}")
#         return 0.0, 0.0
#
#
# # ==========================================
# # 主流程
# # ==========================================
# def run_rag_with_evaluation(query):
#     print(f"\n❓ 測試問題: {query}")
#     print("-" * 50)
#
#     # 1. Recall (Hybrid Search)
#     # 這裡的 invoke 會同時跑 Vector 和 BM25，然後混合結果
#     initial_docs = retriever.invoke(query)
#
#     # 2. Re-rank (Cross-Encoder 決選)
#     # 將 Hybrid 抓回來的 20 筆 (10+10) 進行精準排序
#     pairs = [[query, doc.page_content] for doc in initial_docs]
#     scores = reranker_model.predict(pairs)
#     scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
#     top_3_docs = [doc for doc, score in scored_docs[:3]]
#
#     # 3. Generation (使用 llm_generator / 12B)
#     context_text = "\n\n".join([doc.page_content for doc in top_3_docs])
#     print(f"🤖 球員 (12B) 生成回答中...")
#
#     chain = rag_prompt | llm_generator | StrOutputParser()
#     answer = chain.invoke({"context": context_text, "question": query})
#     print(f"💬 回答: {answer.strip()}")
#
#     # 4. Evaluation (使用 llm_judge / 27B)
#     print("-" * 50)
#     print(f"⚖️ 裁判 (27B) 評分中...")
#     f_score, r_score = calculate_ragas_score(query, answer, context_text)
#
#     print(f"📊 評分報告:")
#     print(f"   ➤ Faithfulness (無幻覺): {f_score:.2f} / 1.0")
#     print(f"   ➤ Relevance (切題度):   {r_score:.2f} / 1.0")
#
#     if f_score < 0.8:
#         print("   ⚠️  警示：球員可能產生幻覺！")
#     elif r_score < 0.5:
#         print("   ⚠️  警示：球員答非所問！")
#     else:
#         print("   ✅ Pass：表現優良。")
#     print("=" * 60)
#
#
# # ==========================================
# # 📝 10 題完整測試題庫 (簡單、陷阱、否定)
# # ==========================================
# questions = [
#     # --- 簡單題 ---
#     "請問富邦尊御世界卡的正卡年費是多少錢？附卡要年費嗎？",
#     "我要預約連假期間的『國內機場接送』，最晚需要在幾個工作天前預約？",
#     "請問道路救援服務專線的電話號碼是多少？",
#
#     # --- 陷阱題 (考驗 12B 的邏輯) ---
#     "我上週剛買了機票，金額是 12,000 元，我是富邦世界卡的卡友（非理財會員），請問我可以預約免費機場接送嗎？",
#     "我用尊御世界卡刷了機票，但是是在 7 個月前（約 210 天前）刷的，現在要出國可以用機場外圍停車嗎？",
#     "我是富邦無限卡持卡人，我兒子今年 26 歲未婚，跟我一起出國，我幫他刷了全額機票，請問他有旅遊平安險的保障嗎？",
#     "我持有富邦世界卡，上期帳單一般消費 18,000 元，請問我去『台灣聯通』停車可以免費停幾小時？",
#
#     # --- 否定題 (排除條款) ---
#     "我為了湊免年費的門檻，去全聯福利中心買了很多東西，請問這些消費算在『一般消費』裡面嗎？",
#     "我剛買的新手機被偷了，可以用信用卡的『全球購物保障』申請理賠嗎？",
#     "我的車有改裝過，底盤比較低（離地 15 公分），車子拋錨了可以使用免費道路救援拖吊嗎？"
# ]
#
# # 執行測試
# for q in questions:
#     run_rag_with_evaluation(q)
#     time.sleep(3)
#
# # 清理
# vectorstore.delete_collection()
# print("\n✅ 所有測試完成！")
import os
import sys  # 用於中斷程式
import time
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# 🟢 新增：混合檢索需要的模組
# 混合檢索需要這兩個
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# 🔍 RAG 工程師的除錯檢查：確認 Key 是否存在
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ 錯誤：找不到 GOOGLE_API_KEY！")
    print("   請確認目錄下是否有 .env 檔案，且內容包含 GOOGLE_API_KEY=AIza...")
    sys.exit(1) # 直接停止，避免後面報錯


# ==========================================
# 🛠️ 設定區：雙模型架構 (Player vs Judge)
# ==========================================
file_path = "fubon.pdf"

# 🟢 球員：負責回答問題 (輕量級)
GENERATOR_MODEL = "gemma-3-12b-it"

# 🟢 裁判：負責評分 (重量級，邏輯更強)
JUDGE_MODEL = "gemma-3-27b-it"

print(f"🚀 啟動 RAG 系統 (Level 5 - Hybrid Search 混合檢索版)...")
print(f"🏃 球員模型 (Generator): {GENERATOR_MODEL}")
print(f"👨‍⚖️ 裁判模型 (Judge):     {JUDGE_MODEL}")
print(f"📄 處理檔案: {file_path}")


# ==========================================
# 1. 資料處理
# ==========================================
def load_and_split_pdf(file_path):
    print("   [系統提示] PDFPlumber 解析中...")
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    return splits


# ==========================================
# 2. 載入檢索模型
# ==========================================
print("   [系統提示] 載入 Embedding & Cross-Encoder...")
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
reranker_model = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

# ==========================================
# 3. 建立 Hybrid Retriever (關鍵修改區) 🟢
# ==========================================
pdf_splits = load_and_split_pdf(file_path)

# --- A. 建立向量檢索 (Vector Search) ---
# 擅長：語意理解 (例如知道「費用」跟「錢」有關)
print("   [系統提示] 建立 Chroma 向量索引...")
vectorstore = Chroma.from_documents(
    documents=pdf_splits,
    embedding=embedding_model,
    collection_name="fubon_hybrid_final"  # 改名避免衝突
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# --- B. 建立關鍵字檢索 (Keyword Search - BM25) ---
# 擅長：精確匹配 (例如精準抓到 "180天", "尊御世界卡", "20,000" 這些字)
print("   [系統提示] 建立 BM25 關鍵字索引...")
bm25_retriever = BM25Retriever.from_documents(pdf_splits)
bm25_retriever.k = 10  # 讓它也抓 10 筆

# --- C. 融合 (Ensemble) ---
# 將兩者的結果結合，權重各半 (0.5/0.5)
print("   [系統提示] 啟動 Hybrid Ensemble (混合檢索)...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# 將最終的檢索器設定為混合檢索器
retriever = ensemble_retriever

# ==========================================
# 4. 設定 LLM (雙模型實例化)
# ==========================================
print("   [系統提示] 初始化 LLM 模型中...")

try:
    # ✅ 修改點：顯式傳入 google_api_key，避免觸發 DefaultCredentialsError
    llm_generator = ChatGoogleGenerativeAI(
        model=GENERATOR_MODEL,
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY  # 強制指定 Key
    )

    llm_judge = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY  # 強制指定 Key
    )
except Exception as e:
    print(f"❌ 模型初始化失敗: {e}")
    sys.exit(1)


# RAG 回答 Prompt
# ==========================================
# 💎 工程級 Prompt Template (V4.2 - Fix Data Leakage)
# ==========================================
rag_template = """# Role
你是一位專業、邏輯嚴謹的「台北富邦銀行頂級卡權益審核專員」。你的任務是根據 <context> 準確回答客戶問題。

# Task
請閱讀 <context>，並針對 <question> 進行資格審核與回覆。

# Constraints
1. **嚴格引用**：回答必須基於 <context> 內容，回答結尾請標註來源。
2. **數值比對**：若問題涉及「金額」、「天數」、「年齡」，必須在思考過程中與手冊條款進行比對。
3. **排除條款**：若問題涉及「一般消費」，請特別檢查「排除項目」（如：全聯、稅款、學費）。
4. **誠實回答**：若 <context> 未提及，回答「手冊中未提及」。
5. **語言**：繁體中文（台灣）。

# Instruction (CoT)
在回答前，請務必先進行 <thinking> 步驟：
1. **識別變數**：用戶的卡別、身分、消費金額、時間點。
2. **查找條款**：在 <context> 中找到對應規則。
3. **邏輯判定**：
   - 資格檢查：用戶金額 vs 門檻？(例如：8000 < 10000 -> 不合格)
   - 期限檢查：天數 vs 限制？(例如：100天 < 90天 -> 過期)
   - 排除檢查：是否在排除名單？
4. **生成回答**：根據判定結果回覆。

# Few-Shot Examples (已修正：使用非測試集的案例)
<example>
User: 我上個月刷了 5 萬元，可以免費使用機場貴賓室嗎？
Context: ...使用機場貴賓室需支付費用，惟若使用前 90 天內刷卡購買機票或團費單筆達 NT$20,000 以上，可享免費用乙次...
Answer:
<thinking>
1. 用戶變數：消費 50,000 元（一般消費，非機票）。
2. 條款限制：需刷「機票或團費」且單筆達 NT$20,000。
3. 判定：雖然 50,000 > 20,000，但用戶未說明是用於「機票/團費」，且條款指定類別。若假設為一般消費則不符。
</thinking>
根據手冊規定，免費使用機場貴賓室的條件為「刷付機票或團費」單筆達 NT$20,000。若您的 5 萬元消費並非用於機票或團費，恐無法享有免費資格。
</example>

<example>
User: 我要繳小孩的學費，這筆錢算一般消費嗎？可以集紅利嗎？
Context: ...一般消費定義...不包含：基金、稅款、學費、公用事業費...
Answer:
<thinking>
1. 用戶變數：繳學費。
2. 條款限制：一般消費排除項目包含「學費」。
3. 判定：學費屬於排除項目 -> 不算一般消費 -> 不給紅利。
</thinking>
根據一般消費定義，學費屬於「排除項目」，因此不算在一般消費內，也無法計算紅利點數。
</example>

# Context
{context}

# Question
{question}

Answer:"""

rag_prompt = PromptTemplate.from_template(rag_template)


# ==========================================
# 🟢 RAGAS 評分邏輯 (使用 llm_judge / 27B)
# ==========================================
def calculate_ragas_score(question, answer, contexts):
    # Faithfulness (無幻覺)
    faithfulness_prompt = PromptTemplate.from_template("""
    你是一位嚴格的 RAG 評測員。
    請檢查「AI 回答」是否包含「參考片段」中沒有的幻覺資訊。

    【參考片段】：
    {contexts}

    【AI 回答】：
    {answer}

    請回傳一個 0.0 到 1.0 的分數 (1.0 代表完全忠實於原文，無幻覺)。
    只回傳數字，不要有其他文字。
    """)

    # Relevance (切題度)
    relevance_prompt = PromptTemplate.from_template("""
    你是一位嚴格的 RAG 評測員。
    請評分「AI 回答」是否精準回答了「用戶問題」，且沒有答非所問。

    【用戶問題】：
    {question}

    【AI 回答】：
    {answer}

    請回傳一個 0.0 到 1.0 的分數 (1.0 代表非常切題)。
    只回傳數字，不要有其他文字。
    """)

    try:
        # 🔴 關鍵：這裡使用 llm_judge (27B) 來執行評分
        f_chain = faithfulness_prompt | llm_judge | StrOutputParser()
        r_chain = relevance_prompt | llm_judge | StrOutputParser()

        f_str = f_chain.invoke({"contexts": contexts, "answer": answer}).strip()
        r_str = r_chain.invoke({"question": question, "answer": answer}).strip()

        # 解析數字
        f_match = re.findall(r"[-+]?\d*\.\d+|\d+", f_str)
        r_match = re.findall(r"[-+]?\d*\.\d+|\d+", r_str)

        f_score = float(f_match[0]) if f_match else 0.0
        r_score = float(r_match[0]) if r_match else 0.0

        return min(f_score, 1.0), min(r_score, 1.0)

    except Exception as e:
        print(f"   [評分系統錯誤] {e}")
        return 0.0, 0.0


# ==========================================
# 主流程 (修改版：支援 Streaming)
# ==========================================
def run_rag_with_evaluation(query):
    print(f"\n❓ 測試問題: {query}")
    print("-" * 50)

    # 1. Recall (Hybrid Search)
    # 這裡的 invoke 會同時跑 Vector 和 BM25，然後混合結果
    initial_docs = retriever.invoke(query)

    # 2. Re-rank (Cross-Encoder 決選)
    # 將 Hybrid 抓回來的 20 筆 (10+10) 進行精準排序
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker_model.predict(pairs)
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_3_docs = [doc for doc, score in scored_docs[:3]]

    # 3. Generation (使用 llm_generator / 12B)
    context_text = "\n\n".join([doc.page_content for doc in top_3_docs])
    print(f"🤖 球員 (12B) 生成回答中 (Streaming)...")
    print("-" * 20)  # 分隔線，準備開始打字

    chain = rag_prompt | llm_generator | StrOutputParser()

    # --- 🟢 修改開始：從 invoke 改成 stream ---
    full_answer = ""  # 用來收集完整的回答，給裁判評分用

    # 使用 chain.stream 進行串流
    for chunk in chain.stream({"context": context_text, "question": query}):
        # end="" 讓它不要換行，flush=True 確保字會馬上跳出來
        print(chunk, end="", flush=True)
        full_answer += chunk

    print()  # 最後印一個換行
    # --- 🟢 修改結束 ---

    # 4. Evaluation (使用 llm_judge / 27B)
    # 裁判必須等球員講完話才能評分，所以這裡不能 Stream
    print("-" * 50)
    print(f"⚖️ 裁判 (27B) 評分中...")

    # 注意：這裡把收集好的 full_answer 傳進去
    f_score, r_score = calculate_ragas_score(query, full_answer, context_text)

    print(f"📊 評分報告:")
    print(f"   ➤ Faithfulness (無幻覺): {f_score:.2f} / 1.0")
    print(f"   ➤ Relevance (切題度):   {r_score:.2f} / 1.0")

    if f_score < 0.8:
        print("   ⚠️  警示：球員可能產生幻覺！")
    elif r_score < 0.5:
        print("   ⚠️  警示：球員答非所問！")
    else:
        print("   ✅ Pass：表現優良。")
    print("=" * 60)


# ==========================================
# 📝 10 題完整測試題庫 (簡單、陷阱、否定)
# ==========================================
questions = [
    # --- 簡單題 ---
    "請問富邦尊御世界卡的正卡年費是多少錢？附卡要年費嗎？",
    "我要預約連假期間的『國內機場接送』，最晚需要在幾個工作天前預約？",
    "請問道路救援服務專線的電話號碼是多少？",

    # --- 陷阱題 (考驗 12B 的邏輯) ---
    "我上週剛買了機票，金額是 12,000 元，我是富邦世界卡的卡友（非理財會員），請問我可以預約免費機場接送嗎？",
    "我用尊御世界卡刷了機票，但是是在 7 個月前（約 210 天前）刷的，現在要出國可以用機場外圍停車嗎？",
    "我是富邦無限卡持卡人，我兒子今年 26 歲未婚，跟我一起出國，我幫他刷了全額機票，請問他有旅遊平安險的保障嗎？",
    "我持有富邦世界卡，上期帳單一般消費 18,000 元，請問我去『台灣聯通』停車可以免費停幾小時？",

    # --- 否定題 (排除條款) ---
    "我為了湊免年費的門檻，去全聯福利中心買了很多東西，請問這些消費算在『一般消費』裡面嗎？",
    "我剛買的新手機被偷了，可以用信用卡的『全球購物保障』申請理賠嗎？",
    "我的車有改裝過，底盤比較低（離地 15 公分），車子拋錨了可以使用免費道路救援拖吊嗎？"
]

# 執行測試
for q in questions:
    run_rag_with_evaluation(q)
    time.sleep(3)

# 清理
vectorstore.delete_collection()
print("\n✅ 所有測試完成！")

