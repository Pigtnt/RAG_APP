import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 匯入設定與 Prompts
from config import Config
from prompts import rag_prompt, faithfulness_prompt, relevance_prompt

class RAGModel:
    def __init__(self):
        print("🔧 [Model] 初始化 RAG 引擎...")
        
        # --- 1. Embedding & Database ---
        # [Core] 向量模型 (使用 CPU)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # [Core] 向量資料庫連線
        self.vector_store = Chroma(
            persist_directory=Config.DB_PATH,
            embedding_function=self.embedding_model,
            collection_name=Config.COLLECTION_NAME
        )
        
        # --- 2. Retrieval Strategy (Hybrid) ---
        print("📚 [Model] 重建索引與 Re-ranker...")
        all_docs = self.vector_store.get()["documents"] 
        doc_objects = [Document(page_content=t) for t in all_docs]
        
        # [Param] 初步檢索數量 (k=10)
        chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(doc_objects)
        bm25_retriever.k = 10
        
        # [Param] 混合檢索權重 (Vector=0.5, Keyword=0.5)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # --- 3. Refinement & Generation ---
        # [Core] Re-ranker 模型
        self.reranker = CrossEncoder(Config.RERANKER_MODEL, device='cpu')
        
        # [Param] 回答生成模型 (Temperature=0.1 降低隨機性)
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GENERATOR_MODEL, 
            temperature=0.1, 
            google_api_key=Config.API_KEY
        )

        # [Param] 評分裁判模型 (Temperature=0.0 要求絕對理性)
        print(f"⚖️ [Model] 初始化裁判模型 ({Config.JUDGE_MODEL})...")
        self.judge_llm = ChatGoogleGenerativeAI(
            model=Config.JUDGE_MODEL, 
            temperature=0.0, 
            google_api_key=Config.API_KEY
        )

    def get_answer(self, query):
        """
        [Logic] RAG 主流程：檢索 -> 重排序 -> 上下文組裝 -> 生成 (串流)
        """
        
        # Step 1: 初步混合檢索 (Recall)
        initial_docs = self.ensemble_retriever.invoke(query)
        
        # Step 2: 精確重排序 (Rerank)
        # [Logic] 計算 (Query, Doc) 相似度分數並排序
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # [Param] 最終選取 Top-3 文件進入 Prompt
        top_docs = [doc for doc, score in scored_docs[:3]]
        
        # Step 3: 組裝 Context
        context = "\n\n".join([doc.page_content for doc in top_docs])
        
        # Step 4: 生成回答 (Streaming)
        chain = rag_prompt | self.llm | StrOutputParser()
        original_stream = chain.stream({"context": context, "question": query})

        # [Logic] 串流過濾器：移除 <thinking> 標籤，只回傳最終答案
        def clean_stream(stream):
            buffer = ""
            thinking_ended = False
            
            print("\n🤖 [LLM Raw Output Start] ------------------") # Debug
            
            for chunk in stream:
                print(chunk, end="", flush=True) # Debug: 即時印出

                if thinking_ended:
                    yield chunk
                    continue
                
                buffer += chunk
                
                # 偵測思考結束標籤
                if "</thinking>" in buffer:
                    thinking_ended = True
                    parts = buffer.split("</thinking>")
                    real_answer = parts[-1] 
                    if real_answer: yield real_answer
                    buffer = "" 

            print("\n🤖 [LLM Raw Output End] --------------------") # Debug

            # [Safety Net] 若模型未輸出結束標籤，則強行輸出所有內容防止空白
            if not thinking_ended and buffer:
                print("\n⚠️ 警告: 未偵測到 </thinking> 標籤，直接輸出所有內容。")
                yield buffer

        return clean_stream(original_stream), top_docs

    def calculate_score(self, question, answer, source_docs):
        """
        [Logic] RAGAS Lite 評分機制
        Returns: (Faithfulness Score, Relevance Score)
        """
        print(f"⚖️ [Model] 啟動裁判模型 ({Config.JUDGE_MODEL}) 進行評分...")
        contexts = "\n\n".join([doc.page_content for doc in source_docs])
        
        try:
            # 建構評分 Chain
            f_chain = faithfulness_prompt | self.judge_llm | StrOutputParser()
            r_chain = relevance_prompt | self.judge_llm | StrOutputParser()

            # [Logic] 同步執行評分 (會阻塞直到 LLM 回傳)
            f_str = f_chain.invoke({"contexts": contexts, "answer": answer}).strip()
            r_str = r_chain.invoke({"question": question, "answer": answer}).strip()

            # [Logic] 使用 Regex 提取數值 (防止 LLM 輸出多餘文字)
            f_match = re.findall(r"[-+]?\d*\.\d+|\d+", f_str)
            r_match = re.findall(r"[-+]?\d*\.\d+|\d+", r_str)

            f_score = float(f_match[0]) if f_match else 0.0
            r_score = float(r_match[0]) if r_match else 0.0
            
            return f_score, r_score
            
        except Exception as e:
            print(f"⚠️ 評分過程發生錯誤: {e}")
            return 0.0, 0.0