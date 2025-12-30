from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 匯入您的設定與 Prompt
from config import Config
from prompts import rag_prompt  # ✅ 從 prompts.py 匯入 Prompt Template

class RAGModel:
    def __init__(self):
        print("🔧 [Model] 初始化 RAG 引擎...")
        
        # 1. 準備 Embedding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. 載入 Chroma DB
        self.vector_store = Chroma(
            persist_directory=Config.DB_PATH,
            embedding_function=self.embedding_model,
            collection_name=Config.COLLECTION_NAME
        )
        
        # 3. 準備 Retriever (Hybrid)
        print("📚 [Model] 重建索引與 Re-ranker...")
        all_docs = self.vector_store.get()["documents"] 
        doc_objects = [Document(page_content=t) for t in all_docs]
        
        chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(doc_objects)
        bm25_retriever.k = 10
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # 4. Re-ranker
        self.reranker = CrossEncoder(Config.RERANKER_MODEL, device='cpu')
        
        # 5. LLM
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GENERATOR_MODEL, 
            temperature=0.1, 
            google_api_key=Config.API_KEY
        )

    def get_answer(self, query):
        """接收 query，回傳 (回答串流, 參考文件)"""
        
        # Step 1: 檢索
        initial_docs = self.ensemble_retriever.invoke(query)
        
        # Step 2: Rerank
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:3]]
        
        # 組裝 Context
        context = "\n\n".join([doc.page_content for doc in top_docs])
        
        # Step 3: 生成
        chain = rag_prompt | self.llm | StrOutputParser()
        original_stream = chain.stream({"context": context, "question": query})

        # ✅ 加入過濾邏輯：只回傳 </thinking> 之後的內容
        def clean_stream(stream):
            buffer = ""
            thinking_ended = False
            
            print("\n🤖 [LLM Raw Output Start] ------------------") # Debug 用
            
            for chunk in stream:
                # 1. Debug: 在終端機即時印出，確認後端有收到字
                print(chunk, end="", flush=True) 

                if thinking_ended:
                    yield chunk
                    continue
                
                buffer += chunk
                
                # 2. 偵測思考結束標籤
                if "</thinking>" in buffer:
                    thinking_ended = True
                    # 取出標籤後的真正回答
                    parts = buffer.split("</thinking>")
                    real_answer = parts[-1] 
                    
                    # 有時候標籤後緊接著換行，可以保留或 strip() 看您需求
                    if real_answer:
                        yield real_answer
                    
                    buffer = "" # 清空緩衝區，釋放記憶體

            print("\n🤖 [LLM Raw Output End] --------------------") # Debug 用

            # 3. 🔥 關鍵安全網 (Safety Net) 🔥
            # 如果串流結束了，但 thinking_ended 還是 False (代表模型沒乖乖輸出 </thinking>)
            # 此時必須把 buffer 裡的內容全部吐出來，不然使用者會看到空白！
            if not thinking_ended and buffer:
                print("\n⚠️ 警告: 未偵測到 </thinking> 標籤，直接輸出所有內容。")
                yield buffer

        # 回傳「過濾後」的串流
        return clean_stream(original_stream), top_docs