import streamlit as st
from model import RAGModel
import time

# ==========================================
# [Controller] 初始化與快取
# ==========================================
st.set_page_config(page_title="富邦權益 RAG 助手", page_icon="🤖")

@st.cache_resource
def load_model():
    return RAGModel()

try:
    rag_engine = load_model()
except Exception as e:
    st.error(f"模型載入失敗: {e}")
    st.stop()

# ==========================================
# ✅ [新增] 初始化狀態鎖
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False  # 預設為「非處理中」

# ==========================================
# [View] 頁面佈局
# ==========================================
st.title("💳 台北富邦銀行權益審核助手")
st.caption("MVC 架構展示 | Hybrid Search + Rerank | Gemma-3-27b")

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # ✅ [新增] 如果歷史紀錄中有 'sources' 欄位，就顯示出來
        if "sources" in message:
            with st.expander("查看參考來源 (Evidence)"):
                for src in message["sources"]:
                    st.markdown(f"**來源**: {src['source']} (Page {src['page']})")
                    st.text(src['content'])
# ==========================================
# ✅ [修改] 輸入框與處理邏輯
# ==========================================

# 1. 顯示輸入框 (透過 processing 變數控制 disabled 狀態)
# 當 processing 為 True 時，輸入框會變灰，無法輸入
prompt = st.chat_input(
    "請輸入客戶問題 (例如：尊御世界卡年費多少？)", 
    disabled=st.session_state.processing
)

# 2. 如果收到輸入，鎖定狀態並刷新
if prompt:
    # 存入使用者問題
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 開啟鎖定
    st.session_state.processing = True
    # 強制重跑一遍，讓輸入框立刻變灰
    st.rerun()

# 3. 如果處於「處理中」狀態，執行後端邏輯
if st.session_state.processing:
    # 取得最後一則使用者訊息 (因為 refresh 後 prompt 變數會清空，要從 history 拿)
    last_user_message = st.session_state.messages[-1]["content"]
    
    # with st.chat_message("user"):
    #     st.markdown(last_user_message)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 使用 st.status 讓使用者知道現在進度 (比 spinner 更好看)
        with st.status("正在檢索資料與思考中...", expanded=True) as status:
            try:
                st.write("🔍 檢索相關條款...")
                # 呼叫模型
                stream, source_docs = rag_engine.get_answer(last_user_message)
                
                st.write("🧠 進行邏輯推演...")
                
                # 串流輸出
                for chunk in stream:
                    # 一旦開始有 output，就可以把狀態收起來
                    status.update(label="思考完成", state="complete", expanded=False)
                    
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # 顯示來源
                saved_sources = []
                for doc in source_docs:
                    saved_sources.append({
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "content": doc.page_content[:200] + "..."
                    })

                # ✅ [修改] 顯示來源 (這是給當下這一輪看的，保持不變)
                with st.expander("查看參考來源 (Evidence)"):
                    for src in saved_sources:
                        st.markdown(f"**來源**: {src['source']} (Page {src['page']})")
                        st.text(src['content'])
                
                # ✅ [關鍵修改] 存入歷史紀錄時，多存一個 "sources" 欄位
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": saved_sources  # 把處理好的來源存進去
                })
                
            except Exception as e:
                st.error(f"發生錯誤: {e}")
            
            finally:
                # ✅ 關鍵：無論成功或失敗，最後都要解鎖
                st.session_state.processing = False
                # 再次刷新，讓輸入框變回可輸入
                st.rerun()