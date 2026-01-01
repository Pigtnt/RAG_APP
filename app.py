import streamlit as st
from model import RAGModel
import time

# --- System Setup ---
# [Config] 頁面標題與圖示
st.set_page_config(page_title="富邦權益 RAG 助手", page_icon="🤖")

@st.cache_resource
def load_model():
    return RAGModel()

try:
    rag_engine = load_model()
except Exception as e:
    st.error(f"模型載入失敗: {e}")
    st.stop()

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# [Logic] 狀態鎖：控制輸入框是否停用 (True=鎖定, False=解鎖)
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- View: UI Layout ---
st.title("💳 台北富邦銀行權益審核助手")
st.caption("MVC 架構展示 | Hybrid Search + Rerank | Gemma-3-12b")

# [View] 渲染歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 若包含來源資料則顯示折疊區塊
        if "sources" in message:
            with st.expander("查看參考來源 (Evidence)"):
                for src in message["sources"]:
                    st.markdown(f"**來源**: {src['source']} (Page {src['page']})")
                    st.text(src['content'])

# --- Controller: Interaction Logic ---

# 1. Input Area
# [Logic] disabled參數綁定狀態鎖，防止重複提交
prompt = st.chat_input(
    "請輸入客戶問題 (例如：尊御世界卡年費多少？)", 
    disabled=st.session_state.processing
)

# 2. Trigger Event
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.processing = True # 上鎖
    st.rerun() # 強制刷新以更新 UI 狀態 (輸入框變灰)

# 3. Backend Execution (Locked State)
if st.session_state.processing:
    last_user_message = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        source_docs = []

        # [View] 進度狀態顯示
        with st.status("正在檢索資料與思考中...", expanded=True) as status:
            try:
                # === Phase 1: Retrieval & Generation ===
                st.write("🔍 檢索相關條款...")
                stream, source_docs = rag_engine.get_answer(last_user_message)
                
                st.write("🧠 進行邏輯推演...")
                for chunk in stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # [Logic] 儲存對話紀錄 (含來源)
                saved_sources = [{
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "content": doc.page_content[:200] + "..."
                } for doc in source_docs]
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": saved_sources
                })

                # === Phase 2: Evaluation (RAGAS) ===
                status.update(label="回答完成，正在進行品質評分...", state="running", expanded=True)
                
                if full_response and source_docs:
                    st.write("⚖️ 裁判模型閱卷中 (請查看 Terminal)...")
                    
                    # [Core] 呼叫評分模型 (同步執行，會阻塞 UI 直到完成)
                    f_score, r_score = rag_engine.calculate_score(
                        question=last_user_message,
                        answer=full_response,
                        source_docs=source_docs
                    )

                    # [Log] 輸出至 Terminal
                    print("\n" + "="*50)
                    print(f"❓ 問題: {last_user_message}")
                    print(f"🤖 回答: {full_response[:50]}...")
                    print("-" * 20)
                    print(f"📊 [RAGAS 評分報告] F={f_score:.2f}, R={r_score:.2f}")
                    
                    # [Param] 評分警示門檻 (可調整)
                    if f_score < 0.8: print("   ⚠️  警示：可能產生幻覺 (Hallucination)！")
                    elif r_score < 0.5: print("   ⚠️  警示：答非所問 (Irrelevant)！")
                    else: print("   ✅ Pass：表現優良。")
                    print("="*50 + "\n")

                    st.write(f"📊 評分完成 (F={f_score}, R={r_score})")

                status.update(label="所有程序執行完畢", state="complete", expanded=False)

            except Exception as e:
                st.error(f"發生錯誤: {e}")
                print(f"❌ Error: {e}")
            
            finally:
                # === Phase 3: Unlock ===
                # [Logic] 確保無論成功失敗，最後一定解鎖並刷新
                st.session_state.processing = False
                st.rerun()