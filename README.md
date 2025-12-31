# RAG練習_MVC架構_網頁APP

# 安裝 uv 環境
```
uv: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
```

步驟1 : 同步專案環境並建立虛擬環境
```
uv sync
```

步驟2 : 建立 `.env` 檔案，取得並新增 `GEMINI_API_KEY` 至 `.env`

```.env
GEMINI_API_KEY = "你的GEMINI_API_KEY"
```

前往 `https://aistudio.google.com/api-keys` 取得你的GEMINI_API_KEY

步驟3 : 建立資料庫（執行 ingest 腳本）
```
uv run ingest.py
```

步驟4 : 啟動 APP（使用 streamlit）
```
uv run streamlit run app.py
```

步驟5 : 進入APP網頁
```
http://localhost:8501
```
