import sys
import os

print("=== Python 搜尋路徑 (sys.path) ===")
for p in sys.path:
    print(p)

print("\n=== 嘗試匯入 langchain ===")
try:
    import langchain

    print(f"✅ 成功匯入 langchain！")
    print(f"📂 真實檔案位置: {langchain.__file__}")

    # 檢查是否有 retrievers
    if hasattr(langchain, 'retrievers'):
        print("✅ langchain.retrievers 存在！")
    else:
        print("❌ langchain.retrievers 不存在！(這就是問題所在)")
        print(f"   請檢查上面的「真實檔案位置」，它是不是指向你自己的資料夾？")

except ImportError as e:
    print(f"❌ 匯入失敗: {e}")
except Exception as e:
    print(f"❌ 發生其他錯誤: {e}")

print("\n=== 檢查當前目錄檔案 ===")
files = os.listdir(".")
for f in files:
    if "langchain" in f:
        print(f"⚠️ 發現可疑檔案/資料夾: {f}")