import os
import glob
import streamlit as st
from datetime import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer, util
"""
設計一個名為“Memory track"的圖像搜尋引擎，以下為設計參考準則：
## 輸出風格：具有酷炫感、科技感之配色
## 功能排版：
## 功能：
- 可批量上傳檔案包含（jpg, png, jpeg)
- 

## 使用框架： python streamlit

"""

# ====== 基本設定 ======
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 模型 (CLIP 用於文字-圖片向量檢索)
model = SentenceTransformer("clip-ViT-B-32")

# 全域圖片索引
image_index = []
image_paths = []


# ====== 輔助函式 ======
def save_uploaded_files(uploaded_files):
    """批量儲存上傳圖片到本地"""
    paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        paths.append(file_path)
    return paths


def build_image_index():
    """建立圖片索引 (向量化)"""
    global image_index, image_paths
    image_paths = glob.glob(os.path.join(UPLOAD_DIR, "*"))
    if not image_paths:
        return
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_index = model.encode(images, convert_to_tensor=True, show_progress_bar=True)


def search_images(query, top_k=3, date_filter=None):
    """以文字搜尋圖片"""
    if not image_index:
        return []

    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, image_index, top_k=len(image_paths))[0]

    results = []
    for h in hits:
        img_path = image_paths[h["corpus_id"]]

        # 日期篩選
        if date_filter:
            file_time = datetime.fromtimestamp(os.path.getmtime(img_path))
            if file_time.date() < date_filter:
                continue

        results.append(img_path)
        if len(results) >= top_k:
            break
    return results


# ====== Streamlit 介面 ======
st.set_page_config(page_title="智能文字搜圖系統", layout="wide")

# CSS 科技美感 + SVG icon + 按鈕發光
st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 0 0px #58a6ff;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #58a6ff;
        transform: scale(1.05);
    }
    .upload-box {
        border: 2px dashed #58a6ff;
        padding: 20px;
        border-radius: 12px;
        background-color: #161b22;
    }
    .svg-icon {
        width: 24px;
        height: 24px;
        margin-right: 8px;
        vertical-align: middle;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 智能文字搜圖系統")

# ====== 上傳檔案區 ======
st.subheader("📂 批量上傳圖片")
uploaded_files = st.file_uploader("拖曳或選擇圖片檔案", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    saved = save_uploaded_files(uploaded_files)
    st.success(f"成功上傳 {len(saved)} 個檔案 ✅")
    build_image_index()

# ====== 搜尋區 ======
st.subheader("🔎 文字檢索")
query = st.text_input("請輸入搜尋關鍵字")

col1, col2 = st.columns(2)
with col1:
    top_k = st.number_input("返回圖片數量", min_value=1, max_value=10, value=3, step=1)
with col2:
    date_filter = st.date_input("檔案日期篩選 (僅顯示此日期之後的圖片)", value=None)

if st.button("🚀 開始搜尋"):
    results = search_images(query, top_k, date_filter if date_filter else None)
    if results:
        st.success(f"找到 {len(results)} 張相關圖片")
        cols = st.columns(len(results))
        for i, img_path in enumerate(results):
            with cols[i]:
                st.image(img_path, use_column_width=True)
                st.caption(os.path.basename(img_path))
    else:
        st.warning("未找到符合的圖片 😢")
