# ====================== text to search images ======================
import numpy as np
import os, torch, clip
from PIL import Image
import glob
import matplotlib.pyplot as plt
import faiss


model_dir = "./results/clip_flickr8k_ft.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("Loaing model successfully!")
model.eval()

def encode_images(img_list):
    feats = []
    with torch.no_grad():
        for p in img_list:
            img = clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f   = model.encode_image(img)
            feats.append(f.squeeze().cpu().numpy())
    feats = np.stack(feats)
    return feats / np.linalg.norm(feats, axis=1, keepdims=True) # 正規化


def create_vector_index(img_feats):
    # 確保 img_feats 是 float32
    img_feats = img_feats.astype("float32")
    # 正規化向量 (確保與 txt_feats 正規化分是一致)
    faiss.normalize_L2(img_feats)
    # 建立 index (L2 距離) 
    index = faiss.IndexFlatIP(img_feats.shape[1])  # IP = Inner Product (可近似 cosine)
    index.add(img_feats)  # 加入所有圖片向量
    print("Index total vectors:", index.ntotal)
    return index


def search(query, img_list, img_feats, index=None, topk=5):
    with torch.no_grad():
        txt_feat = model.encode_text(clip.tokenize([query]).to(device))
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    sims = (txt_feat.cpu().numpy() @ img_feats.T).squeeze()
    # the (image, text) threshold score
    idxs = sims.argsort()[::-1][:topk] 
    return [(img_list[i], sims[i]) for i in idxs]
    # 使用 faiss index 搜尋
    # txt_feat = txt_feat.cpu().numpy().astype("float32")
    # sims, idxs = index.search(txt_feat, topk)
    # return [(img_list[i], sims[0][j]) for j, i in enumerate(idxs[0])]


def display_top_images(results:list):
    """
    Args:
        results (list): 搜尋到的所有圖片列表
    """
    # 設定顯示排版
    n = len(results)
    cols = 5
    rows = n // cols + 1 if n % cols != 0 else n // cols # 自動算需要幾列
    plt.figure(figsize=(15, 5 * rows))
    for i, (path, score) in enumerate(results, 1):
        filename = path.split("/")[-1]
        img = Image.open(path)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{filename}\nscore={score:.3f}", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    folder = "test_dataset" # Use only the first 1000 images for indexing demo.
    sample_imgs = glob.glob(os.path.join(folder, "*.[jp][pn]g"))  
    sample_imgs = [i for i in sample_imgs]
    sample_feats  = encode_images(sample_imgs)
    # 加入 faiss 搜尋索引
    #faiss_index = create_vector_index(sample_feats)
    query = "一個穿綠色服小男孩躺在地上微笑"
    search_results = search(query, sample_imgs, sample_feats, topk=30)
    for p, s in search_results:
        print(f"Here's the related image {p.split('/')[-1]:30s}, similarity score={s:.3f}")
    # 顯示所有圖片
    display_top_images(search_results)
    # Recall@K 評估準確率
    
    