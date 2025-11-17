from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import os
import cv2
import pandas as pd
import re

# Define the image transformation
def preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # 常態分布 (RGB:0-255)
    ])
    
# Custom Dataset for fine-tuning
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, transform=None): # transform including resize, crop,..etc
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
        
    def __len__(self):return len(self.image_paths)
    # magic method)，用來讓物件支援像「索引」一樣的存取行為，例如 obj[index]
    def __getitem__(self, idx): 
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        if self.transform:
            image = self.transform(image)
        return image, text

    # consider ViT-B/32 with input 224x224
    def resize_image(self, image:Image.Image, size = (224,224)): 
        return image.resize(size)
    
    # 拼接圖片
    def image_collage(self, out_dir="./result", thumb_width:int=300,  thumb_heigh:int=300):
        thumb_size = (thumb_width, thumb_heigh)
        num_thumb = 2
        # 每四張拼接一次
        for batch_idx in range(0, len(self.image_paths), num_thumb**2):
            batch = self.image_paths[batch_idx : batch_idx + num_thumb**2]
            bg = Image.new("RGB", (thumb_size[0] * num_thumb, thumb_size[1] * num_thumb), "#000000")
            for j, path in enumerate(batch):
                img = Image.open(path).convert("RGB")
                resize_img = self.resize_image(img, thumb_size)  # 假設 resize_image 回傳 PIL Image
                x = (j % 2) * thumb_size[0]
                y = (j // 2) * thumb_size[1]
                bg.paste(resize_img, (x, y))
            out_path = os.path.join(out_dir, f"concat_{batch_idx//4}.jpg")
            bg.save(out_path)
            print("saving collage successfully!")
    
    def image_overlay(self):
        pass

# 將數據整理成一張圖像對三種不同 caption JSON 格式
def aggregate_image_captions(image_path:str, captions:str):
    """
    Inputs:
        captions = "1. a man riding a bike\n2. a cyclist wearing a helmet\n3. an outdoor biking scene"
    Returns:
        images = ["img.jpg", "img.jpg", "img.jpg"]
        captions_lst = [
            "a man riding a bike",
            "a cyclist wearing a helmet",
            "an outdoor biking scene"
        ]
    """
    # 使用正則表達式“切分”，依據換行符號或數字標號 (1. 2. 3.)，被切分的部分可能留下 ''
    split_captions = re.split(r'(?:\d+\s*[\.\)]\s*|\n+)', captions)
    # 去除空白與空字串
    captions_lst = [cap.strip() for cap in split_captions if cap.strip()]
    # 複製 image_path，數量與 captions 一致
    images = [image_path] * len(captions_lst)
    return images, captions_lst


# 移除caption中重複的文字
def deduplicate_contxt(text: str, sep: str = "，"):
    """
    精確句子去重
    :param text: 原始文字
    :param sep: 切分符號，默認是 "。" (適用中文)
    :return: 去重後的文字
    """
    # 切分句子
    sentences = text.split(sep)
    # 去掉空字串並保留順序去重
    unique_sentences = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))
    # 拼回文字
    return sep.join(unique_sentences) + (sep if text.endswith(sep) else "")


if __name__=="__main__":
    # command "yolo settings" 可查看目前參數設定
    # crop_objects("origin_dataset/13274_632087950224364_8692510268907875417_n(1).jpg")
    df=pd.read_csv("captions_0924.csv")