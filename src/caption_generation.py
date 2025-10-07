import os
import argparse
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
import csv
import pandas as pd

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GENERAL_PROMPT = "不需要任何開場白，直接完整描述你觀察到什麼,並不做額外猜測,輸出語系為繁體中文,輸出格式一致"
AGUMENTED_PROMPT = "提供三種多樣化的完整描述，且不需要任何開場白,並不做額外猜測,輸出語系為繁體中文" 

class VLMImageCaptionGenerator:
    """VLM圖片caption生成器"""
    
    def __init__(self, model_name: str = "qwen2.5vl:7b", device: Optional[str] = None):
        """
        初始化VLM模型
        
        Args:
            model_name: 模型名稱，預設使用 qwen2.5vl:7
            device: 設備類型 (cuda/cpu)，如果為None則自動選擇
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"正在載入模型: {model_name}")
        logger.info(f"使用設備: {self.device}")
        
        try:
            # self.model = model_name
            logger.info("模型載入成功!")
            
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        載入圖片
        
        Args:
            image_path: 圖片路徑或URL
            
        Returns:
            PIL Image對象
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                # 從URL載入圖片
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                # 從本地路徑載入圖片
                image = Image.open(image_path)
                image = self.resize_image(image)
                # print(image.size)
            
            # 轉換為RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            new_path = "./new_dataset/resized_" + os.path.basename(image_path)
            image.save(new_path)
            print("finished saving new picture!")
            return new_path
            
        except Exception as e:
            logger.error(f"圖片載入失敗 {image_path}: {e}")
            raise
    
    def resize_image(self, image:Image.Image, size = (512, 512)):
        return image.resize(size)
    
    def generate_caption(self, image_path: str, max_length:int=None) -> str:
        """
        為圖片生成caption
        
        Args:
            image_path: 圖片路徑或URL
            max_length: 生成caption的最大長度
            
        Returns:
            生成的caption文字
        """
        try:
            # 載入圖片並處理成 PIL 物件
            # new_image_path = self.load_image(image_path)
            # 處理圖片
            messages = ollama.chat(
                model=self.model_name, 
                messages=[
                    {
                        "role": "user",
                        "content": AGUMENTED_PROMPT,
                        "images": [image_path]
                    },
                ],
                options={
                    "num_predict": 1000,
                    "frequency_penalty": 0.6   
                }
            )
            # 解碼生成的文字
            caption = messages["message"]["content"]
            print(125, f"The image {image_path}:{caption}")
            return caption.strip()
            
        except Exception as e:
            logger.error(f"生成caption失敗: {e}")
            raise
    
    
    def generate_captions_batch(self, image_paths: List[str], max_length: int = 50, workers: int = 3) -> List[str]:
        """
        平行批量生成圖片 captions
        
        Args:
            image_paths: 圖片路徑列表
            max_length: 生成caption的最大長度
            workers: 平行 worker 數量（可調整
            
        Returns:
            caption列表
        """
        results = [None] * len(image_paths)
        
        def process(idx, path):
            try:
                logger.info(f"開始處理 {len(image_paths)} 張圖片")
                print(f"正在處理第 {idx+1} / {len(image_paths)}張圖片")
                return idx, self.generate_caption(path)
            except Exception as e:
                logger.error(f"處理圖片 {path} 的失敗: {e}")
                return idx, None
            
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(process, i, p): i for i, p in enumerate(image_paths)}
            for future in as_completed(future_to_idx):
                idx, caption = future.result()
                results[idx] = caption
        return results
    
    def save_captions_to_file(self, image_paths: List[str], captions: List[str], output_file: str):
        """
        將生成的caption保存到文件
        
        Args:
            image_paths: 圖片路徑列表
            captions: caption列表
            output_file: 輸出文件路徑
        """
        try:
            with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL) # quoting=csv.QUOTE_ALL 自動處理換行
                # 若檔案是新建的才寫表頭
                if f.tell() == 0:
                    writer.writerow(['image_path', 'Caption'])
                for image_path, caption in zip(image_paths, captions):
                    writer.writerow([image_path, caption])
            logger.info(f"Caption已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存文件失敗: {e}")
            raise
        
def main():
    parser = argparse.ArgumentParser(description="VLM圖片Caption生成器")
    parser.add_argument("--image_path", type=str, help="單張圖片路徑或URL")
    parser.add_argument("--image_dir", type=str, help="圖片目錄路徑")
    parser.add_argument("--output_file", type=str, default="captions_0924.csv", help="輸出文件路徑")
    parser.add_argument("--model_name", type=str, default="qwen2.5vl:7b", help="VLM模型名稱")
    parser.add_argument("--max_length", type=int, default=1000, help="生成caption的最大長度")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="設備類型")
    args = parser.parse_args()
    # 檢查參數
    if not args.image_path and not args.image_dir:
        parser.error("請提供 --image_path 或 --image_dir 參數")
    
    try:
        # 初始化生成器
        generator = VLMImageCaptionGenerator(
            model_name=args.model_name,
            device=args.device
        )
        
        image_paths = []
        if args.image_dir:
                # 處理目錄中的圖片
                image_dir = Path(args.image_dir)
                if not image_dir.exists():
                    raise FileNotFoundError(f"目錄不存在: {args.image_dir}")
                
                # 支持的圖片格式
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
                image_paths = [
                    str(f) for f in image_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
                
                if not image_paths:
                    print(f"在目錄 {args.image_dir} 中未找到圖片文件")
                    return
                
                print(f"找到 {len(image_paths)} 張圖片")
                
                # 批量生成caption
                captions = generator.generate_captions_batch(image_paths, args.max_length)
                print(235, len(captions))
                # 保存結果
                generator.save_captions_to_file(image_paths, captions, args.output_file)
                # 顯示結果
                print("\n生成的Caption:")
                print("-" * 50)
                for image_path, caption in zip(image_paths, captions):
                    print(f"{Path(image_path).name}: {caption}")
    except Exception as e:
        logger.error(f"程序執行失敗: {e}")
        return 1
    return 0

# 將 open-source data 翻譯為繁體中文
def fast_translator(text):
    # 這裡使用換成 API
    return f"{text} (中文翻譯)"

def open_data_optimization(
        file_txt = "captions.txt", 
        file_csv = "open_caption_data.csv",
        origin_csv = "captions_0917.csv",
        file_merged = "merged_caption.csv",
        translate=True
    ):
    # 逐行讀取 image/ caption
    with open(file_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:] # 從第 2 行開始
        
    with open(file_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","Caption"])  # 寫入標題列
        for line in lines:
            # 去掉換行符號，再用第一個逗號分隔
            parts = line.strip().split(",", 1)  
            if len(parts) == 2:
                image, caption = parts
                writer.writerow([image, caption])
                
    # Step 1: TXT → DataFrame
    
    # Step 2: 合併
    df_origin = pd.read_csv(origin_csv)
    df_merged = pd.concat([df_origin, df_new], ignore_index=True, axis=0)
    # Step 3: 輸出最終的 image / caption 資料
    df_merged.to_csv(file_merged, index=False, encoding="utf-8")
    print(f"已輸出合併後 CSV: {file_merged}")
    
    
if __name__ == "__main__":
    exit(main()) 
    