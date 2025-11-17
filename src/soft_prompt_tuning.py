"""
使用 soft prompt 技術來 fine-tune 視覺語言模型。
Soft prompt 是一種參數高效的 fine-tuning 方法，只訓練可學習的 prompt tokens，
而不修改原始模型的參數。

主要概念：
1. Soft Prompt: 可學習的連續向量，作為模型的輸入
2. 凍結原始模型: 保持預訓練權重不變
3. 只訓練 prompt: 大大減少可訓練參數數量
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPProcessor
)
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import os
import json
from preprocess_data import *
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

saving_path = Path("soft_prompt_results")
saving_path.mkdir(parents=True, exist_ok=True)

# 1. 定義 Soft Prompt 類別
class SoftPrompt(nn.Module):
    """
    Soft Prompt 模組
    這個模組創建可學習的連續向量，作為模型的輸入 prompt。
    這些向量會在訓練過程中更新，而原始模型保持凍結。
    """
    def __init__(self, 
                prompt_length: int = 10,
                hidden_size: int = 768,
                vocab_size: int = 50257):
        """
        初始化 Soft Prompt
        
        Args:
            prompt_length: soft prompt tokens 的數量
            hidden_size: 隱藏層維度
            vocab_size: 詞彙表大小
        """
        super().__init__()
        
        # 創建可學習的 prompt embeddings
        # shape: (prompt_length, hidden_size)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_size) * 0.02
        )
        # 創建 prompt token ids（不需要梯度，因為直接使用 embeddings）
        # 這些只是用於記錄，實際訓練的是 embeddings
        # 這是 PyTorch nn.Module 提供的方法
        # 註冊一個 不需要參與梯度更新 的張量（tensor），但它會成為模型的一部分，並且在 model.to(device) 時會自動搬到 GPU/CPU
        self.register_buffer('prompt_token_ids', 
                           torch.randint(0, vocab_size, (prompt_length,), dtype=torch.long))
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        
    def forward(self, batch_size: int = 1):
        """
        前向傳播
        
        Args:
            batch_size: 批次大小
            
        Returns:
            prompt_embeddings: 擴展到指定批次大小的 prompt embeddings
        """
        # 將 prompt embeddings 擴展到批次大小
        # shape: (batch_size, prompt_length, hidden_size)
        return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    
# 2. 定義 VLM 模型（整合 Soft Prompt）
class VLMWithSoftPrompt(nn.Module):
    """
    整合 Soft Prompt 的視覺語言模型
    這個模型將 CLIP 視覺編碼器和語言模型結合，
    並使用 soft prompt 來引導模型的行為。
    """
    def __init__(self, 
                    vision_model_name: str = "openai/clip-vit-base-patch32",
                    language_model_name: str = "gpt2",
                    prompt_length: int = 10):
        """
        初始化 VLM 模型
        Args:
            vision_model_name: CLIP 視覺模型名稱
            language_model_name: 語言模型名稱
            prompt_length: soft prompt 長度
        """
        super().__init__()
        
        # 載入預訓練模型
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        # 設置 tokenizer 的 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 創建 soft prompt
        self.soft_prompt = SoftPrompt(
            prompt_length=prompt_length,
            hidden_size=self.language_model.config.hidden_size,
            vocab_size=self.language_model.config.vocab_size
        )
        # 凍結原始模型參數
        self._freeze_models()
        # 投影層：將視覺特徵投影到語言模型的隱藏空間 (全連接層)
        self.vision_projection = nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size
        )
        
    def _freeze_models(self):
        """凍結原始模型的參數"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
            
    def get_trainable_parameters(self):
        """獲取可訓練的參數（只有 soft prompt 和投影層）"""
        trainable_params = []
        trainable_params.extend(self.soft_prompt.parameters())
        trainable_params.extend(self.vision_projection.parameters())
        return trainable_params
    
    def forward(self, 
                images: torch.Tensor,
                text_ids: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        前向傳播
        
        Args:
            images: 輸入圖像 (batch_size, channels, height, width)
            text_ids: 文本 token ids (batch_size, seq_len)
            attention_mask: 注意力遮罩
            
        Returns:
            outputs: 語言模型的輸出
        """
        batch_size = images.shape[0]
        
        # 1. 提取視覺特徵
        vision_outputs = self.vision_model(images)
        vision_features = vision_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 2. 投影視覺特徵到語言模型空間 (執行 foward )
        projected_vision = self.vision_projection(vision_features)
        
        # 3. 獲取 soft prompt embeddings
        soft_prompt_embeds = self.soft_prompt(batch_size)
        
        # 4. 獲取文本 embeddings
        text_embeds = self.language_model.get_input_embeddings()(text_ids)
        
        # 5. 組合所有 embeddings: [soft_prompt, vision, text]
        combined_embeds = torch.cat([
            soft_prompt_embeds,      # (batch_size, prompt_len, hidden_size)
            projected_vision,        # (batch_size, vision_seq_len, hidden_size)
            text_embeds             # (batch_size, text_seq_len, hidden_size)
        ], dim=1)
        
        # 6. 創建新的 attention mask
        prompt_len = soft_prompt_embeds.shape[1]
        vision_len = projected_vision.shape[1]
        text_len = text_embeds.shape[1]
        
        if attention_mask is not None:
            new_attention_mask = torch.cat([
                torch.ones(batch_size, prompt_len + vision_len, device=attention_mask.device),
                attention_mask
            ], dim=1)
        else:
            new_attention_mask = torch.ones(batch_size, prompt_len + vision_len + text_len, device=images.device)
        
        # 7. 通過語言模型
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=new_attention_mask,
            labels=None  # 我們會手動計算損失
        )
        return outputs
    
    
# 3. 定義數據集
class VLMDataset(Dataset):
    """
    視覺語言模型數據集
    這個數據集處理圖像-文本對，用於訓練 VLM。
    """
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 128,
                 image_size: int = 224):
        """
        初始化數據集
        
        Args:
            data_path: 數據文件路徑（JSON 格式）
            tokenizer: 文本 tokenizer
            max_length: 最大文本長度
            image_size: 圖像大小
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        
        # 載入數據
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # 初始化圖像處理器
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """獲取單個樣本"""
        item = self.data[idx]
        
        # 載入和處理圖像
        image = Image.open(item['image_path']).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # 處理圖像
        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )
        
        # 處理文本
        text = item['Caption']
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'images': image_inputs['pixel_values'].squeeze(0),
            'text_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'text': text
        }


def loss_plot(num_epochs, loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='b')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(saving_path / "loss_curve.png", dpi=300)
    plt.show()


# 4. 訓練函數
def train_vlm_with_soft_prompt(
    model: VLMWithSoftPrompt,
    train_dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    訓練 VLM 模型（使用 Soft Prompt）
    
    Args:
        model: VLM 模型
        train_dataloader: 訓練數據加載器
        num_epochs: 訓練輪數
        learning_rate: 學習率
        device: 設備
    """
    
    model = model.to(device)
    model.train()
    
    # 只優化可訓練的參數
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate
    )
    
    # 損失函數
    criterion = nn.CrossEntropyLoss()
    
    print(f"開始訓練，設備: {device}")
    print(f"可訓練參數數量: {sum(p.numel() for p in model.get_trainable_parameters()):,}")
    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 移動數據到設備
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向傳播
            outputs = model(images, text_ids, attention_mask)
            
            # 計算損失（只對文本部分計算）
            # 我們需要跳過 soft prompt 和視覺特徵部分
            prompt_len = model.soft_prompt.prompt_length
            vision_len = model.vision_model.config.image_size // model.vision_model.config.patch_size
            vision_len = (vision_len ** 2) + 1  # +1 for CLS token
            
            # 獲取文本部分的 logits
            text_logits = outputs.logits[:, prompt_len + vision_len:, :]
            text_labels = text_ids
            
            # 計算損失
            loss = criterion(
                text_logits.reshape(-1, text_logits.size(-1)),
                text_labels.reshape(-1)
            )
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} 完成，平均損失: {avg_loss:.4f}")
    print("訓練完成！")
    return loss_history


# 5. 主函數和示例
def create_sample_data():
    """創建示例數據文件"""
    # read the csv with text, image pair data
    data_path = "captions_final.csv"
    if not os.path.exists(data_path):
        raise(f"檔案 {data_path} 不存在，請確認路徑是否正確")
    data = pd.read_csv(data_path)
    # 轉換成 JSON list
    training_data =  data.to_dict(orient="records")
    # 創建目錄
    os.makedirs("sample_data", exist_ok=True)
    # 保存數據
    with open("sample_data/train_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print("訓練數據已創建！")
    

def main():
    """主函數 - 完整的訓練流程"""
    EPOCH = 10 
    LR = 1e-4
    print("=== Soft Prompt VLM Fine-tuning 教學 ===")
    # 1. 創建示例數據
    print("1. 將 csv 數據轉換為...")
    create_sample_data()
    
    # 2. 初始化模型
    print("\n2. 初始化 VLM 模型...")
    model = VLMWithSoftPrompt(
        vision_model_name="openai/clip-vit-base-patch32",
        language_model_name="gpt2",
        prompt_length=10
    )
    
    # 3. 準備數據
    print("\n3. 準備訓練數據...")
    dataset = VLMDataset(
        data_path="sample_data/train_data.json",
        tokenizer=model.tokenizer,
        max_length=128
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    # 4. 開始訓練
    print("\n4. 開始訓練...")
    loss_history = train_vlm_with_soft_prompt(
        model=model,
        train_dataloader=dataloader,
        num_epochs=EPOCH,
        learning_rate=LR
    )
    
    # 5. 保存模型
    print("\n5. 保存模型...")
    torch.save({
        'soft_prompt_state_dict': model.soft_prompt.state_dict(),
        'vision_projection_state_dict': model.vision_projection.state_dict(),
        'model_config': {
            'prompt_length': model.soft_prompt.prompt_length,
            'hidden_size': model.soft_prompt.hidden_size
        }
    }, './results/soft_prompt_vlm_checkpoint.pth')
    
    loss_plot(EPOCH, loss_history)
    print("\n=== 訓練完成！===")
    print("模型已保存為: soft_prompt_vlm_checkpoint.pth")
    print("\n使用說明:")
    print("1. 這個實現只訓練 soft prompt 和視覺投影層")
    print("2. 原始 CLIP 和 GPT-2 模型保持凍結")
    print("3. 大大減少了可訓練參數數量")
    print("4. 適合在有限計算資源下進行 fine-tuning")


if __name__ == "__main__":
    main()
