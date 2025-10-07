"""
Soft Prompt RLHF (Reinforcement Learning from Human Feedback)
============================================================

這個腳本展示了如何使用強化學習來進一步優化 soft prompt，
通過人類反饋來改善模型的輸出質量。

主要概念：
1. PPO (Proximal Policy Optimization): 用於優化 soft prompt
2. Reward Model: 評估生成文本的質量
3. Human Feedback: 通過人類評分來訓練獎勵模型
4. Soft Prompt Optimization: 使用 RL 來優化 soft prompt 參數

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPProcessor,
    Trainer,
    TrainingArguments
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json
import wandb
from tqdm import tqdm
import scipy.stats as stats

# ============================================================================
# 1. 獎勵模型 (Reward Model)
# ============================================================================

class RewardModel(nn.Module):
    """
    獎勵模型：評估生成文本的質量
    
    這個模型學習預測人類對生成文本的評分，
    用於在 RLHF 過程中提供獎勵信號。
    """
    def __init__(
            self, 
            language_model_name: str = "gpt2",
            hidden_size: int = 768):
        super().__init__()
        # 載入預訓練語言模型作為基礎
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        # 凍結語言模型參數
        for param in self.language_model.parameters():
            param.requires_grad = False
        # 添加獎勵頭
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向傳播
        
        Args:
            input_ids: 輸入 token ids
            attention_mask: 注意力遮罩
            
        Returns:
            reward: 預測的獎勵分數
        """
        # 獲取語言模型的最後一層隱藏狀態
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最後一個 token 的隱藏狀態
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
        # 預測獎勵分數
        reward = self.reward_head(last_token_hidden)  # (batch_size, 1)
        return reward
        
# ============================================================================
# 2. 人類反饋數據集
# ============================================================================
class HumanFeedbackDataset(Dataset):
    """
    人類反饋數據集
    
    包含圖像、生成文本和人類評分的數據集，
    用於訓練獎勵模型和進行 RLHF。
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 128,
                 image_processor=None):
        """
        初始化數據集
        
        Args:
            data_path: 數據文件路徑
            tokenizer: 分詞器
            max_length: 最大文本長度
            image_processor: 圖像處理器
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        
        # 載入數據
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 載入圖像
        image = Image.open(item['image_path']).convert('RGB')
        if self.image_processor:
            image = self.image_processor(image, return_tensors='pt')['pixel_values']
        
        # 處理文本
        text = item['text']
        if 'human_score' in item:
            human_score = item['human_score']
        else:
            human_score = 0.5  # 預設分數
            
        # Tokenize 文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'human_score': torch.tensor(human_score, dtype=torch.float),
            'text': text
        }

# ============================================================================
# 3. Soft Prompt RLHF 模型
# ============================================================================
class SoftPromptRLHF(nn.Module):
    """
    整合 RLHF 的 Soft Prompt 模型
    
    這個模型結合了 soft prompt 和 RLHF，
    使用強化學習來優化 soft prompt 參數。
    """
    
    def __init__(self, 
                 vision_model_name: str = "openai/clip-vit-base-patch32",
                 language_model_name: str = "gpt2",
                 prompt_length: int = 10,
                 hidden_size: int = 768):
        super().__init__()
        
        # 載入預訓練模型
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        # 設置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 創建 soft prompt
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, hidden_size) * 0.02
        )
        
        # 視覺投影層
        self.vision_projection = nn.Linear(
            self.vision_model.config.hidden_size, 
            hidden_size
        )
        
        # 凍結原始模型
        self._freeze_models()
        
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        
    def _freeze_models(self):
        """凍結原始模型參數"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
            
    def forward(self, 
                images, 
                input_ids=None, 
                attention_mask=None,
                generate: bool = False,
                max_length: int = 128):
        """
        前向傳播
        
        Args:
            images: 輸入圖像
            input_ids: 輸入 token ids
            attention_mask: 注意力遮罩
            generate: 是否生成文本
            max_length: 生成的最大長度
            
        Returns:
            模型輸出
        """
        batch_size = images.shape[0]
        
        # 1. 提取視覺特徵
        vision_outputs = self.vision_model(images)
        vision_features = vision_outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token
        vision_features = self.vision_projection(vision_features)  # (batch_size, hidden_size)
        
        # 2. 準備 soft prompt
        soft_prompt_embeddings = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, prompt_length, hidden_size)
        
        if generate:
            # 生成模式
            return self._generate_text(vision_features, max_length)
        else:
            # 訓練模式
            return self._forward_pass(vision_features, input_ids, attention_mask)
    
    def _forward_pass(self, vision_features, input_ids, attention_mask):
        """訓練模式的前向傳播"""
        batch_size = vision_features.shape[0]
        
        # 獲取文本 embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # 組合輸入：soft_prompt + vision_features + text_embeddings
        vision_features_expanded = vision_features.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        combined_embeddings = torch.cat([
            self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1),
            vision_features_expanded,
            text_embeddings
        ], dim=1)
        
        # 調整注意力遮罩
        new_attention_mask = torch.cat([
            torch.ones(batch_size, self.prompt_length + 1, device=input_ids.device),
            attention_mask
        ], dim=1)
        
        # 前向傳播
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=new_attention_mask
        )
        
        return outputs
    
    def _generate_text(self, vision_features, max_length):
        """生成文本"""
        batch_size = vision_features.shape[0]
        
        # 準備初始輸入
        soft_prompt_embeddings = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        vision_features_expanded = vision_features.unsqueeze(1)
        
        # 初始 embeddings
        combined_embeddings = torch.cat([
            soft_prompt_embeddings,
            vision_features_expanded
        ], dim=1)
        
        # 生成文本
        generated_ids = self.language_model.generate(
            inputs_embeds=combined_embeddings,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return generated_ids


# ============================================================================
# 4. RLHF 訓練器
# ============================================================================

class SoftPromptRLHFTrainer:
    """
    Soft Prompt RLHF 訓練器
    
    使用 PPO 算法來優化 soft prompt 參數，
    基於人類反饋來改善模型輸出。
    """
    def __init__(self,
                model: SoftPromptRLHF,
                reward_model: RewardModel,
                tokenizer,
                device: str = "mps"):
        self.model = model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # PPO 配置
        self.ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=4,
            mini_batch_size=2,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            max_grad_norm=1.0,
            seed=42
        )
        
        # 初始化 PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=None,  # 使用當前模型作為參考
            tokenizer=self.tokenizer,
            dataset=None,  # 將在訓練時設置
            data_collator=self._collate_fn
        )

