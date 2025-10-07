"""
RL 組件使用示例
==============

這個腳本展示了如何使用強化學習組件來改善 soft prompt。
"""
import torch
import json
from transformers import AutoTokenizer
from PIL import Image
import numpy as np

# 導入我們的 RL 組件
from soft_prompt_rlhf import (
    SoftPromptRLHF, 
    RewardModel, 
    SoftPromptRLHFTrainer,
    HumanFeedbackDataset
)