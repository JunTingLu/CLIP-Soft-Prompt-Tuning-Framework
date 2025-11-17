# CLIP-Soft-Prompt-Tuning-Framework
A lightweight framework for fine-tuning Vision-Language Models (VLM) using Soft Prompt Tuning — achieving parameter-efficient adaptation without modifying the original model weights.

---

### 🎯 Introduction to Soft Prompt
Soft Prompting introduces **learnable continuous vectors** that guide the model’s behavior.  
Unlike traditional *hard prompts* (human-readable text), soft prompts are **trainable embeddings** prepended to the model’s input.

#### 🧱 Traditional Hard Prompt
```
Input: "Please describe this image:"
Image: [Dog image]
Output: "This is a cute dog..."
```
#### 🧠 Soft Prompt
```
Input: [Learnable Soft Vector] + Image Features + Text
Soft Prompt: [0.1, -0.3, 0.8, ...]
Image: [Visual Feature Vector]
Text: "This is a ..."
Output: "This is a cute little dog..."
```
#### ✨ Core Concepts
- Generate **trainable prompt embeddings**
- Achieve **parameter efficiency** (only train soft prompts)
- **Freeze** the original pre-trained model weights

---
### 🛠️ Soft Prompt Training Flow
1. Load pre-trained VLM (e.g., CLIP + GPT-2)
2. Initialize soft prompt embeddings
3. Freeze model backbone
4. Train only the soft prompt vectors
5. Evaluate with text-to-image retrieval metrics (e.g., Recall@K)

### 🏆 Why Soft Prompt Tuning?
- ✅ No need to fine-tune all model parameters  
- ⚡ Faster training, lower GPU memory usage  
- 🔐 Keep original model knowledge intact  
- 📈 Improves downstream task performance efficiently

---
### 📁 File Structure
```text
CLIP-Soft-Prompt-Tuning-Framework/
│
├── requirements.txt        
├── README.md               
│
└── src/
    ├── preprocess_data.py  
    └── soft_prompt_tuning.py            
```

---
### 🚀 Quick Start
1. **Installation**
```
pip install -r requirements.txt
```
2. **Data Format Example**
```
  [
    {
      "image_path": "./dataset/20230730_205018.jpg",
      "Caption": "這張圖片顯示的是夜空中的景象。"
    },
    {
      "image_path": "./dataset/IMG_20210611_021241_419.jpg",
      "Caption": "這張照片中，一個人側身站立，背對著攝影機，頭髮是短髮。"
    },...
  ]
```
Run preprocessing:
```bash
python src/preprocess_data.py \
    --input_path ./dataset/captions.csv \
    --output_path ./dataset/captions_cleaned.csv
```

3. **Train Soft Prompt Tuning**
```bash
python src/soft_prompt_tuning.py \
    --file_path ./dataset/captions_cleaned.csv \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda

```

### 🔧 Key Modules
1. `SoftPrompt` class
  ```
   class SoftPrompt(nn.Module):
    def __init__(self, 
                prompt_length: int = 10,
                hidden_size: int = 768,
                vocab_size: int = 50257):
        super().__init__()
        # create a learnable prompt embeddings
        # shape: (prompt_length, hidden_size)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_size) * 0.02
        )
  ```
  - `prompt_length`：soft prompt tokens number
  - `hidden_size`：the dimension of hidden layer
  - `vocab_size`：vocabulary size
  
2. `VLMWithSoftPrompt` class
  ```
  class VLMWithSoftPrompt(nn.Module):
    def __init__(self, 
                    vision_model_name: str = "openai/clip-vit-base-patch32",
                    language_model_name: str = "gpt2",
                    prompt_length: int = 10):
        super().__init__()
        # load pre-trained model
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        # create soft prompt
        self.soft_prompt = SoftPrompt(
            prompt_length=prompt_length,
            hidden_size=self.language_model.config.hidden_size,
            vocab_size=self.language_model.config.vocab_size
        )
        # frozen model's parameters
        self._freeze_models()
  ```
---
### 📈 Evaluation Metric — Recall@K
Recall@K measures the proportion of correct target images retrieved within the top K results, serving as a key metric for evaluating text-to-image retrieval accuracy.
A higher Recall@K indicates stronger alignment between textual and visual representations.

## Development Notes
- Uses CLIP from openai/clip-vit-base-patch32
- Only optimizes soft prompts (encoder frozen)
- Supports GPU / MPS / CPU fallback
- Dataset loader auto-resizes images to CLIP input size

### 📚 Reference
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [OpenAI CLIP](https://github.com/openai/CLIP)
