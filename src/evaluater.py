from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


EVAL_PROMPT = """
[Question]: {question}
[Reference Answer]: {gold_answer}
[Model A Answer]: {baseline_output}
[Model B Answer]: {finetuned_output}

Act as an impartial judge. 
Evaluate both Model A and Model B’s answers with respect to correctness, relevance, and completeness.
Give a score from 1-10 for each, and declare which is better.
"""
SYS_PROMPT = """
[Question]: {question}
[Model Answer]: {baseline_output}

Act as an impartial judge. 
Evaluate the Model’s answers with respect to correctness, relevance, and completeness.
Give a score from 1-10 for each, and declare which is better.
"""

class EvalLLm:
    def __init__(self,
            model_a_resp=None, 
            model_b_resp=None, 
            ground_truth=None, 
            prompt:str=None, 
            model_name:str="Qwen/Qwen2.5-7B-Instruct",
            device:str=None):
        self.model = model_name
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_a_resp = model_a_resp
        self.model_b_resp = model_b_resp
        self.ground_truth = ground_truth
        self.prompt = prompt

    def evaluation(self, question:str):
        """
        對 prompt 進行評估，回傳模型輸出
        """
        sys_prompt = EVAL_PROMPT if self.ground_truth else SYS_PROMPT
        messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        #  編碼輸入
        generated_ids =  self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # 生成回答
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("模型回答：\n", response)

    def judgement(self, scores:list, full_score:int=10):
        """
        根據分數列表計算準確率
        """
        if not scores:
            print("⚠️ scores 為空，無法計算")
            return 0.0
        accuracy = 0
        correct = sum(1 for s in scores if s == full_score)
        accuracy = correct / len(scores) * 100
        avg_score = sum(scores) / len(scores)
        print(f"Accuracy: {accuracy:.2f}% | 平均信心分數: {avg_score:.2f}/{full_score}")
        return {"accuracy": accuracy, "avg_score": avg_score}
   
   
def exe_evl(input_text:str, model_a, model_b=None, prompt:str=None):
    """
    使用 LLM 評估 RHLF 模型輸出
    """
    tuning_result = model_a(prompt)
    print("RLHF LLM Generated text:", tuning_result)
    eval_llm = EvalLLm(tuning_result, prompt)  
    scores = eval_llm.evaluation(prompt)
    eval_llm.judgement(scores)
    
        