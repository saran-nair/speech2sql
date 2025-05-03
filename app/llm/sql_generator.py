from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"🔍 Loading local model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_sql(self, nl_question: str, schema_context: str) -> str:
        prompt = f"""
You are a helpful assistant that converts natural language into SQL.

Schema:
{schema_context}

Question:
{nl_question}

SQL:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("SQL:")[-1].strip()
