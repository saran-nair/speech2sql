from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"🔍 Loading local model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate_sql(self, nl_question: str, schema_context: str) -> str:
        prompt = f"""
You are a helpful assistant that only returns syntactically correct SQL queries.

Only return SQL — do not include explanations, comments, or results.

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

        # Clean up the output to extract only the SQL part
        sql_part = result.split("SQL:")[-1].strip()
        lines = sql_part.split("\n")
        sql_only = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith(("select", "with", "insert", "update", "delete")):
                sql_only.append(line)
            elif ";" in line or line.endswith(";"):
                sql_only.append(line)
                break
            else:
                sql_only.append(line)
                break

        return " ".join(sql_only).strip() if sql_only else sql_part