# 🧠 Speech-to-SQL: Voice-Based Database Querying with ASR + RAG + GenAI

This project enables users to **query databases using natural speech** by combining:
- 🎙️ Automatic Speech Recognition (ASR) via fine-tuned Whisper,
- 📄 Retrieval-Augmented Generation (RAG) using database schema context,
- 🤖 Generative AI to translate natural language to SQL.

## 🚀 Demo Use Case
> Say: _"What are the total sales in Germany last quarter?"_  
The system transcribes your speech, retrieves relevant DB schema info, generates SQL using GenAI, and optionally executes it.

---

## 📦 Features
- ✅ Fine-tuned Whisper model for accent-robust speech recognition.
- 🔍 Schema-aware RAG to improve query relevance.
- 🧠 LLM-based SQL generation (OpenAI, GPT-4, or local models).
- 💡 Optional query execution with result visualization.
- ☁️ Cloud-deployable, Dockerized, and fully modular.

---

## 🔧 Tech Stack
| Layer         | Tools |
|---------------|-------|
| ASR           | Fine-tuned Whisper |
| RAG           | FAISS + LangChain or custom retriever |
| LLM           | OpenAI GPT-4 / LLaMA2 + prompt engineering |
| Backend       | FastAPI or Flask |
| Frontend      | Streamlit / React |
| Database      | PostgreSQL (locally or via Supabase/RDS) |
| Deployment    | Docker, Render, AWS, Hugging Face Spaces |

---

## 🛠️ Local Setup

```bash
git clone https://github.com/yourusername/speech-to-sql-genai.git
cd speech-to-sql-genai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys and DB credentials here

# Run backend
cd app/api && uvicorn main:app --reload

# In another terminal, launch frontend
streamlit run app/ui/streamlit_app.py
