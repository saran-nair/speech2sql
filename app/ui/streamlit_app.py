import streamlit as st
import sqlite3
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from app.asr.whisper_inference import ASRModel
from app.asr.audio_utils import convert_to_wav
from app.rag.schema_embedder import SchemaEmbedder
from app.rag.retriever import SchemaRetriever
from app.llm.sql_generator import LocalLLM

# Load models
asr_model = ASRModel(
    base_model_name="openai/whisper-medium",
    adapter_path="Fine_Tuned_Model"
)

embedder = SchemaEmbedder("app/db/mock_schema.sql")
retriever = SchemaRetriever(embedder)
llm = LocalLLM(model_name="microsoft/phi-2")

st.set_page_config(page_title="Speech to SQL", layout="centered")
st.title("🎙️ Speech to SQL Web App")

# Upload audio
audio_file = st.file_uploader("Upload audio file (.wav or .mp3)", type=["wav", "mp3"])

if audio_file is not None:
    # Save uploaded audio to a temporary file
    temp_input_path = "temp_audio_input.wav"

    with open(temp_input_path, "wb") as f:
        f.write(audio_file.read())

    # If your convert_to_wav() function handles format conversion, use it
    wav_path = convert_to_wav(temp_input_path)  # Optional, depending on input

    # Transcribe
    transcript = asr_model.transcribe(wav_path)

    st.subheader("📝 Transcription")
    st.success(transcript)

    # SQL generation
    schema_context = "\n".join(retriever.retrieve(transcript))
    sql_query = llm.generate_sql(transcript, schema_context)
    st.subheader("🧾 Raw LLM Output")
    st.text(sql_query)

    st.subheader("🧠 Generated SQL")
    st.code(sql_query, language="sql")

    # Optional DB query execution
    with st.expander("🔁 Show query result from mock DB"):
        try:
            conn = sqlite3.connect("app/db/mock.db")
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()

            if rows:
                st.write("📄 Result:")
                for row in rows:
                    st.write(row)
            else:
                st.info("No results returned.")
        except Exception as e:
            st.error(f"Error executing SQL: {e}")
