# app/api/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import uuid
from app.asr.whisper_inference import ASRModel
from app.asr.audio_utils import convert_to_wav

# Paths
BASE_MODEL = "openai/whisper-medium"
ADAPTER_PATH = "C:/Users/saran/Desktop/Speech to SQL/speech-to-sql-genai/Fine_Tuned_Model"

# Initialize ASR model
asr_model = ASRModel(base_model_name=BASE_MODEL, adapter_path=ADAPTER_PATH)

# FastAPI app
app = FastAPI(title="Speech-to-SQL ASR Service")

@app.post("/asr/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save file temporarily
    contents = await file.read()
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    
    with open(temp_filename, "wb") as f:
        f.write(contents)

    # Convert to wav (if needed)
    wav_path = convert_to_wav(temp_filename)

    try:
        transcript = asr_model.transcribe(wav_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_filename)
        if os.path.exists("output.wav"):
            os.remove("output.wav")

    return {"transcription": transcript}