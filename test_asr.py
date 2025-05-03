# test_asr.py

from app.asr.whisper_inference import ASRModel
from app.asr.audio_utils import convert_to_wav

audio_file = r"C:\Users\saran\Desktop\Speech to SQL\speech-to-sql-genai\data\samples\sample.wav"
wav_path = convert_to_wav(audio_file)

# Update paths below
BASE_MODEL = "openai/whisper-medium"
ADAPTER_PATH = r"C:/Users/saran/Desktop/Speech to SQL/speech-to-sql-genai/Fine_Tuned_Model"

asr = ASRModel(base_model_name=BASE_MODEL, adapter_path=ADAPTER_PATH)
print("📝 Transcription:", asr.transcribe(wav_path))