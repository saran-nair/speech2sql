import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import ffmpeg
import numpy as np
import os

class ASRModel:
    def __init__(self, base_model_name="openai/whisper-medium", adapter_path=None):
        if not adapter_path:
            raise ValueError("Adapter path is required.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = WhisperProcessor.from_pretrained(adapter_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        self.model = base_model.from_pretrained(base_model_name)
        self.model.load_adapter(adapter_path)
        self.model.to(self.device).eval()

    def transcribe(self, audio_path: str) -> str:
        print("🎧 Loading audio:", audio_path)
        input_array, sampling_rate = self.load_audio(audio_path)

        inputs = self.processor(input_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def load_audio(self, path, sr=16000):
        out, _ = (
            ffmpeg.input(path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        return audio, sr
