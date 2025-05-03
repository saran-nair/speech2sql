import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import torchaudio

class ASRModel:
    def __init__(self, base_model_name: str = "openai/whisper-medium", adapter_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor from adapter path (it contains tokenizer etc.)
        self.processor = WhisperProcessor.from_pretrained(adapter_path)

        # Load base Whisper model
        print("🧠 Loading base Whisper model...")
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)

        # Load LoRA adapter on top
        print("🔗 Merging fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.model.to(self.device)

    def transcribe(self, audio_path: str) -> str:
        print("🎧 Loading audio:", audio_path)
        speech_array, sampling_rate = torchaudio.load(audio_path)
        speech_array = speech_array.squeeze().numpy()

        # Preprocess audio
        inputs = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        # Generate tokens and decode
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription

