import ffmpeg
import os

def convert_to_wav(input_path: str, output_path: str = "output.wav"):
    if not input_path.endswith(".wav"):
        print("🎧 Converting to WAV...")
        ffmpeg.input(input_path).output(output_path).run(overwrite_output=True)
        return output_path
    return input_path