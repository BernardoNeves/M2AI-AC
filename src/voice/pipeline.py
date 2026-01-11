from typing import Dict, List
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from text.pipeline import TextAnalysisPipeline
from voice.models import SpeechToTextModel, VoiceEmotionModel


class VoiceAnalysisPipeline:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.stt = SpeechToTextModel(device)
        self.voice_emotion = VoiceEmotionModel(device)
        self.text_pipeline = TextAnalysisPipeline(device)

    def analyze(self, audio_path: str) -> dict:
        transcription = self.stt.transcribe(audio_path)
        text_result = self.text_pipeline.analyze(transcription)
        voice_emotion = self.voice_emotion.predict(audio_path)

        return {
            "audio_path": audio_path,
            "transcription": transcription,
            "voice_emotion": voice_emotion,
            "text_analysis": text_result,
        }

    def analyze_many(self, entry: Dict) -> List[Dict]:
        audio_paths = entry.get("files")
        if not audio_paths:
            return []

        results = []
        for path in audio_paths:
            result = self.analyze(path)
            result["id"] = entry["set_id"]
            result["metadata"] = {
                "audio_path": path,
                "set_id": entry["set_id"],
                "gender": entry["gender"],
                "age": entry["age"],
                "country": entry["country"],
            }
            results.append(result)
        return results


def save_spectrogram(audio_path: str, output_path: str):
    y, sr = librosa.load(audio_path)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency spectrogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
