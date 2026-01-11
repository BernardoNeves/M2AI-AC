from transformers import pipeline


class SpeechToTextModel:
    def __init__(self, device: str):
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device,
        )

    def transcribe(self, audio_path: str) -> str:
        result = self.asr(audio_path)
        return result["text"]


class VoiceEmotionModel:
    def __init__(self, device: str):
        self.classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=device,
        )
        self.pad_mapping = {
            "joy": {"valence": "positive", "arousal": "high"},
            "surprise": {"valence": "positive", "arousal": "high"},
            "anger": {"valence": "negative", "arousal": "high"},
            "fear": {"valence": "negative", "arousal": "high"},
            "disgust": {"valence": "negative", "arousal": "low"},
            "sadness": {"valence": "negative", "arousal": "low"},
            "neutral": {"valence": "neutral", "arousal": "neutral"},
        }
        self.label_map = {
            "hap": "joy",
            "sad": "sadness",
            "ang": "anger",
            "neu": "neutral",
        }

    def predict(self, audio_path: str) -> dict:
        results = self.classifier(audio_path, top_k=3)
        label = results[0]["label"]
        dimensions = self.pad_mapping.get(
            label, {"valence": "neutral", "arousal": "neutral"}
        )

        return {
            "label": self.label_map.get(label, "neutral"),
            "valence": dimensions["valence"],
            "arousal": dimensions["arousal"],
            "score": results[0]["score"],
        }
