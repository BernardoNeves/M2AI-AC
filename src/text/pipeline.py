import torch

from data.preprocessing import truncate_text
from .models import (
    TextCorrector,
    SentimentAnalyzer,
    TextEmotionModel,
    StatementTypeAnalyzer,
)


class TextAnalysisPipeline:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.corrector = TextCorrector()
        self.sentiment = SentimentAnalyzer()
        self.emotion = TextEmotionModel(device)
        self.statement = StatementTypeAnalyzer()

    def analyze(self, text: str) -> dict:
        text = truncate_text(text)

        corrected = self.corrector.correct(text)
        sentiment_result = self.sentiment.analyze(corrected)

        return {
            "original_text": text,
            "corrected_text": corrected,
            "analysis": {
                **sentiment_result,
                "emotion": self.emotion.predict(corrected),
                "statement_type": self.statement.analyze(
                    corrected, sentiment_result["subjectivity"]
                ),
            },
        }

    def analyze_many(self, texts: list[str]) -> list[dict]:
        return [self.analyze(text) for text in texts]
