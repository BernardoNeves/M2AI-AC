import language_tool_python
import spacy
from textblob import TextBlob
from transformers import pipeline


class TextEmotionModel:
    def __init__(self, device):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=device,
            truncation=True,
            max_length=512,
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

    def predict(self, text):
        result = self.classifier(text, truncation=True, max_length=512)
        label = result[0]["label"]

        dimensions = self.pad_mapping.get(
            label, {"valence": "neutral", "arousal": "neutral"}
        )

        return {
            "label": label,
            "valence": dimensions["valence"],
            "arousal": dimensions["arousal"],
        }


class TextCorrector:
    def __init__(self, language: str = "en-US"):
        self.tool = language_tool_python.LanguageTool(language)

    def correct(self, text: str) -> str:
        matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text


class StatementTypeAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, text: str, subjectivity: float) -> list[str]:
        doc = self.nlp(text)
        types = set()

        has_negation = any(token.dep_ == "neg" for token in doc)
        types.add("negation" if has_negation else "affirmation")

        first_person = {"i", "me", "my", "mine", "we", "us", "our"}
        has_personal_pronouns = any(token.text.lower() in first_person for token in doc)

        if has_personal_pronouns or subjectivity > 0.5:
            types.add("personal")
        else:
            types.add("factual")

        return list(types)


class SentimentAnalyzer:
    def analyze(
        self,
        text: str,
        polarity_threshold: float = 0.2,
        subjectivity_threshold: float = 0.5,
    ) -> dict:
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "polarity_label": (
                "neutral"
                if abs(polarity) <= polarity_threshold
                else ("positive" if polarity > 0 else "negative")
            ),
            "subjectivity_label": "personal"
            if subjectivity > subjectivity_threshold
            else "factual",
        }
