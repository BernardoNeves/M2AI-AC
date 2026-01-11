import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def min_length(text: str, min_length: int = 5) -> bool:
    return bool(text) and len(text.split()) >= min_length


def truncate_text(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars]
