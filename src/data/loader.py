import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.preprocessing import clean_text, min_length


def load_csv(
    path: str,
    text_column: str = "text",
    limit: int | None = None,
    random: bool = False,
) -> list[str]:
    df = pd.read_csv(Path(path))

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")

    texts = []

    if random:
        df = df.sample(frac=1).reset_index(drop=True)

    for raw_text in df[text_column]:
        cleaned = clean_text(str(raw_text))
        if min_length(cleaned):
            texts.append(cleaned)

        if limit and len(texts) >= limit:
            break

    return texts


def load_result(path: str) -> list[dict]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_audio_csv(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise ValueError(f"Audio folder not found: {folder}")

    audio_files = []
    for file in sorted(os.listdir(folder)):
        ext = os.path.splitext(file)[1].lower()
        if ext in {".wav", ".mp3", ".flac"}:
            audio_files.append(os.path.join(folder, file))

    if not audio_files:
        raise ValueError(f"No supported audio files found in {folder}")

    return audio_files


def load_audio_dataset(
    csv_path: str,
    audio_root: str = "datasets/audio/files",
    limit: int | None = None,
    random: bool = False,
) -> List[Dict]:
    df = pd.read_csv(csv_path)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if random and limit is not None:
        df = df.sample(n=limit).reset_index(drop=True)
    elif limit is not None:
        df = df.head(limit)

    dataset = []

    for _, row in df.iterrows():
        set_id = row["set_id"]
        folder_path = Path(audio_root) / set_id
        if not folder_path.exists() or not folder_path.is_dir():
            continue

        audio_files = sorted([str(p) for p in folder_path.glob("*.wav")])
        if not audio_files:
            continue

        dataset.append(
            {
                "set_id": set_id,
                "files": audio_files,
                "text": row.get("text", ""),
                "gender": row.get("gender", ""),
                "age": row.get("age", None),
                "country": row.get("country", ""),
            }
        )

    return dataset
