import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import tabulate

LOG_DIR = "logs"


def _file_handler(logfile: str, level: int) -> logging.FileHandler:
    h = logging.FileHandler(logfile)
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    return h


def _console_handler(level: str) -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    return h


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(log_level)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    logger.addHandler(_console_handler(log_level))
    return logger


def set_log_file(dataset: str, model: str) -> Tuple[logging.Logger, str]:
    logger = logging.getLogger()
    os.makedirs(LOG_DIR, exist_ok=True)
    logfile = os.path.join(
        LOG_DIR, f"{dataset}_{model}_{str(int(time.time() * 1000))}.log"
    )
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    logger.addHandler(_file_handler(logfile, logger.level))
    return logger, logfile


def log_text_results(results: list[dict], truncate: int = 30):
    table = []

    for result in results:
        emo = result["analysis"]["emotion"]

        table.append(
            [
                result["original_text"][:truncate]
                + ("..." if len(result["original_text"]) > truncate else ""),
                result["corrected_text"][:truncate]
                + ("..." if len(result["corrected_text"]) > truncate else ""),
                result["analysis"]["polarity_label"],
                result["analysis"]["subjectivity_label"],
                emo["label"],
                emo["valence"],
                emo["arousal"],
                ", ".join(result["analysis"]["statement_type"]),
            ]
        )

    if not table:
        print("No texts to analyze.")
        return

    headers = [
        "Original",
        "Corrected",
        "Polarity",
        "Subjectivity",
        "Emotion",
        "Valence",
        "Arousal",
        "Statement Type",
    ]
    print(tabulate.tabulate(table, headers=headers, tablefmt="grid"))


def log_audio_results(results: list[dict], truncate: int = 30):
    table = []
    for result in results:
        voice_emo = result["voice_emotion"]
        text_res = result["text_analysis"]
        text_emo = text_res["analysis"]["emotion"]

        table.append(
            [
                result["id"],
                Path(result["audio_path"]).name,
                result["transcription"][:truncate]
                + ("..." if len(result["transcription"]) > truncate else ""),
                voice_emo["label"],
                voice_emo["valence"],
                voice_emo["arousal"],
                text_emo["label"],
                text_res["analysis"]["subjectivity_label"],
                ", ".join(text_res["analysis"]["statement_type"]),
            ]
        )

    if not table:
        print("No audio to analyze.")
        return

    headers = [
        "Set ID",
        "File",
        "Transcription",
        "Voice Emo",
        "V. Valence",
        "V. Arousal",
        "Text Emo",
        "Text Subj",
        "Statement",
    ]
    print(tabulate.tabulate(table, headers=headers, tablefmt="grid"))
