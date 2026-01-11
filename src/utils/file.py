import json
import logging
import os
import time
from typing import Any, List, Dict

from voice.pipeline import save_spectrogram

RESULTS_DIR = "results"


def save_results(results: List[dict], filename: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(
        RESULTS_DIR, f"{filename}_{str(int(time.time() * 1000))}.json"
    )
    try:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"All results saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save all results to {filepath}: {e}")

    if any("audio_path" in r for r in results):
        for r in results:
            if "audio_path" in r:
                audio_path = r["audio_path"]
                id = r.get("id", time.time() * 1000)
                output_path = os.path.join(
                    filepath.rsplit(".", 1)[0],
                    f"{id}_{os.path.basename(audio_path).rsplit('.', 1)[0]}.png",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_spectrogram(audio_path, output_path=output_path)


def load_json_file(path: str) -> List[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse '{path}': {e}")

    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"File '{path}' does not contain a JSON object or list.")


def find_files(input_paths: List[str]) -> List[List[str]]:
    found = []
    for p in input_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")
        if os.path.isfile(p):
            if p.lower().endswith(".json"):
                found.append([p])
            continue
        pfound = []
        for root, _, files in os.walk(p):
            for f in files:
                if f.lower().endswith(".json"):
                    pfound.append(os.path.join(root, f))
        if not pfound:
            raise FileNotFoundError(f"No JSON files found in directory: {p}")
        found.append(pfound)
    return found


def load_jsons(paths: List[str]) -> Dict[str, List[Any]]:
    file_groups = find_files(paths)
    file_map = {}
    for group in file_groups:
        for filepath in group:
            name = os.path.basename(filepath).rsplit(".", 1)[0]
            try:
                data = load_json_file(filepath)
                file_map[name] = data
                print(f"Loaded {len(data)} entries from {name}")
                logging.info(f"Loaded {len(data)} entries from {name}")
            except Exception as e:
                logging.error(f"Error loading {name}: {e}")
    print(f"Total files loaded: {len(file_map)}")
    return file_map
