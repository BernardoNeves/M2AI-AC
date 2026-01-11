import argparse
import logging
import os
import sys
import time


from data.loader import load_audio_dataset, load_csv
from text.pipeline import TextAnalysisPipeline
from utils.file import load_jsons, save_results
from utils.logger import log_audio_results, log_text_results, set_log_file, setup_logger
from voice.pipeline import VoiceAnalysisPipeline


def run_command(config_paths):
    files = load_jsons(config_paths)
    if not files:
        logging.error("No valid configuration files found. Exiting.")
        sys.exit(1)

    for i, (name, configs) in enumerate(files.items(), 1):
        results = []
        logging.info(
            f"Running file {i}/{len(files)}: {name} with {len(configs)} configurations"
        )
        for j, config in enumerate(configs, 1):
            logging.info(f"Running configuration {j}/{len(configs)}")
            result = run(config)
            if result:
                results.append(result)
        logging.info(f"Completed running {len(results)}/{len(configs)} configurations.")
    logging.info("All groups completed.")


def load_command(results_path):
    files = load_jsons(results_path)
    logging.info(f"Loaded {len(files)} results files for plotting.")
    if not files:
        logging.error("No valid results files found. Exiting.")
        sys.exit(1)

    for i, (name, results) in enumerate(files.items(), 1):
        logging.info(
            f"Running file {i}/{len(files)}: {name} with {len(results)} configurations"
        )
        if results and "audio_path" in results[0]:
            log_audio_results(results)
        else:
            log_text_results(results)


def main():
    parser = argparse.ArgumentParser(description="Affective Computing")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run from config files")
    run_parser.add_argument(
        "config", type=str, nargs="+", help="Config file(s) or directories"
    )
    run_parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    load_parser = subparsers.add_parser("load", help="Load results from saved files")
    load_parser.add_argument(
        "results", type=str, nargs="+", help="Results JSON file(s)"
    )
    load_parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()
    if args.command not in ["run", "load"]:
        parser.print_help()
        sys.exit(1)

    setup_logger(args.log_level)
    if args.command == "run":
        run_command(args.config)
    elif args.command == "load":
        load_command(args.results)
    return


def run(config):
    dataset = config.get("dataset", "")
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    mode = config.get("mode", "text")
    limit = config.get("limit", None)
    random = config.get("random", False)

    results = []
    texts = []
    audio_dataset = []

    if mode == "text":
        texts = load_csv(dataset, text_column="text", limit=limit, random=random)
    elif mode == "audio":
        audio_dataset = load_audio_dataset(csv_path=dataset, limit=limit, random=random)
    else:
        raise ValueError("Provide a valid mode: 'text' or 'audio'")

    if mode == "text":
        if not texts:
            logging.warning("No texts to analyze.")
            return
        pipeline = TextAnalysisPipeline()
        results = pipeline.analyze_many(texts)

        save_results(results, dataset_name)

        log_text_results(results)

    elif mode == "audio":
        if not audio_dataset:
            logging.warning("No audio files to analyze.")
            return
        pipeline = VoiceAnalysisPipeline()
        for entry in audio_dataset:
            results = pipeline.analyze_many(entry)

        save_results(results, dataset_name)
        log_audio_results(results)
    return results


if __name__ == "__main__":
    main()
