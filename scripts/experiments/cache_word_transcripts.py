#!/usr/bin/env python3
"""Cache local Whisper word timestamps for the 10 scored scorecard episodes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from podcast_processor.experiments.word_transcripts import transcribe_word_timestamps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("docs/experiments/auto_gold/runs/prod-scorecard"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/auto_gold/runs/quote-anchor/words"),
    )
    parser.add_argument("--model", default=os.environ.get("WHISPER_LOCAL_MODEL") or "base")
    parser.add_argument("--show-id", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    downloads = json.loads((args.audio_dir / "downloads.json").read_text(encoding="utf-8"))
    shows = args.show_id or [
        sid
        for sid, row in downloads.items()
        if row.get("audio_path") and sid not in {"bill_simmons", "marketplace"}
    ]
    repo = Path(__file__).resolve().parents[2]
    sized: list[tuple[int, str, Path]] = []
    for show_id in shows:
        audio = Path(str(downloads[show_id]["audio_path"]))
        if not audio.is_absolute():
            audio = repo / audio
        sized.append((audio.stat().st_size if audio.is_file() else 1 << 62, show_id, audio))
    sized.sort()
    whisper_model = None
    for _, show_id, audio in sized:
        cache = args.output_dir / f"{show_id}.json"
        if cache.is_file():
            logging.info("cached %s", cache)
            continue
        if not audio.is_file():
            logging.error("missing audio %s", audio)
            return 2
        if whisper_model is None:
            import whisper

            logging.info("loading whisper model=%s", args.model)
            whisper_model = whisper.load_model(name=args.model)
        transcribe_word_timestamps(
            audio, cache, model_name=args.model, model=whisper_model
        )
        logging.info("wrote %s", cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
