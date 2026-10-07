#!/usr/bin/env python3
"""Forced-align segment Whisper transcripts onto audio ($0 Gemini)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from podcast_processor.experiments.word_transcripts import (
    payload_from_aligned_words,
    payload_is_aligned,
    save_word_cache,
    align_words_to_segments,
    load_word_payload,
    segment_rows_from_transcript,
)

REPO = Path(__file__).resolve().parents[2]
ORIG_AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
ORIG_SEGS = ORIG_AUDIO / "transcripts"
HOLD = REPO / "docs/experiments/auto_gold/runs/quote-anchor/holdout"
DEFAULT_OUT = REPO / "docs/experiments/auto_gold/runs/quote-anchor/words-aligned"

ORIG_SHOWS = (
    "bbc_football_daily",
    "bbc_global_news",
    "npr_up_first",
    "wtf_maron",
    "planet_money",
    "serial",
    "crime_junkie",
    "npr_wait_wait",
    "darknet_diaries",
    "atp",
)
HOLD_SHOWS = ("marketplace", "bill_simmons", "npr_indicator", "this_american_life")


def _audio_for(show_id: str, downloads: dict, base: Path) -> Path | None:
    row = downloads.get(show_id) or {}
    raw = row.get("audio_path")
    if not raw:
        return None
    audio = Path(str(raw))
    if not audio.is_absolute():
        audio = REPO / audio
    return audio if audio.is_file() else None


def _segs_for(show_id: str) -> Path | None:
    if show_id in ORIG_SHOWS:
        path = ORIG_SEGS / f"{show_id}.json"
        return path if path.is_file() else None
    path = HOLD / "baseline" / "transcripts" / f"{show_id}.json"
    if path.is_file():
        return path
    alt = HOLD / "quote-anchor" / "transcripts" / f"{show_id}.json"
    return alt if alt.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.environ.get("WHISPER_LOCAL_MODEL") or "base")
    parser.add_argument("--show-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    orig_dl = json.loads((ORIG_AUDIO / "downloads.json").read_text(encoding="utf-8"))
    hold_dl = json.loads((HOLD / "downloads.json").read_text(encoding="utf-8"))
    wanted = args.show_id or list(ORIG_SHOWS) + list(HOLD_SHOWS)
    jobs: list[tuple[str, Path, Path]] = []
    for sid in wanted:
        segs = _segs_for(sid)
        audio = _audio_for(sid, orig_dl if sid in ORIG_SHOWS else hold_dl, ORIG_AUDIO)
        if segs is None or audio is None:
            logging.error("missing inputs for %s segs=%s audio=%s", sid, segs, audio)
            continue
        jobs.append((sid, segs, audio))
    if not jobs:
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    summary = []
    for sid, segs_path, audio in jobs:
        cache = args.output_dir / f"{sid}.json"
        if cache.is_file() and not args.force:
            payload = load_word_payload(cache)
            if payload_is_aligned(payload):
                logging.info("cached %s", cache)
                summary.append(
                    {
                        "show_id": sid,
                        "cached": True,
                        "align_s": payload.get("align_s"),
                        "n_words": len(payload.get("words") or []),
                    }
                )
                continue
        rows = segment_rows_from_transcript(
            json.loads(segs_path.read_text(encoding="utf-8"))
        )
        if model is None:
            import whisper

            logging.info("loading whisper model=%s", args.model)
            t_load = time.time()
            model = whisper.load_model(name=args.model)
            logging.info("model load %.1fs", time.time() - t_load)
        logging.info("align %s n_seg=%s duration_hint=%.0fs", sid, len(rows), rows[-1][1] if rows else 0)
        words, align_s = align_words_to_segments(
            audio, rows, model=model, language="en"
        )
        payload = payload_from_aligned_words(
            words,
            rows,
            model_name=args.model,
            audio_path=audio,
            align_s=align_s,
        )
        save_word_cache(cache, payload)
        logging.info("wrote %s words=%s align_s=%.1f", cache, len(words), align_s)
        summary.append(
            {
                "show_id": sid,
                "cached": False,
                "align_s": round(align_s, 2),
                "n_words": len(words),
                "n_segments": len(rows),
            }
        )
    (args.output_dir / "align_wall.json").write_text(
        json.dumps({"model": args.model, "shows": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model": args.model, "shows": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
