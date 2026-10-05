"""Local Whisper word-timestamp cache for quote-anchor eval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from podcast_processor.boundary_snap import interpolate_word_times
from podcast_processor.quote_anchor import WordTiming

logger = logging.getLogger(__name__)


def words_from_payload(payload: dict[str, Any]) -> list[WordTiming]:
    words: list[WordTiming] = []
    for row in payload.get("words") or []:
        token = str(row.get("word") or "").strip()
        if not token:
            continue
        words.append(
            WordTiming(
                word=token,
                start=float(row["start"]),
                end=float(row["end"]),
            )
        )
    if words:
        return words
    for seg in payload.get("segments") or []:
        text = str(seg.get("text") or "")
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        for span in interpolate_word_times(start, end, text):
            words.append(WordTiming(word=span.word, start=span.start, end=span.end))
    return words


def save_word_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def segments_from_payload(payload: dict[str, Any]) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    for seg in payload.get("segments") or []:
        rows.append(
            (
                float(seg.get("start") or 0.0),
                float(seg.get("end") or 0.0),
                str(seg.get("text") or ""),
            )
        )
    return rows


def load_word_cache(path: Path) -> list[WordTiming] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = words_from_payload(payload)
    return words or None


def load_word_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def transcribe_word_timestamps(
    audio_path: Path,
    cache_path: Path,
    *,
    model_name: str = "base",
    language: str = "en",
    model: Any | None = None,
) -> list[WordTiming]:
    cached = load_word_cache(cache_path)
    if cached:
        return cached
    import whisper

    logger.info("word-timestamp whisper %s model=%s", audio_path, model_name)
    loaded = model if model is not None else whisper.load_model(name=model_name)
    result = loaded.transcribe(
        str(audio_path),
        fp16=False,
        language=language,
        word_timestamps=True,
    )
    words_out: list[dict[str, Any]] = []
    segments_out: list[dict[str, Any]] = []
    for seg in result.get("segments") or []:
        seg_words = []
        for item in seg.get("words") or []:
            token = str(item.get("word") or "").strip()
            if not token:
                continue
            row = {
                "word": token,
                "start": float(item.get("start") if item.get("start") is not None else seg.get("start") or 0.0),
                "end": float(item.get("end") if item.get("end") is not None else seg.get("end") or 0.0),
            }
            seg_words.append(row)
            words_out.append(row)
        segments_out.append(
            {
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": str(seg.get("text") or ""),
                "words": seg_words,
            }
        )
        if not seg_words:
            for span in interpolate_word_times(
                float(seg.get("start") or 0.0),
                float(seg.get("end") or 0.0),
                str(seg.get("text") or ""),
            ):
                words_out.append(
                    {"word": span.word, "start": span.start, "end": span.end}
                )
    payload = {
        "model": model_name,
        "word_timestamps": True,
        "audio_path": str(audio_path),
        "words": words_out,
        "segments": segments_out,
    }
    save_word_cache(cache_path, payload)
    return words_from_payload(payload)
