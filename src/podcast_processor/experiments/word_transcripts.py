"""Word timings for quote-anchor: forced-align segment Whisper text.

openai-whisper `word_timestamps=True` rewinds `seek` to the last DTW word
inside each 30s window (`whisper/transcribe.py`). On full episodes that
skips DAI/music beds that segment decode already transcribed (Indicator
Wise+donate, Up First 865–887). Alignment does not re-decode and cannot
drop segment speech.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from podcast_processor.boundary_snap import interpolate_word_times
from podcast_processor.quote_anchor import WordTiming

logger = logging.getLogger(__name__)

ALIGNED_BACKEND = "segment_align"
ALIGN_WINDOW_S = 28.0
ALIGN_PAD_S = 0.25
MAX_ALIGN_AUDIO_S = 29.9
SegmentRow = tuple[float, float, str]


def payload_is_aligned(payload: dict[str, Any] | None) -> bool:
    return bool(payload) and str(payload.get("backend") or "") == ALIGNED_BACKEND


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


def payload_from_whisper_result(
    result: dict[str, Any],
    *,
    model_name: str,
    audio_path: Path | str,
) -> dict[str, Any]:
    """Legacy decoder-path payload. Do not use for matching; prefer segment_align."""
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
                "start": float(
                    item.get("start")
                    if item.get("start") is not None
                    else seg.get("start") or 0.0
                ),
                "end": float(
                    item.get("end")
                    if item.get("end") is not None
                    else seg.get("end") or 0.0
                ),
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
    return {
        "model": model_name,
        "word_timestamps": True,
        "audio_path": str(audio_path),
        "words": words_out,
        "segments": segments_out,
    }


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


def segment_rows_from_transcript(raw: Any) -> list[SegmentRow]:
    rows = raw.get("segments") if isinstance(raw, dict) else raw
    out: list[SegmentRow] = []
    for seg in rows or []:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        out.append((start, end, text))
    return out


def group_segment_rows(
    rows: list[SegmentRow], *, max_span_s: float = ALIGN_WINDOW_S
) -> list[list[SegmentRow]]:
    """Pack consecutive segments into windows Whisper can align (≤30s mel)."""
    groups: list[list[SegmentRow]] = []
    current: list[SegmentRow] = []
    origin = 0.0
    for row in rows:
        start, end, _text = row
        if end - start > max_span_s:
            if current:
                groups.append(current)
                current = []
            groups.append([row])
            continue
        if not current:
            current = [row]
            origin = start
            continue
        if end - origin <= max_span_s:
            current.append(row)
            continue
        groups.append(current)
        current = [row]
        origin = start
    if current:
        groups.append(current)
    return groups


def _interpolate_rows(rows: list[SegmentRow]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for start, end, text in rows:
        words.extend(lexicon_words_for_segment(start, end, text, dtw_words=None))
    return words


def lexicon_words_for_segment(
    start: float,
    end: float,
    text: str,
    dtw_words: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Keep every segment token. Use DTW times only when the word count matches."""
    expected = interpolate_word_times(start, end, text)
    if not expected:
        return []
    linear = [
        {"word": span.word, "start": span.start, "end": span.end} for span in expected
    ]
    if not dtw_words:
        return linear
    sane: list[dict[str, Any]] = []
    for row in dtw_words:
        a = float(row["start"])
        b = float(row["end"])
        if b < a:
            a, b = b, a
        if b - a > 2.5 or b < start - 0.25 or a > end + 0.25:
            continue
        sane.append({"word": str(row.get("word") or "").strip(), "start": a, "end": b})
    if len(sane) != len(expected):
        return linear
    out: list[dict[str, Any]] = []
    prev = start
    for span, row in zip(expected, sane, strict=True):
        a = max(prev, float(row["start"]))
        b = float(row["end"])
        if b <= a:
            b = a + max(0.02, span.end - span.start)
        out.append({"word": span.word, "start": a, "end": b})
        prev = b
    return out


def lock_words_to_segment_lexicon(
    dtw_words: list[dict[str, Any]],
    rows: list[SegmentRow],
) -> list[dict[str, Any]]:
    """Re-label DTW timings with segment tokens so quote matching cannot drop speech."""
    assigned: list[list[dict[str, Any]]] = [[] for _ in rows]
    for row in dtw_words:
        a = float(row["start"])
        b = float(row["end"])
        best_i = -1
        best_ov = 0.0
        for i, (start, end, _text) in enumerate(rows):
            ov = min(b, end) - max(a, start)
            if ov > best_ov:
                best_ov = ov
                best_i = i
        if best_i >= 0 and best_ov > 0:
            assigned[best_i].append(row)
    locked: list[dict[str, Any]] = []
    for (start, end, text), bucket in zip(rows, assigned, strict=True):
        locked.extend(lexicon_words_for_segment(start, end, text, dtw_words=bucket))
    locked.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return locked


def align_words_to_segments(
    audio_path: Path,
    rows: list[SegmentRow],
    *,
    model: Any,
    language: str = "en",
) -> tuple[list[dict[str, Any]], float]:
    """DTW-align known segment text onto audio. One decoder pass per ~28s group."""
    from whisper.audio import (
        N_FRAMES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from whisper.timing import add_word_timestamps
    from whisper.tokenizer import get_tokenizer

    started = time.time()
    if not rows:
        return [], 0.0
    audio = load_audio(str(audio_path))
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task="transcribe")
    n_mels = int(getattr(model.dims, "n_mels", 80))
    words: list[dict[str, Any]] = []
    for group in group_segment_rows(rows):
        try:
            words.extend(
                _align_group(
                    audio,
                    group,
                    model=model,
                    tokenizer=tokenizer,
                    n_mels=n_mels,
                    sample_rate=SAMPLE_RATE,
                    n_frames=N_FRAMES,
                    pad_or_trim=pad_or_trim,
                    log_mel_spectrogram=log_mel_spectrogram,
                    add_word_timestamps=add_word_timestamps,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "segment align failed %.1f-%.1f; interpolating group",
                group[0][0],
                group[-1][1],
            )
            words.extend(_interpolate_rows(group))
    return words, time.time() - started


def _align_group(
    audio: Any,
    group: list[SegmentRow],
    *,
    model: Any,
    tokenizer: Any,
    n_mels: int,
    sample_rate: int,
    n_frames: int,
    pad_or_trim: Any,
    log_mel_spectrogram: Any,
    add_word_timestamps: Any,
) -> list[dict[str, Any]]:
    t0 = max(0.0, float(group[0][0]) - ALIGN_PAD_S)
    t1 = float(group[-1][1]) + ALIGN_PAD_S
    if t1 - t0 > MAX_ALIGN_AUDIO_S:
        t1 = t0 + MAX_ALIGN_AUDIO_S
    a0 = int(t0 * sample_rate)
    a1 = min(int(audio.shape[0]), int(t1 * sample_rate))
    clip = audio[a0:a1]
    if clip.shape[0] < int(0.05 * sample_rate):
        return _interpolate_rows(group)
    mel = log_mel_spectrogram(clip, n_mels=n_mels)
    num_frames = int(mel.shape[-1])
    mel = pad_or_trim(mel, n_frames).to(model.device)
    segs: list[dict[str, Any]] = []
    sources: list[SegmentRow] = []
    skipped: list[SegmentRow] = []
    for row in group:
        start, end, text = row
        tokens = [tok for tok in tokenizer.encode(text) if tok < tokenizer.eot]
        if not tokens:
            skipped.append(row)
            continue
        rel_start = max(0.0, start - t0)
        rel_end = min(t1 - t0, max(rel_start + 0.02, end - t0))
        segs.append(
            {
                "seek": 0,
                "start": rel_start,
                "end": rel_end,
                "text": text,
                "tokens": tokens,
            }
        )
        sources.append(row)
    if not segs:
        return _interpolate_rows(group)
    add_word_timestamps(
        segments=segs,
        model=model,
        tokenizer=tokenizer,
        mel=mel,
        num_frames=min(num_frames, n_frames),
        last_speech_timestamp=0.0,
    )
    aligned: list[dict[str, Any]] = []
    for seg, row in zip(segs, sources, strict=True):
        start, end, text = row
        seg_words = []
        for item in seg.get("words") or []:
            token = str(item.get("word") or "").strip()
            if not token:
                continue
            seg_words.append(
                {
                    "word": token,
                    "start": t0 + float(item["start"]),
                    "end": t0 + float(item["end"]),
                }
            )
        aligned.extend(
            lexicon_words_for_segment(
                start, end, text, dtw_words=seg_words or None
            )
        )
    if skipped:
        aligned.extend(_interpolate_rows(skipped))
    aligned.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return aligned


def payload_from_aligned_words(
    words: list[dict[str, Any]],
    rows: list[SegmentRow],
    *,
    model_name: str,
    audio_path: Path | str,
    align_s: float,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "backend": ALIGNED_BACKEND,
        "word_timestamps": False,
        "audio_path": str(audio_path),
        "align_s": round(align_s, 3),
        "words": words,
        "segments": [
            {"start": start, "end": end, "text": text} for start, end, text in rows
        ],
    }


def load_word_cache(path: Path, *, aligned_only: bool = True) -> list[WordTiming] | None:
    payload = load_word_payload(path)
    if payload is None:
        return None
    if aligned_only and not payload_is_aligned(payload):
        return None
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
    segments: list[SegmentRow] | None = None,
) -> list[WordTiming]:
    cached = load_word_cache(cache_path, aligned_only=True)
    if cached:
        return cached
    import whisper

    loaded = model if model is not None else whisper.load_model(name=model_name)
    rows = list(segments or [])
    if not rows:
        logger.info("segment whisper then align %s model=%s", audio_path, model_name)
        result = loaded.transcribe(
            str(audio_path),
            fp16=False,
            language=language,
            word_timestamps=False,
        )
        rows = segment_rows_from_transcript(result)
    logger.info("aligning %s segments on %s", len(rows), audio_path)
    words, align_s = align_words_to_segments(
        audio_path, rows, model=loaded, language=language
    )
    payload = payload_from_aligned_words(
        words,
        rows,
        model_name=model_name,
        audio_path=audio_path,
        align_s=align_s,
    )
    save_word_cache(cache_path, payload)
    logger.info("aligned %s words in %.1fs -> %s", len(words), align_s, cache_path)
    return words_from_payload(payload)
