"""Cactus Whistle STT: 16 kHz mono, ≤30 s per pass, word timestamps.

Chunk full episodes into overlapping windows, stitch words, and flag
320-token / fast-speech truncation plus 30 s seam drops. Optional
sponsor-lexicon keyword biasing. Not thread-safe (engine lock).
"""

from __future__ import annotations

import json
import logging
import os
import resource
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

from podcast_processor.quote_anchor import WordTiming, normalize_tokens

logger = logging.getLogger(__name__)

WHISTLE_BACKEND = "whistle"
SAMPLE_RATE = 16000
MAX_CLIP_S = 28.0
OVERLAP_S = 1.5
TOKEN_CAP = 320
TOKEN_CAP_WARN = 280
SEAM_GAP_S = 0.4
KEYWORD_CAP = 100
HOLE_S = 5.0

TranscribeFn = Callable[..., dict[str, Any]]

_lock = threading.Lock()


def _silence_telemetry() -> None:
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")


def payload_is_whistle(payload: dict[str, Any] | None) -> bool:
    return bool(payload) and str(payload.get("backend") or "") == WHISTLE_BACKEND


def sponsor_keywords(limit: int = KEYWORD_CAP) -> list[str]:
    from podcast_processor.sponsor_lexicon import SPONSOR_NAMES

    out: list[str] = []
    seen: set[str] = set()
    for name in SPONSOR_NAMES:
        key = str(name).strip()
        folded = key.lower()
        if not key or folded in seen:
            continue
        seen.add(folded)
        out.append(key)
        if len(out) >= limit:
            break
    return out


def load_mono_16k(audio_path: Path | str) -> Any:
    """Load 16 kHz mono float32 via Whisper's ffmpeg path (no soxr/[mic])."""
    from whisper.audio import load_audio

    return load_audio(str(audio_path))


def _needle_transcribe(
    audio: Any,
    *,
    language: str | None = "en",
    keywords: list[str] | None = None,
    word_timestamps: bool = True,
) -> dict[str, Any]:
    _silence_telemetry()
    import needle

    return needle.transcribe(
        audio,
        language=language,
        keywords=keywords,
        word_timestamps=word_timestamps,
    )


def transcribe_clip(
    samples: Any,
    *,
    language: str | None = "en",
    keywords: list[str] | None = None,
    word_timestamps: bool = True,
    transcribe_fn: TranscribeFn | None = None,
) -> dict[str, Any]:
    fn = transcribe_fn or _needle_transcribe
    n = int(samples.shape[-1]) if hasattr(samples, "shape") else len(samples)
    max_n = int(MAX_CLIP_S * SAMPLE_RATE)
    if n > int(30.0 * SAMPLE_RATE):
        raise ValueError(f"Whistle clip is {n / SAMPLE_RATE:.2f}s; max 30s")
    if n > max_n:
        logger.warning("clip %.2fs over %.1fs window; engine max is 30s", n / SAMPLE_RATE, MAX_CLIP_S)
    with _lock:
        return fn(
            samples,
            language=language,
            keywords=keywords,
            word_timestamps=word_timestamps,
        )


def _clip_windows(n_samples: int) -> list[tuple[int, int]]:
    if n_samples <= 0:
        return []
    win = int(MAX_CLIP_S * SAMPLE_RATE)
    hop = int((MAX_CLIP_S - OVERLAP_S) * SAMPLE_RATE)
    if hop <= 0:
        hop = win
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(n_samples, start + win)
        windows.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return windows


def _normalize_words(raw: list[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for row in raw:
        token = str(row.get("word") or row.get("text") or "").strip()
        if not token:
            continue
        start = float(row["start"]) + offset_s if row.get("start") is not None else offset_s
        if row.get("end") is not None:
            end = float(row["end"]) + offset_s
        else:
            end = start + 0.08
        words.append(
            {
                "word": token,
                "start": start,
                "end": max(start + 0.02, end),
                "probability": float(row.get("probability") or 0.0),
            }
        )
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return words


def _same_token(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = normalize_tokens(str(left.get("word") or ""))
    b = normalize_tokens(str(right.get("word") or ""))
    return bool(a) and a == b


def stitch_chunk_words(
    chunks: list[tuple[float, float, list[dict[str, Any]]]],
    *,
    overlap_s: float = OVERLAP_S,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first copy of overlap tokens; return (words, seam_gap events)."""
    if not chunks:
        return [], []
    out: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for i, (t0, _t1, words) in enumerate(chunks):
        if i == 0:
            out.extend(words)
            continue
        seam = t0 + overlap_s / 2.0
        while out and float(out[-1]["start"]) >= seam:
            out.pop()
        kept = [w for w in words if float(w["start"]) >= seam - 0.02]
        if out and kept:
            hole = float(kept[0]["start"]) - float(out[-1]["end"])
            if hole >= SEAM_GAP_S:
                gaps.append(
                    {
                        "seam_s": round(seam, 3),
                        "gap_s": round(hole, 3),
                        "left": out[-1]["word"],
                        "right": kept[0]["word"],
                    }
                )
        for word in kept:
            if (
                out
                and _same_token(out[-1], word)
                and abs(float(out[-1]["start"]) - float(word["start"])) < 0.2
            ):
                continue
            out.append(word)
    return out, gaps


def clip_truncated(
    *,
    n_words: int,
    last_word_end: float,
    clip_start: float,
    clip_end: float,
) -> bool:
    """True when the 320-token cap is likely hit, not when a clip just ends in pause."""
    del last_word_end, clip_start, clip_end
    return n_words >= TOKEN_CAP_WARN


@dataclass
class ClipStat:
    start: float
    end: float
    n_words: int
    last_word_end: float
    truncated: bool
    wall_s: float
    n_chars: int


@dataclass
class WhistleTranscript:
    words: list[dict[str, Any]]
    text: str
    clips: list[ClipStat] = field(default_factory=list)
    seam_gaps: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    cpu_s: float = 0.0
    audio_s: float = 0.0
    keywords: list[str] = field(default_factory=list)
    n_truncated: int = 0

    def payload(self, *, audio_path: Path | str, model: str = "whistle") -> dict[str, Any]:
        return {
            "model": model,
            "backend": WHISTLE_BACKEND,
            "word_timestamps": True,
            "audio_path": str(audio_path),
            "wall_s": round(self.wall_s, 3),
            "cpu_s": round(self.cpu_s, 3),
            "audio_s": round(self.audio_s, 3),
            "keywords": list(self.keywords),
            "n_truncated": self.n_truncated,
            "seam_gaps": list(self.seam_gaps),
            "clips": [asdict(clip) for clip in self.clips],
            "words": [
                {"word": w["word"], "start": w["start"], "end": w["end"]}
                for w in self.words
            ],
            "segments": _segments_from_words(self.words),
            "text": self.text,
        }


def _segments_from_words(
    words: list[dict[str, Any]], *, max_span_s: float = 28.0
) -> list[dict[str, Any]]:
    if not words:
        return []
    segs: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = [words[0]]
    origin = float(words[0]["start"])
    for word in words[1:]:
        if float(word["end"]) - origin <= max_span_s:
            bucket.append(word)
            continue
        segs.append(
            {
                "start": float(bucket[0]["start"]),
                "end": float(bucket[-1]["end"]),
                "text": " ".join(item["word"] for item in bucket),
            }
        )
        bucket = [word]
        origin = float(word["start"])
    segs.append(
        {
            "start": float(bucket[0]["start"]),
            "end": float(bucket[-1]["end"]),
            "text": " ".join(item["word"] for item in bucket),
        }
    )
    return segs


def transcribe_audio(
    audio_path: Path | str,
    *,
    language: str | None = "en",
    keywords: list[str] | None = None,
    transcribe_fn: TranscribeFn | None = None,
    samples: Any | None = None,
) -> WhistleTranscript:
    path = Path(audio_path)
    audio = samples if samples is not None else load_mono_16k(path)
    n = int(audio.shape[0]) if hasattr(audio, "shape") else len(audio)
    audio_s = n / float(SAMPLE_RATE)
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    chunks: list[tuple[float, float, list[dict[str, Any]]]] = []
    clips: list[ClipStat] = []
    for a0, a1 in _clip_windows(n):
        t0 = a0 / float(SAMPLE_RATE)
        t1 = a1 / float(SAMPLE_RATE)
        clip_t0 = time.perf_counter()
        result = transcribe_clip(
            audio[a0:a1],
            language=language,
            keywords=keywords,
            transcribe_fn=transcribe_fn,
        )
        raw_words = list(result.get("words") or [])
        words = _normalize_words(raw_words, t0)
        last_end = float(words[-1]["end"]) if words else t0
        truncated = clip_truncated(
            n_words=len(words),
            last_word_end=last_end,
            clip_start=t0,
            clip_end=t1,
        )
        if truncated:
            logger.warning(
                "whistle truncation? %s %.1f-%.1fs words=%s last_end=%.2f",
                path.name,
                t0,
                t1,
                len(words),
                last_end,
            )
        clips.append(
            ClipStat(
                start=t0,
                end=t1,
                n_words=len(words),
                last_word_end=round(last_end, 3),
                truncated=truncated,
                wall_s=round(time.perf_counter() - clip_t0, 3),
                n_chars=len(str(result.get("text") or "")),
            )
        )
        chunks.append((t0, t1, words))
    stitched, seam_gaps = stitch_chunk_words(chunks)
    wall_s = time.perf_counter() - started
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    text = " ".join(w["word"] for w in stitched)
    return WhistleTranscript(
        words=stitched,
        text=text,
        clips=clips,
        seam_gaps=seam_gaps,
        wall_s=wall_s,
        cpu_s=cpu_s,
        audio_s=audio_s,
        keywords=list(keywords or []),
        n_truncated=sum(1 for clip in clips if clip.truncated),
    )


def save_whistle_cache(path: Path, transcript: WhistleTranscript, audio_path: Path | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcript.payload(audio_path=audio_path), indent=2) + "\n",
        encoding="utf-8",
    )


def transcribe_word_timestamps_whistle(
    audio_path: Path,
    cache_path: Path,
    *,
    keywords: list[str] | None = None,
    language: str | None = "en",
    transcribe_fn: TranscribeFn | None = None,
) -> list[WordTiming]:
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload_is_whistle(payload):
            words = [
                WordTiming(word=str(row["word"]), start=float(row["start"]), end=float(row["end"]))
                for row in payload.get("words") or []
                if str(row.get("word") or "").strip()
            ]
            if words:
                return words
    transcript = transcribe_audio(
        audio_path,
        language=language,
        keywords=keywords,
        transcribe_fn=transcribe_fn,
    )
    save_whistle_cache(cache_path, transcript, audio_path)
    logger.info(
        "whistle %s words=%s wall=%.1fs cpu=%.1fs audio=%.1fs trunc=%s seams=%s -> %s",
        audio_path.name,
        len(transcript.words),
        transcript.wall_s,
        transcript.cpu_s,
        transcript.audio_s,
        transcript.n_truncated,
        len(transcript.seam_gaps),
        cache_path,
    )
    return [
        WordTiming(word=str(row["word"]), start=float(row["start"]), end=float(row["end"]))
        for row in transcript.words
    ]


def word_holes(words: list[WordTiming], *, min_s: float = HOLE_S) -> list[tuple[float, float]]:
    holes: list[tuple[float, float]] = []
    prev_end = 0.0
    for word in words:
        if word.start - prev_end >= min_s:
            holes.append((prev_end, word.start))
        prev_end = max(prev_end, word.end)
    return holes


def aligned_token_mae(
    left: list[WordTiming],
    right: list[WordTiming],
    *,
    max_skew_s: float = 2.0,
) -> dict[str, float]:
    """Two-pointer match of equal tokens in time order; median |dt| of starts."""
    left_toks = [(normalize_tokens(w.word)[0], w) for w in left if normalize_tokens(w.word)]
    right_toks = [(normalize_tokens(w.word)[0], w) for w in right if normalize_tokens(w.word)]
    i = 0
    j = 0
    deltas: list[float] = []
    while i < len(left_toks) and j < len(right_toks):
        needle, a = left_toks[i]
        tok, b = right_toks[j]
        if needle == tok and abs(b.start - a.start) <= max_skew_s:
            deltas.append(abs(b.start - a.start))
            i += 1
            j += 1
            continue
        if b.start < a.start - 0.35:
            j += 1
            continue
        if a.start < b.start - 0.35:
            i += 1
            continue
        i += 1
        j += 1
    deltas.sort()
    n = len(deltas)
    median = deltas[n // 2] if n else 0.0
    mean = sum(deltas) / float(n or 1)
    return {
        "n_matched": float(n),
        "mae_s": mean,
        "median_abs_s": median,
        "p90_abs_s": deltas[int(0.9 * (n - 1))] if n else 0.0,
        "n_left": float(len(left)),
        "n_right": float(len(right)),
    }
