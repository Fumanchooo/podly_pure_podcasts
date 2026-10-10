"""$0 loudness and speaker-change cues for the richer ad spotter.

CPU-only: ffmpeg → float32 PCM, numpy RMS/LUFS-ish and MFCC-lite (log-mel
centroid + pitch-ish ZCR). No extra ML deps. Results cache next to audio.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from podcast_processor.ad_spotter import SpotHit

SAMPLE_RATE = 16000
FRAME_S = 0.50
HOP_S = 0.25
LOUD_DELTA_DB = 4.0
SPEAKER_Z = 2.8
MIN_RUN_S = 1.5
MAX_RUN_S = 90.0


def load_mono_f32(path: Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32), sr
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio, sr


def _frames(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = max(1, int(FRAME_S * sr))
    hop = max(1, int(HOP_S * sr))
    if audio.size < frame:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    n = 1 + (audio.size - frame) // hop
    rms = np.empty(n, dtype=np.float64)
    zcr = np.empty(n, dtype=np.float64)
    centroid = np.empty(n, dtype=np.float64)
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
    for i in range(n):
        sl = audio[i * hop : i * hop + frame].astype(np.float64)
        rms[i] = np.sqrt(np.mean(sl * sl) + 1e-12)
        signs = np.sign(sl)
        zcr[i] = np.mean(np.abs(np.diff(signs))) * 0.5
        spec = np.abs(np.fft.rfft(sl * window))
        denom = float(np.sum(spec) + 1e-12)
        centroid[i] = float(np.sum(freqs * spec) / denom)
    return rms, zcr, centroid


def _db(rms: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(rms, 1e-8))


def _runs(
    mask: np.ndarray, hop_s: float, *, min_s: float, max_s: float
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        start = i * hop_s
        end = j * hop_s + (FRAME_S - hop_s)
        dur = end - start
        if min_s <= dur <= max_s:
            spans.append((start, end))
        i = j
    return spans


def loudness_hits_from_rms(
    rms: np.ndarray, *, hop_s: float = HOP_S, delta_db: float = LOUD_DELTA_DB
) -> list[SpotHit]:
    if rms.size < 8:
        return []
    db = _db(rms)
    # Surrounding host level: ±8s ring, excluding ±1.5s so a loud spot
    # is not compared to itself.
    ring = max(3, int(8.0 / hop_s))
    hole = max(1, int(1.5 / hop_s))
    med = np.empty_like(db)
    n = len(db)
    for i in range(n):
        lo = max(0, i - ring)
        hi = min(n, i + ring + 1)
        inner_lo = max(0, i - hole)
        inner_hi = min(n, i + hole + 1)
        surround = np.concatenate([db[lo:inner_lo], db[inner_hi:hi]])
        med[i] = np.median(surround) if surround.size else db[i]
    mask = (db - med) >= delta_db
    hits: list[SpotHit] = []
    for start, end in _runs(mask, hop_s, min_s=MIN_RUN_S, max_s=MAX_RUN_S):
        hits.append(
            SpotHit(
                start=start,
                end=end,
                kind="loudness",
                cue="rms_step",
                weight=1.15,
            )
        )
    return hits


def speaker_hits_from_features(
    rms: np.ndarray,
    zcr: np.ndarray,
    centroid: np.ndarray,
    *,
    hop_s: float = HOP_S,
    z_thresh: float = SPEAKER_Z,
) -> list[SpotHit]:
    n = min(len(rms), len(zcr), len(centroid))
    if n < 16:
        return []
    feats = np.stack([_db(rms[:n]), zcr[:n], centroid[:n] / 1000.0], axis=1)
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0) + 1e-6
    z = np.sqrt(np.sum(((feats - med) / mad) ** 2, axis=1))
    voiced = _db(rms[:n]) > (np.median(_db(rms[:n])) - 8.0)
    mask = (z >= z_thresh) & voiced
    hits: list[SpotHit] = []
    for start, end in _runs(mask, hop_s, min_s=MIN_RUN_S, max_s=MAX_RUN_S):
        hits.append(
            SpotHit(
                start=start,
                end=end,
                kind="speaker",
                cue="voice_change",
                weight=1.25,
            )
        )
    return hits


def _cache_path(audio_path: Path) -> Path:
    return audio_path.with_name(audio_path.name + ".spot_signals.json")


def _cache_key(audio_path: Path) -> str:
    st = audio_path.stat()
    blob = f"{audio_path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha1(blob.encode()).hexdigest()


def audio_spot_hits(
    audio_path: Path | str | None,
    *,
    cache: bool = True,
) -> tuple[list[SpotHit], list[SpotHit]]:
    """Return (loudness_hits, speaker_hits). Empty if audio is missing."""
    if not audio_path:
        return [], []
    path = Path(audio_path)
    if not path.is_file():
        return [], []
    cache_file = _cache_path(path)
    key = _cache_key(path)
    if cache and cache_file.is_file():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload.get("key") == key:
                loud = [
                    SpotHit(**row) for row in payload.get("loudness", [])
                ]
                speak = [
                    SpotHit(**row) for row in payload.get("speaker", [])
                ]
                return loud, speak
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass
    audio, sr = load_mono_f32(path)
    rms, zcr, centroid = _frames(audio, sr)
    hop = HOP_S
    loud = loudness_hits_from_rms(rms, hop_s=hop)
    speak = speaker_hits_from_features(rms, zcr, centroid, hop_s=hop)
    if cache:
        cache_file.write_text(
            json.dumps(
                {
                    "key": key,
                    "loudness": [
                        {
                            "start": h.start,
                            "end": h.end,
                            "kind": h.kind,
                            "cue": h.cue,
                            "weight": h.weight,
                            "word_from": h.word_from,
                            "word_to": h.word_to,
                        }
                        for h in loud
                    ],
                    "speaker": [
                        {
                            "start": h.start,
                            "end": h.end,
                            "kind": h.kind,
                            "cue": h.cue,
                            "weight": h.weight,
                            "word_from": h.word_from,
                            "word_to": h.word_to,
                        }
                        for h in speak
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return loud, speak
