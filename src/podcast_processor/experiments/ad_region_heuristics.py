"""Inspection-only ad-region flags. Must not be imported by the cutter.

Cheap $0 flags scored against frozen gold. Never writes cuts.
"""

from __future__ import annotations

import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_processor.experiments.auto_gold.audio_ops import FFMPEG, probe_duration_seconds
from podcast_processor.experiments.auto_gold.scorecard import (
    covered_seconds,
    inner_span_pairs,
    merge_spans,
    overlap_seconds,
)
from podcast_processor.quote_anchor import SegmentSpan, normalize_tokens

# (a) lexical
PROMO_CODE_RE = re.compile(
    r"\b(promo code|use code|code [A-Z0-9]{3,}|coupon)\b", re.I
)
BROUGHT_RE = re.compile(r"\bbrought to you by\b", re.I)
SUPPORT_RE = re.compile(
    r"\bsupport for (this (podcast|show)|npr|npr news) comes from\b", re.I
)
DOT_COM_SLASH_RE = re.compile(r"\bdot com slash\b|\.com/", re.I)
URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+|\b\w+\.(?:com|org|net|io)\b", re.I
)
TERMS_RE = re.compile(r"\bterms apply\b", re.I)
SPONSOR_RE = re.compile(r"\bsponsor(?:ed|s|ship)?\b", re.I)

LEXICAL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("promo_code", PROMO_CODE_RE),
    ("brought_to_you_by", BROUGHT_RE),
    ("support_comes_from", SUPPORT_RE),
    ("dot_com_slash", DOT_COM_SLASH_RE),
    ("url", URL_RE),
    ("terms_apply", TERMS_RE),
    ("sponsor", SPONSOR_RE),
)

# Expand a lexical hit by this many seconds on each side so a cue can cover
# the surrounding read without becoming a whole-show flag.
LEXICAL_PAD_S = 8.0
NGRAM_N = 8
NGRAM_PAD_S = 6.0
SILENCE_BRACKET_S = 0.8
MUSIC_HOLE_S = 5.0
LOUDNESS_RATIO = 2.0
PREROLL_S = 90.0
ENDROLL_S = 90.0


@dataclass
class HeuristicScore:
    name: str
    ad_seconds: float
    ad_seconds_hit: float
    recall: float
    not_ad_seconds: float
    not_ad_seconds_flagged: float
    precision: float
    n_ads: int
    n_ads_hit: int


@dataclass
class MissedAdNote:
    show_id: str
    start: float
    end: float
    kind: str
    covered: float
    span_seconds: float
    heuristics: list[str]


@dataclass
class HeuristicReport:
    scores: list[HeuristicScore] = field(default_factory=list)
    missed: list[MissedAdNote] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _pad_span(start: float, end: float, pad: float, duration: float | None) -> tuple[float, float]:
    lo = max(0.0, start - pad)
    hi = end + pad
    if duration is not None:
        hi = min(hi, duration)
    return lo, hi


def lexical_spans(
    segments: list[SegmentSpan],
    *,
    duration: float | None = None,
    pad_s: float = LEXICAL_PAD_S,
) -> dict[str, list[tuple[float, float]]]:
    hits: dict[str, list[tuple[float, float]]] = {name: [] for name, _ in LEXICAL_RULES}
    union: list[tuple[float, float]] = []
    for seg in segments:
        text = seg.text or ""
        for name, pattern in LEXICAL_RULES:
            if pattern.search(text):
                span = _pad_span(seg.start, seg.end, pad_s, duration)
                hits[name].append(span)
                union.append(span)
    hits["lexical_any"] = merge_spans(union)
    return {name: merge_spans(spans) for name, spans in hits.items()}


def silence_bracket_spans(
    segments: list[SegmentSpan],
    gaps: list[tuple[float, float]],
    *,
    min_silence_s: float = SILENCE_BRACKET_S,
) -> list[tuple[float, float]]:
    """Regions of speech bounded by long silences on both sides."""
    long_gaps = [(a, b) for a, b in gaps if b - a >= min_silence_s]
    if len(long_gaps) < 2:
        return []
    spans: list[tuple[float, float]] = []
    for i, (_g0, g1) in enumerate(long_gaps[:-1]):
        n0, n1 = long_gaps[i + 1]
        inner = (g1, n0)
        if inner[1] - inner[0] < 3.0:
            continue
        speech = sum(
            overlap_seconds(inner[0], inner[1], seg.start, seg.end) for seg in segments
        )
        if speech >= 2.0:
            spans.append(inner)
    return merge_spans(spans)


def music_hole_spans(
    segments: list[SegmentSpan],
    words: list[tuple[float, float]],
    *,
    min_hole_s: float = MUSIC_HOLE_S,
) -> list[tuple[float, float]]:
    """Segment-speech ranges with no word timings (beds/stingers)."""
    if not segments:
        return []
    holes: list[tuple[float, float]] = []
    for seg in segments:
        covered = covered_seconds(seg.start, seg.end, list(words)) if words else 0.0
        dur = seg.end - seg.start
        if dur >= min_hole_s and covered < 0.5:
            holes.append((seg.start, seg.end))
        elif dur >= 0.4 and covered < 0.15 and dur < min_hole_s:
            holes.append((seg.start, seg.end))
    # Also gaps between words inside the episode
    ordered = sorted(words)
    for i in range(len(ordered) - 1):
        gap = ordered[i + 1][0] - ordered[i][1]
        if gap >= min_hole_s:
            holes.append((ordered[i][1], ordered[i + 1][0]))
    return merge_spans(holes)


def _pcm_rms_frames(audio: Path, hop_s: float = 1.0, sr: int = 8000) -> list[float]:
    import subprocess

    cmd = [
        FFMPEG,
        "-v",
        "error",
        "-i",
        str(audio),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-",
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return []
    pcm = completed.stdout
    hop = max(1, int(sr * hop_s))
    n = len(pcm) // 2
    if n < hop:
        return []
    samples = struct.unpack("<" + "h" * n, pcm[: n * 2])
    frames: list[float] = []
    for i in range(0, n - hop + 1, hop):
        sl = samples[i : i + hop]
        acc = sum(float(s) * float(s) for s in sl) / float(len(sl))
        frames.append(math.sqrt(acc))
    return frames


def loudness_change_spans(
    audio: Path | None,
    duration: float | None,
    *,
    ratio: float = LOUDNESS_RATIO,
) -> list[tuple[float, float]]:
    if audio is None or not audio.is_file():
        return []
    if duration is not None and duration > 7200.0:
        return []
    frames = _pcm_rms_frames(audio)
    if len(frames) < 12:
        return []
    spans: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, rms in enumerate(frames):
        lo = max(0, i - 10)
        neighborhood = [frames[j] for j in range(lo, i) if frames[j] > 1.0]
        if not neighborhood:
            if run_start is not None and i - run_start >= 4:
                spans.append((float(run_start), float(i)))
            run_start = None
            continue
        med = sorted(neighborhood)[len(neighborhood) // 2]
        changed = med > 1.0 and (rms / med >= ratio or med / max(rms, 1.0) >= ratio)
        if changed:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start >= 4:
                end = float(i)
                if duration is not None:
                    end = min(end, duration)
                spans.append((float(run_start), end))
            run_start = None
    if run_start is not None and len(frames) - run_start >= 4:
        end = float(len(frames))
        if duration is not None:
            end = min(end, duration)
        spans.append((float(run_start), end))
    return merge_spans(spans)


def _publisher_bucket(show_id: str) -> str:
    if show_id.startswith("npr_") or show_id in {"planet_money"}:
        return "npr"
    if show_id.startswith("bbc_"):
        return "bbc"
    return show_id


def repeated_ngram_spans(
    by_show: dict[str, list[SegmentSpan]],
    *,
    n: int = NGRAM_N,
    pad_s: float = NGRAM_PAD_S,
) -> dict[str, list[tuple[float, float]]]:
    """n-grams that repeat across episodes of the same publisher/show."""
    grams: dict[str, dict[tuple[str, ...], list[tuple[str, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for show_id, segments in by_show.items():
        bucket = _publisher_bucket(show_id)
        for seg in segments:
            tokens = normalize_tokens(seg.text or "")
            if len(tokens) < n:
                continue
            for i in range(0, len(tokens) - n + 1):
                gram = tuple(tokens[i : i + n])
                grams[bucket][gram].append((show_id, seg.start, seg.end))
    out: dict[str, list[tuple[float, float]]] = {sid: [] for sid in by_show}
    for _bucket, gram_hits in grams.items():
        for _gram, rows in gram_hits.items():
            shows = {sid for sid, _, _ in rows}
            if len(shows) < 2:
                continue
            for sid, start, end in rows:
                out[sid].append((max(0.0, start - pad_s), end + pad_s))
    return {sid: merge_spans(spans) for sid, spans in out.items()}


def position_prior_spans(duration: float | None) -> dict[str, list[tuple[float, float]]]:
    dur = float(duration or 0.0)
    preroll = [(0.0, min(PREROLL_S, dur))] if dur > 0 else []
    end = [(max(0.0, dur - ENDROLL_S), dur)] if dur > 0 else []
    mid: list[tuple[float, float]] = []
    if dur > 180.0:
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            center = dur * frac
            mid.append((max(0.0, center - 20.0), min(dur, center + 20.0)))
    return {
        "position_preroll": preroll,
        "position_end": end,
        "position_mid": merge_spans(mid),
        "position_any": merge_spans(preroll + end + mid),
    }


def score_flags_against_gold(
    windows: list[dict[str, Any]],
    flags_by_show: dict[str, list[tuple[float, float]]],
) -> HeuristicScore:
    name = ""
    ad_seconds = 0.0
    ad_hit = 0.0
    not_ad_seconds = 0.0
    not_ad_flagged = 0.0
    n_ads = 0
    n_ads_hit = 0
    for row in windows:
        show_id = str(row["show_id"])
        flags = flags_by_show.get(show_id) or []
        start = float(row["start"])
        end = float(row["end"])
        is_ad = bool(row["is_ad"])
        if is_ad:
            spans = inner_span_pairs(row, start, end)
            for s0, s1 in spans:
                dur = max(0.0, s1 - s0)
                ad_seconds += dur
                hit = covered_seconds(s0, s1, flags)
                ad_hit += hit
                n_ads += 1
                if dur > 0 and hit / dur >= 0.25:
                    n_ads_hit += 1
        else:
            dur = max(0.0, end - start)
            not_ad_seconds += dur
            not_ad_flagged += covered_seconds(start, end, flags)
    recall = ad_hit / ad_seconds if ad_seconds > 0 else 0.0
    flagged = ad_hit + not_ad_flagged
    precision = ad_hit / flagged if flagged > 0 else 0.0
    return HeuristicScore(
        name=name,
        ad_seconds=ad_seconds,
        ad_seconds_hit=ad_hit,
        recall=recall,
        not_ad_seconds=not_ad_seconds,
        not_ad_seconds_flagged=not_ad_flagged,
        precision=precision,
        n_ads=n_ads,
        n_ads_hit=n_ads_hit,
    )


def missed_or_edge_late(
    windows: list[dict[str, Any]],
    cuts_by_show: dict[str, list[tuple[float, float]]],
    flags: dict[str, dict[str, list[tuple[float, float]]]],
) -> list[MissedAdNote]:
    notes: list[MissedAdNote] = []
    for row in windows:
        if not row.get("is_ad") or row.get("skipped"):
            continue
        show_id = str(row["show_id"])
        start = float(row["start"])
        end = float(row["end"])
        cuts = cuts_by_show.get(show_id) or []
        spans = inner_span_pairs(row, start, end)
        for s0, s1 in spans:
            covered = covered_seconds(s0, s1, cuts)
            dur = max(0.0, s1 - s0)
            frac = covered / dur if dur > 0 else 0.0
            overlapping = [
                (a, b) for a, b in merge_spans(cuts) if overlap_seconds(a, b, s0, s1) > 0
            ]
            kind = ""
            if frac < 0.5:
                kind = "missed"
            elif overlapping and max(b for _, b in overlapping) < s1 - 2.0:
                kind = "edge_late"
            if not kind:
                continue
            hit_names = [
                name
                for name, by_show in flags.items()
                if covered_seconds(s0, s1, by_show.get(show_id) or []) > 0.5
            ]
            notes.append(
                MissedAdNote(
                    show_id=show_id,
                    start=s0,
                    end=s1,
                    kind=kind,
                    covered=covered,
                    span_seconds=dur,
                    heuristics=hit_names,
                )
            )
    return notes
