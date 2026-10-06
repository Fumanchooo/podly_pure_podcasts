#!/usr/bin/env python3
"""Compare word_timestamps=True holes vs segment Whisper speech ($0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_processor.experiments.auto_gold.scorecard import (  # noqa: E402
    inner_span_pairs,
    overlap_seconds,
)
from podcast_processor.experiments.word_transcripts import (  # noqa: E402
    load_word_payload,
    words_from_payload,
)

MIN_HOLE_S = 5.0
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
ORIG_WORDS = REPO / "docs/experiments/auto_gold/runs/quote-anchor/words"
ORIG_SEGS = REPO / "docs/experiments/auto_gold/runs/prod-scorecard/transcripts"
ORIG_LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
HOLD = REPO / "docs/experiments/auto_gold/runs/quote-anchor/holdout"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _seg_rows(path: Path) -> list[tuple[float, float, str]]:
    raw = _load_json(path)
    rows = raw.get("segments") if isinstance(raw, dict) else raw
    out: list[tuple[float, float, str]] = []
    for seg in rows or []:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append((float(seg["start"]), float(seg["end"]), text))
    return out


def _word_holes(words, *, min_hole: float) -> list[tuple[float, float]]:
    holes: list[tuple[float, float]] = []
    prev_end: float | None = None
    for word in words:
        if prev_end is not None:
            gap = word.start - prev_end
            if gap >= min_hole:
                holes.append((prev_end, word.start))
        prev_end = max(prev_end or 0.0, word.end)
    return holes


def _gold_spans(windows: list[dict], show_id: str) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for row in windows:
        if str(row.get("show_id")) != show_id or row.get("skipped"):
            continue
        if not row.get("is_ad"):
            continue
        spans.extend(inner_span_pairs(row, float(row["start"]), float(row["end"])))
    return spans


def _speech_in_hole(
    hole: tuple[float, float], segs: list[tuple[float, float, str]]
) -> list[tuple[float, float]]:
    h0, h1 = hole
    pieces: list[tuple[float, float]] = []
    for a, b, _text in segs:
        lo = max(h0, a)
        hi = min(h1, b)
        if hi > lo:
            pieces.append((lo, hi))
    return pieces


def _covered(t0: float, t1: float, spans: list[tuple[float, float]]) -> float:
    return float(sum(overlap_seconds(t0, t1, a, b) for a, b in spans))


def audit_show(
    show_id: str,
    words_path: Path,
    segs_path: Path,
    gold: list[tuple[float, float]],
) -> dict:
    payload = load_word_payload(words_path)
    words = words_from_payload(payload) if payload else []
    segs = _seg_rows(segs_path)
    holes = []
    speech_s = 0.0
    ad_s = 0.0
    n_ad_overlap = 0
    for hole in _word_holes(words, min_hole=MIN_HOLE_S):
        speech = _speech_in_hole(hole, segs)
        if not speech:
            continue
        hole_speech = float(sum(b - a for a, b in speech))
        speech_s += hole_speech
        ad_hit = _covered(hole[0], hole[1], gold)
        if ad_hit > 0:
            n_ad_overlap += 1
            ad_s += float(sum(_covered(a, b, gold) for a, b in speech))
        holes.append(
            {
                "start": round(hole[0], 2),
                "end": round(hole[1], 2),
                "hole_s": round(hole[1] - hole[0], 2),
                "speech_s": round(hole_speech, 2),
                "ad_s": round(ad_hit, 2),
                "preview": next(
                    (t for a, b, t in segs if min(b, hole[1]) > max(a, hole[0])),
                    "",
                )[:80],
            }
        )
    return {
        "show_id": show_id,
        "n_holes": len(holes),
        "hole_s": round(sum(h["hole_s"] for h in holes), 2),
        "speech_s": round(speech_s, 2),
        "n_ad_overlap": n_ad_overlap,
        "ad_s": round(ad_s, 2),
        "holes": holes,
    }


def main() -> int:
    orig_labels = _load_json(ORIG_LABELS)
    hold_labels = _load_json(HOLD / "labels.json")
    orig_windows = list(orig_labels.get("windows") or [])
    hold_windows = list(hold_labels.get("windows") or [])
    rows = []
    for sid in ORIG_SHOWS:
        rows.append(
            audit_show(
                sid,
                ORIG_WORDS / f"{sid}.json",
                ORIG_SEGS / f"{sid}.json",
                _gold_spans(orig_windows, sid),
            )
        )
    for sid in HOLD_SHOWS:
        segs = HOLD / "baseline" / "transcripts" / f"{sid}.json"
        if sid == "marketplace" and not segs.is_file():
            segs = HOLD / "quote-anchor" / "transcripts" / f"{sid}.json"
        rows.append(
            audit_show(
                sid,
                HOLD / "words" / f"{sid}.json",
                segs,
                _gold_spans(hold_windows, sid),
            )
        )
    summary = {
        "min_hole_s": MIN_HOLE_S,
        "n_shows": len(rows),
        "n_holes": sum(r["n_holes"] for r in rows),
        "hole_s": round(sum(r["hole_s"] for r in rows), 2),
        "speech_s": round(sum(r["speech_s"] for r in rows), 2),
        "n_ad_overlap": sum(r["n_ad_overlap"] for r in rows),
        "ad_s": round(sum(r["ad_s"] for r in rows), 2),
        "original10": {
            "n_holes": sum(r["n_holes"] for r in rows if r["show_id"] in ORIG_SHOWS),
            "speech_s": round(
                sum(r["speech_s"] for r in rows if r["show_id"] in ORIG_SHOWS), 2
            ),
            "n_ad_overlap": sum(
                r["n_ad_overlap"] for r in rows if r["show_id"] in ORIG_SHOWS
            ),
            "ad_s": round(sum(r["ad_s"] for r in rows if r["show_id"] in ORIG_SHOWS), 2),
        },
        "holdout": {
            "n_holes": sum(r["n_holes"] for r in rows if r["show_id"] in HOLD_SHOWS),
            "speech_s": round(
                sum(r["speech_s"] for r in rows if r["show_id"] in HOLD_SHOWS), 2
            ),
            "n_ad_overlap": sum(
                r["n_ad_overlap"] for r in rows if r["show_id"] in HOLD_SHOWS
            ),
            "ad_s": round(sum(r["ad_s"] for r in rows if r["show_id"] in HOLD_SHOWS), 2),
        },
        "shows": [
            {k: r[k] for k in ("show_id", "n_holes", "hole_s", "speech_s", "n_ad_overlap", "ad_s")}
            for r in rows
        ],
        "notable": [
            hole | {"show_id": r["show_id"]}
            for r in rows
            for hole in r["holes"]
            if hole["ad_s"] > 0 or hole["speech_s"] >= 8.0
        ],
    }
    out = HOLD.parent / "word_drop_audit.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
