#!/usr/bin/env python3
"""Inspection-only heuristic flags vs frozen gold. $0. Does not change cuts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podcast_processor.experiments.ad_region_heuristics import (
    HeuristicReport,
    HeuristicScore,
    lexical_spans,
    loudness_change_spans,
    missed_or_edge_late,
    music_hole_spans,
    position_prior_spans,
    probe_duration_seconds,
    repeated_ngram_spans,
    score_flags_against_gold,
    silence_bracket_spans,
)
from podcast_processor.experiments.quote_anchor_eval import load_json, load_silence_gaps
from podcast_processor.experiments.word_transcripts import (
    load_word_payload,
    segments_from_payload,
    words_from_payload,
)
from podcast_processor.quote_anchor import SegmentSpan

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "docs/experiments/auto_gold/runs/quote-anchor"
ORIG_LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
HOLDOUT_LABELS = RUN / "holdout/labels.json"
AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
HOLDOUT = RUN / "holdout"
WORDS = RUN / "words-aligned"
SILENCE = REPO / "docs/experiments/auto_gold/runs/boundary-snap-eval/silence_cache"
LIVE_ORIG = RUN / "pipeline-startonly-back/production"
LIVE_HOLDOUT = RUN / "holdout/quote-anchor-startonly-back/production"
OUT = RUN / "heuristics-inspect"


def _segments(show_id: str, audio_dir: Path) -> list[SegmentSpan]:
    word_cache = WORDS / f"{show_id}.json"
    payload = load_word_payload(word_cache)
    rows = segments_from_payload(payload) if payload else []
    if rows:
        return [SegmentSpan(a, b, text) for a, b, text in rows if b > a]
    for base in (audio_dir / "transcripts", HOLDOUT / "baseline" / "transcripts"):
        path = base / f"{show_id}.json"
        if path.is_file():
            return [
                SegmentSpan(float(r["start"]), float(r["end"]), str(r.get("text") or ""))
                for r in json.loads(path.read_text(encoding="utf-8"))
            ]
    return []


def _words(show_id: str) -> list[tuple[float, float]]:
    payload = load_word_payload(WORDS / f"{show_id}.json")
    if not payload:
        return []
    return [(w.start, w.end) for w in words_from_payload(payload)]


def _cuts(prod_dir: Path) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    if not prod_dir.is_dir():
        return out
    for path in prod_dir.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        sid = str(row.get("show_id") or path.stem)
        out[sid] = [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
    return out


def _audio_path(show_id: str, downloads: dict) -> Path | None:
    row = downloads.get(show_id) or {}
    raw = row.get("audio_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO / path
    return path if path.is_file() else None


def _row(score: HeuristicScore, name: str) -> dict:
    return {
        "name": name,
        "ad_seconds_hit": round(score.ad_seconds_hit, 2),
        "ad_seconds": round(score.ad_seconds, 2),
        "recall": round(score.recall, 4),
        "not_ad_seconds_flagged": round(score.not_ad_seconds_flagged, 2),
        "not_ad_seconds": round(score.not_ad_seconds, 2),
        "precision": round(score.precision, 4),
        "ads_hit": f"{score.n_ads_hit}/{score.n_ads}",
    }


def main() -> int:
    orig_labels = load_json(ORIG_LABELS)
    holdout_labels = load_json(HOLDOUT_LABELS) if HOLDOUT_LABELS.is_file() else {"windows": []}
    orig_dl = load_json(AUDIO / "downloads.json")
    holdout_dl = load_json(HOLDOUT / "downloads.json") if (HOLDOUT / "downloads.json").is_file() else {}
    downloads = {**orig_dl, **holdout_dl}
    windows = [
        row
        for row in list(orig_labels.get("windows") or []) + list(holdout_labels.get("windows") or [])
        if not row.get("skipped")
    ]
    show_ids = sorted({str(row["show_id"]) for row in windows if downloads.get(row["show_id"], {}).get("audio_path")})
    segs_by_show = {sid: _segments(sid, AUDIO if sid in orig_dl else HOLDOUT) for sid in show_ids}
    duration: dict[str, float | None] = {}
    gaps: dict[str, list[tuple[float, float]]] = {}
    audio_paths: dict[str, Path | None] = {}
    for sid in show_ids:
        audio = _audio_path(sid, downloads)
        audio_paths[sid] = audio
        duration[sid] = probe_duration_seconds(audio) if audio else None
        if audio:
            loaded = load_silence_gaps(sid, audio, SILENCE)
            gaps[sid] = [(g.start, g.end) for g in loaded]
        else:
            gaps[sid] = []

    ngram = repeated_ngram_spans(segs_by_show)
    flags: dict[str, dict[str, list[tuple[float, float]]]] = {
        "lexical_any": {},
        "lexical_promo_code": {},
        "lexical_brought_to_you_by": {},
        "lexical_support_comes_from": {},
        "lexical_dot_com_slash": {},
        "lexical_url": {},
        "lexical_terms_apply": {},
        "lexical_sponsor": {},
        "audio_silence_bracket": {},
        "audio_music_hole": {},
        "audio_loudness_change": {},
        "repeat_ngram": {},
        "position_preroll": {},
        "position_mid": {},
        "position_end": {},
        "position_any": {},
    }
    for sid in show_ids:
        lex = lexical_spans(segs_by_show[sid], duration=duration[sid])
        for name, spans in lex.items():
            key = name if name == "lexical_any" else f"lexical_{name}"
            if key in flags:
                flags[key][sid] = spans
        flags["audio_silence_bracket"][sid] = silence_bracket_spans(
            segs_by_show[sid], gaps[sid]
        )
        flags["audio_music_hole"][sid] = music_hole_spans(
            segs_by_show[sid], _words(sid)
        )
        flags["audio_loudness_change"][sid] = loudness_change_spans(
            audio_paths[sid], duration[sid]
        )
        flags["repeat_ngram"][sid] = ngram.get(sid) or []
        priors = position_prior_spans(duration[sid])
        for name, spans in priors.items():
            flags[name][sid] = spans

    scored_windows = [row for row in windows if str(row["show_id"]) in show_ids]
    table: list[dict] = []
    report = HeuristicReport()
    for name, by_show in flags.items():
        score = score_flags_against_gold(scored_windows, by_show)
        score.name = name
        report.scores.append(score)
        table.append(_row(score, name))

    live_cuts = {**_cuts(LIVE_ORIG), **_cuts(LIVE_HOLDOUT)}
    report.missed = missed_or_edge_late(scored_windows, live_cuts, flags)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_shows": len(show_ids),
        "shows": show_ids,
        "table": table,
        "missed_or_edge_late": [
            {
                "show_id": item.show_id,
                "start": round(item.start, 3),
                "end": round(item.end, 3),
                "kind": item.kind,
                "covered": round(item.covered, 2),
                "span_seconds": round(item.span_seconds, 2),
                "heuristics": item.heuristics,
            }
            for item in report.missed
        ],
        "changes_cuts": False,
        "gemini_usd": 0.0,
    }
    (OUT / "heuristics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| heuristic | ad-sec hit | recall | not_ad flagged | precision | ads hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table:
        lines.append(
            f"| {row['name']} | {row['ad_seconds_hit']:.1f}/{row['ad_seconds']:.1f} | "
            f"{100.0 * row['recall']:.1f}% | {row['not_ad_seconds_flagged']:.1f}s | "
            f"{100.0 * row['precision']:.1f}% | {row['ads_hit']} |"
        )
    (OUT / "HEURISTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
