#!/usr/bin/env python3
"""Per-ad error table: baseline vs pure-quote vs prompt-v2. $0, no LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podcast_processor.experiments.auto_gold.scorecard import (
    covered_seconds,
    inner_span_pairs,
    merge_spans,
    overlap_seconds,
    span_boundary_errors,
    window_over_under,
)
from podcast_processor.experiments.quote_anchor_eval import load_json, load_silence_gaps
from podcast_processor.experiments.word_transcripts import load_word_payload, words_from_payload
from podcast_processor.boundary_snap import refine_cuts

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
POLICY_PROD = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/production"
PURE = REPO / "docs/experiments/auto_gold/runs/quote-anchor/pure-v1/production"
V2 = REPO / "docs/experiments/auto_gold/runs/quote-anchor/pure-v2/production"
WORDS = REPO / "docs/experiments/auto_gold/runs/quote-anchor/words"
AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
SILENCE = REPO / "docs/experiments/auto_gold/runs/boundary-snap-eval/silence_cache"
OUT = REPO / "docs/experiments/auto_gold/runs/quote-anchor/diagnosis"


def _cuts(path: Path) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for item in path.glob("*.json"):
        row = json.loads(item.read_text(encoding="utf-8"))
        out[str(row.get("show_id") or item.stem)] = [
            (float(a), float(b)) for a, b in row.get("cut_segments_s") or []
        ]
    return out


def _quotes(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for item in path.glob("*.json"):
        row = json.loads(item.read_text(encoding="utf-8"))
        out[str(row.get("show_id") or item.stem)] = list(row.get("quotes") or [])
    return out


def snap_baseline() -> dict[str, list[tuple[float, float]]]:
    downloads = load_json(AUDIO / "downloads.json")
    snapped: dict[str, list[tuple[float, float]]] = {}
    for item in POLICY_PROD.glob("*.json"):
        row = json.loads(item.read_text(encoding="utf-8"))
        sid = str(row.get("show_id") or item.stem)
        if sid in {"bill_simmons", "marketplace"}:
            continue
        cuts = [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        audio = Path(str(downloads[sid]["audio_path"]))
        if not audio.is_absolute():
            audio = REPO / audio
        gaps = load_silence_gaps(sid, audio, SILENCE)
        snapped[sid] = refine_cuts(cuts, audio_path=audio, gaps=gaps, radius=2.0)
    return snapped


def words_around(words, t: float, n: int = 8) -> str:
    if not words:
        return ""
    idx = 0
    while idx < len(words) and words[idx].end < t:
        idx += 1
    lo = max(0, idx - n)
    hi = min(len(words), idx + n)
    return " ".join(w.word for w in words[lo:hi])


def overlapping_quotes(quotes: list[dict], s0: float, s1: float) -> list[dict]:
    hits = []
    for q in quotes:
        if overlap_seconds(s0, s1, float(q["start"]), float(q["end"])) > 0:
            hits.append(q)
    return hits


def classify(
    *,
    s0: float,
    s1: float,
    c0: float | None,
    c1: float | None,
    under: float,
    over: float,
    n_v1: int,
    n_v2: int,
    n_span_in_window: int,
    excerpt: str,
) -> str:
    if c0 is None:
        return "d_split_merge_or_miss"
    late = c0 - s0
    early_end = s1 - c1
    over_start = s0 - c0
    over_end = c1 - s1
    # Label-off heuristic: excerpt is show content / no sponsor cues, and
    # both quote systems miss most of the span.
    excerpt_l = (excerpt or "").lower()
    sponsorish = any(
        p in excerpt_l
        for p in (
            "sponsor",
            "brought to you",
            "support for",
            "promo",
            "code",
            ".com",
            "subscribe",
            "membership",
            "donate",
            "patreon",
            "this message comes",
        )
    )
    if under > 8 and over < 0.5 and not sponsorish and (s1 - s0) > 20:
        return "f_label_off"
    if n_v1 != n_v2 and abs(n_v1 - n_v2) >= 1 and (n_span_in_window > 1 or n_v1 > 1 or n_v2 > 1):
        if under > 2 or over > 2:
            return "d_split_merge"
    if over_start > 1.0 or over_end > 1.0:
        return "c_overcut_show"
    if late > 1.5 and late >= early_end:
        return "a_missed_leadin"
    if early_end > 1.5:
        return "b_missed_outro"
    if late > 0.8:
        return "a_missed_leadin"
    if over > 0.5:
        return "c_overcut_show"
    if under > 0.5:
        return "b_missed_outro"
    return "ok"


def main() -> int:
    labels = load_json(LABELS)
    baseline = snap_baseline()
    v1_cuts = _cuts(PURE)
    v2_cuts = _cuts(V2)
    v1_q = _quotes(PURE)
    v2_q = _quotes(V2)
    rows: list[dict] = []
    class_under: dict[str, float] = {}
    class_over: dict[str, float] = {}
    class_n: dict[str, int] = {}

    for win in labels["windows"]:
        if win.get("skipped") or not win.get("is_ad"):
            continue
        sid = win["show_id"]
        if sid in {"bill_simmons", "marketplace"}:
            continue
        w0, w1 = float(win["start"]), float(win["end"])
        spans = inner_span_pairs(win, w0, w1)
        payload = load_word_payload(WORDS / f"{sid}.json")
        words = words_from_payload(payload) if payload else []
        n_span = len(spans)
        for s0, s1 in spans:
            def union(cuts: list[tuple[float, float]]) -> tuple[float | None, float | None]:
                ov = [
                    (a, b)
                    for a, b in merge_spans(cuts)
                    if overlap_seconds(a, b, s0, s1) > 0
                ]
                if not ov:
                    return None, None
                return min(a for a, _ in ov), max(b for _, b in ov)

            b0, b1 = union(baseline.get(sid) or [])
            q0, q1 = union(v1_cuts.get(sid) or [])
            v0, v1 = union(v2_cuts.get(sid) or [])
            over_v2, under_v2 = window_over_under(w0, w1, [(s0, s1)], v2_cuts.get(sid) or [])
            # window_over_under attributes overcut in the whole candidate window;
            # recompute span-local.
            clipped = merge_spans(
                [
                    (max(c0, s0), min(c1, s1))
                    for c0, c1 in (v2_cuts.get(sid) or [])
                    if min(c1, s1) > max(c0, s0)
                ]
            )
            under = (s1 - s0) - covered_seconds(s0, s1, clipped)
            # overcut vs this span: cut inside the candidate window that is
            # outside this span but we want span-local over: cut beyond span
            # edges that still overlap the span's union cut.
            over = 0.0
            if v0 is not None and v1 is not None:
                over = max(0.0, s0 - v0) + max(0.0, v1 - s1)
                # only count overcut that is inside the gold candidate window
                over = max(0.0, min(s0, w1) - max(v0, w0)) if v0 < s0 else 0.0
                over += max(0.0, min(v1, w1) - max(s1, w0)) if v1 > s1 else 0.0

            qv2 = overlapping_quotes(v2_q.get(sid) or [], s0, s1)
            qv1 = overlapping_quotes(v1_q.get(sid) or [], s0, s1)
            klass = classify(
                s0=s0,
                s1=s1,
                c0=v0,
                c1=v1,
                under=under,
                over=over,
                n_v1=len(qv1),
                n_v2=len(qv2),
                n_span_in_window=n_span,
                excerpt=str(win.get("excerpt") or ""),
            )
            start_err = None if v0 is None else abs(v0 - s0)
            end_err = None if v1 is None else abs(v1 - s1)
            row = {
                "show_id": sid,
                "label": [round(s0, 2), round(s1, 2)],
                "window": [w0, w1],
                "class": klass,
                "v2_under": round(under, 2),
                "v2_over": round(over, 2),
                "v2_span": None if v0 is None else [round(v0, 2), round(v1, 2)],
                "v1_span": None if q0 is None else [round(q0, 2), round(q1, 2)],
                "base_span": None if b0 is None else [round(b0, 2), round(b1, 2)],
                "v2_start_err": None if start_err is None else round(start_err, 2),
                "v2_end_err": None if end_err is None else round(end_err, 2),
                "v1_quotes": [
                    {
                        "q": f"{x['start_quote'][:50]} … {x['end_quote'][:40]}",
                        "t": [round(float(x["start"]), 2), round(float(x["end"]), 2)],
                    }
                    for x in qv1
                ],
                "v2_quotes": [
                    {
                        "q": f"{x['start_quote'][:50]} … {x['end_quote'][:40]}",
                        "t": [round(float(x["start"]), 2), round(float(x["end"]), 2)],
                    }
                    for x in qv2
                ],
                "words_at_label_start": words_around(words, s0),
                "words_at_label_end": words_around(words, s1),
                "words_at_v2_start": words_around(words, v0) if v0 is not None else "",
                "words_at_v2_end": words_around(words, v1) if v1 is not None else "",
                "excerpt": (win.get("excerpt") or "")[:180],
            }
            rows.append(row)
            class_under[klass] = class_under.get(klass, 0.0) + under
            class_over[klass] = class_over.get(klass, 0.0) + over
            class_n[klass] = class_n.get(klass, 0) + 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_ad.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = {
        "n_spans": len(rows),
        "v2_quotes_total": sum(len(v) for v in v2_q.values()),
        "v1_quotes_total": sum(len(v) for v in v1_q.values()),
        "class_n": class_n,
        "class_undercut_s": {k: round(v, 2) for k, v in class_under.items()},
        "class_overcut_s": {k: round(v, 2) for k, v in class_over.items()},
        "undercut_sum": round(sum(class_under.values()), 2),
        "overcut_sum": round(sum(class_over.values()), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"v1 quotes={summary['v1_quotes_total']} v2 quotes={summary['v2_quotes_total']} spans={len(rows)}",
        f"attributed under={summary['undercut_sum']}s over={summary['overcut_sum']}s",
        "",
        "| class | n | undercut s | overcut s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in sorted(class_n, key=lambda x: -(class_under.get(x, 0) + class_over.get(x, 0))):
        lines.append(
            f"| {k} | {class_n[k]} | {class_under.get(k, 0):.1f} | {class_over.get(k, 0):.1f} |"
        )
    lines.append("")
    for row in sorted(rows, key=lambda r: -(r["v2_under"] + r["v2_over"])):
        if row["v2_under"] + row["v2_over"] < 1.0:
            continue
        lines.append(
            f"- {row['show_id']} {row['label']} class={row['class']} "
            f"under={row['v2_under']} over={row['v2_over']} "
            f"v2={row['v2_span']} v1={row['v1_span']} base={row['base_span']}"
        )
        if row["v2_quotes"]:
            lines.append(f"  v2 quote: {row['v2_quotes'][0]['q']}")
        lines.append(f"  label-start words: {row['words_at_label_start']}")
        lines.append(f"  label-end words: {row['words_at_label_end']}")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
