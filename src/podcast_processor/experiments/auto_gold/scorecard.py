"""Score production cut windows against auto-gold is_ad spans.

Pure overlap math. No LLM. Opening/preroll = any gold window that overlaps
[0, preroll_end] (default 90s, including t=0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

PREROLL_END_SECONDS = 90.0
WINDOW_CAUGHT_FRACTION = 0.5
CLAIM_LOCKED = (
    "`general_podcast_ads` remains **LOCKED** "
    "(silver Gemini judge votes only; no human spot-check)."
)


def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted((float(a), float(b)) for a, b in spans if float(b) > float(a))
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def covered_seconds(
    start: float, end: float, spans: list[tuple[float, float]]
) -> float:
    # sum([]) is int 0; beartype requires float.
    return float(
        sum(
            overlap_seconds(start, end, s0, s1)
            for s0, s1 in merge_spans(spans)
        )
    )


def is_opening(start: float, end: float, preroll_end: float = PREROLL_END_SECONDS) -> bool:
    return overlap_seconds(start, end, 0.0, preroll_end) > 0.0


@dataclass
class WindowScore:
    show_id: str
    start: float
    end: float
    is_ad: bool
    sources: list[str]
    covered: float
    opening: bool
    span_seconds: float = 0.0
    span_covered: float = 0.0
    opening_seconds: float = 0.0
    opening_covered: float = 0.0
    opening_span_seconds: float = 0.0
    opening_span_covered: float = 0.0
    overcut_s: float = 0.0
    undercut_s: float = 0.0
    start_errors: list[float] = field(default_factory=list)
    end_errors: list[float] = field(default_factory=list)

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def coverage_frac(self) -> float:
        dur = self.duration()
        return 0.0 if dur <= 0 else min(1.0, self.covered / dur)

    def span_coverage_frac(self) -> float:
        dur = self.span_seconds
        return 0.0 if dur <= 0 else min(1.0, self.span_covered / dur)

    def caught(self, min_frac: float = WINDOW_CAUGHT_FRACTION) -> bool:
        return self.is_ad and self.coverage_frac() >= min_frac

    def span_caught(self, min_frac: float = WINDOW_CAUGHT_FRACTION) -> bool:
        return self.is_ad and self.span_coverage_frac() >= min_frac


@dataclass
class ScoreTotals:
    n_is_ad_windows: int = 0
    n_is_ad_caught: int = 0
    n_is_ad_missed: int = 0
    is_ad_seconds: float = 0.0
    is_ad_seconds_caught: float = 0.0
    is_ad_seconds_missed: float = 0.0
    n_opening_windows: int = 0
    n_opening_caught: int = 0
    n_opening_missed: int = 0
    opening_seconds: float = 0.0
    opening_seconds_caught: float = 0.0
    opening_seconds_missed: float = 0.0
    n_not_ad_windows: int = 0
    not_ad_seconds: float = 0.0
    not_ad_seconds_cut: float = 0.0
    n_t0_windows: int = 0
    n_t0_caught: int = 0
    n_t0_missed: int = 0
    is_ad_span_seconds: float = 0.0
    is_ad_span_seconds_caught: float = 0.0
    opening_span_seconds: float = 0.0
    opening_span_seconds_caught: float = 0.0
    n_opening_span_caught: int = 0
    n_t0_span_caught: int = 0
    overcut_seconds: float = 0.0
    undercut_seconds: float = 0.0
    n_boundary_spans: int = 0
    start_error_median: float = 0.0
    start_error_p90: float = 0.0
    end_error_median: float = 0.0
    end_error_p90: float = 0.0

    def window_recall(self) -> float:
        return _ratio(float(self.n_is_ad_caught), float(self.n_is_ad_windows))

    def seconds_recall(self) -> float:
        return _ratio(self.is_ad_seconds_caught, self.is_ad_seconds)

    def opening_window_recall(self) -> float:
        return _ratio(float(self.n_opening_caught), float(self.n_opening_windows))

    def opening_seconds_recall(self) -> float:
        return _ratio(self.opening_seconds_caught, self.opening_seconds)

    def t0_window_recall(self) -> float:
        return _ratio(float(self.n_t0_caught), float(self.n_t0_windows))

    def span_seconds_recall(self) -> float:
        return _ratio(self.is_ad_span_seconds_caught, self.is_ad_span_seconds)

    def opening_span_seconds_recall(self) -> float:
        return _ratio(self.opening_span_seconds_caught, self.opening_span_seconds)

    def opening_span_window_recall(self) -> float:
        return _ratio(float(self.n_opening_span_caught), float(self.n_opening_windows))

    def t0_span_window_recall(self) -> float:
        return _ratio(float(self.n_t0_span_caught), float(self.n_t0_windows))

    def missed_pct(self) -> float:
        return 100.0 * (1.0 - self.seconds_recall())

    def opening_missed_pct(self) -> float:
        return 100.0 * (1.0 - self.opening_seconds_recall())


@dataclass
class ScorecardResult:
    totals: ScoreTotals
    windows: list[WindowScore] = field(default_factory=list)
    per_show: dict[str, ScoreTotals] = field(default_factory=dict)
    spend_usd: float = 0.0
    spend_cap_usd: float = 0.50
    n_shows_scored: int = 0
    n_shows_catalog: int = 12
    notes: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    start_errors: list[float] = field(default_factory=list)
    end_errors: list[float] = field(default_factory=list)

    def boundary_error_total(self) -> float:
        return float(self.totals.overcut_seconds + self.totals.undercut_seconds)


def inner_span_pairs(row: dict[str, Any], start: float, end: float) -> list[tuple[float, float]]:
    raw_spans = list(row.get("ad_spans") or [])
    if not raw_spans:
        return merge_spans([(start, end)])
    return merge_spans([(float(sp["start"]), float(sp["end"])) for sp in raw_spans])


def window_over_under(
    window_start: float,
    window_end: float,
    span_pairs: list[tuple[float, float]],
    cuts: list[tuple[float, float]],
) -> tuple[float, float]:
    clipped = merge_spans(
        [
            (max(c0, window_start), min(c1, window_end))
            for c0, c1 in cuts
            if min(c1, window_end) > max(c0, window_start)
        ]
    )
    overcut = 0.0
    for a, b in clipped:
        overcut += max(0.0, (b - a) - covered_seconds(a, b, span_pairs))
    undercut = 0.0
    for s0, s1 in span_pairs:
        undercut += max(0.0, (s1 - s0) - covered_seconds(s0, s1, clipped))
    return float(overcut), float(undercut)


def span_boundary_errors(
    window_start: float,
    window_end: float,
    span_pairs: list[tuple[float, float]],
    cuts: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    start_errors: list[float] = []
    end_errors: list[float] = []
    clipped = merge_spans(
        [
            (max(c0, window_start), min(c1, window_end))
            for c0, c1 in cuts
            if min(c1, window_end) > max(c0, window_start)
        ]
    )
    for s0, s1 in span_pairs:
        overlapping = [(a, b) for a, b in clipped if overlap_seconds(a, b, s0, s1) > 0]
        if not overlapping:
            continue
        union_start = min(a for a, _ in overlapping)
        union_end = max(b for _, b in overlapping)
        start_errors.append(abs(union_start - s0))
        end_errors.append(abs(union_end - s1))
    return start_errors, end_errors


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def score_windows(
    rows: list[dict[str, Any]],
    cuts_by_show: dict[str, list[tuple[float, float]]],
    *,
    preroll_end: float = PREROLL_END_SECONDS,
    min_frac: float = WINDOW_CAUGHT_FRACTION,
) -> ScorecardResult:
    windows: list[WindowScore] = []
    per_show: dict[str, ScoreTotals] = {}
    for row in rows:
        show_id = str(row["show_id"])
        start = float(row["start"])
        end = float(row["end"])
        is_ad = bool(row["is_ad"])
        cuts = merge_spans(cuts_by_show.get(show_id) or [])
        covered = covered_seconds(start, end, cuts)
        opening = is_opening(start, end, preroll_end)
        opening_lo = max(start, 0.0)
        opening_hi = min(end, preroll_end)
        opening_seconds = overlap_seconds(start, end, 0.0, preroll_end)
        opening_covered = (
            covered_seconds(opening_lo, opening_hi, cuts) if opening_seconds > 0 else 0.0
        )
        raw_spans = list(row.get("ad_spans") or [])
        span_pairs = inner_span_pairs(row, start, end)
        if raw_spans:
            span_seconds = float(sum(max(0.0, b - a) for a, b in span_pairs))
            span_covered = float(
                sum(covered_seconds(a, b, cuts) for a, b in span_pairs)
            )
            opening_span_seconds = float(
                sum(overlap_seconds(a, b, 0.0, preroll_end) for a, b in span_pairs)
            )
            opening_span_covered = float(
                sum(
                    covered_seconds(max(a, 0.0), min(b, preroll_end), cuts)
                    for a, b in span_pairs
                    if overlap_seconds(a, b, 0.0, preroll_end) > 0
                )
            )
        else:
            span_seconds = max(0.0, end - start)
            span_covered = covered
            opening_span_seconds = opening_seconds
            opening_span_covered = opening_covered
        overcut_s = 0.0
        undercut_s = 0.0
        start_errors: list[float] = []
        end_errors: list[float] = []
        if is_ad:
            overcut_s, undercut_s = window_over_under(start, end, span_pairs, cuts)
            start_errors, end_errors = span_boundary_errors(
                start, end, span_pairs, cuts
            )
        item = WindowScore(
            show_id=show_id,
            start=start,
            end=end,
            is_ad=is_ad,
            sources=list(row.get("sources") or []),
            covered=covered,
            opening=opening,
            span_seconds=span_seconds,
            span_covered=span_covered,
            opening_seconds=opening_seconds,
            opening_covered=opening_covered,
            opening_span_seconds=opening_span_seconds,
            opening_span_covered=opening_span_covered,
            overcut_s=overcut_s,
            undercut_s=undercut_s,
            start_errors=start_errors,
            end_errors=end_errors,
        )
        windows.append(item)
        bucket = per_show.setdefault(show_id, ScoreTotals())
        _accumulate(bucket, item, min_frac=min_frac)
    totals = ScoreTotals()
    all_start: list[float] = []
    all_end: list[float] = []
    for item in windows:
        _accumulate(totals, item, min_frac=min_frac)
        all_start.extend(item.start_errors)
        all_end.extend(item.end_errors)
    totals.start_error_median = percentile(all_start, 0.5)
    totals.start_error_p90 = percentile(all_start, 0.9)
    totals.end_error_median = percentile(all_end, 0.5)
    totals.end_error_p90 = percentile(all_end, 0.9)
    totals.n_boundary_spans = len(all_start)
    return ScorecardResult(
        totals=totals,
        windows=windows,
        per_show=per_show,
        n_shows_scored=len(per_show),
        start_errors=all_start,
        end_errors=all_end,
    )


def _accumulate(
    totals: ScoreTotals, item: WindowScore, *, min_frac: float
) -> None:
    dur = item.duration()
    if item.is_ad:
        totals.n_is_ad_windows += 1
        totals.is_ad_seconds += dur
        totals.is_ad_seconds_caught += item.covered
        totals.is_ad_seconds_missed += max(0.0, dur - item.covered)
        totals.is_ad_span_seconds += item.span_seconds
        totals.is_ad_span_seconds_caught += item.span_covered
        totals.overcut_seconds += item.overcut_s
        totals.undercut_seconds += item.undercut_s
        if item.caught(min_frac):
            totals.n_is_ad_caught += 1
        else:
            totals.n_is_ad_missed += 1
        if item.opening:
            totals.n_opening_windows += 1
            totals.opening_seconds += item.opening_seconds
            totals.opening_seconds_caught += item.opening_covered
            totals.opening_seconds_missed += max(
                0.0, item.opening_seconds - item.opening_covered
            )
            totals.opening_span_seconds += item.opening_span_seconds
            totals.opening_span_seconds_caught += item.opening_span_covered
            if item.caught(min_frac):
                totals.n_opening_caught += 1
            else:
                totals.n_opening_missed += 1
            if item.span_caught(min_frac):
                totals.n_opening_span_caught += 1
        if item.start <= 0.0:
            totals.n_t0_windows += 1
            if item.caught(min_frac):
                totals.n_t0_caught += 1
            else:
                totals.n_t0_missed += 1
            if item.span_caught(min_frac):
                totals.n_t0_span_caught += 1
        return
    totals.n_not_ad_windows += 1
    totals.not_ad_seconds += dur
    totals.not_ad_seconds_cut += item.covered


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def opening_fn_notes(result: ScorecardResult, min_frac: float = WINDOW_CAUGHT_FRACTION) -> list[str]:
    notes: list[str] = []
    for item in result.windows:
        if not item.is_ad or not item.opening:
            continue
        if not item.span_caught(min_frac):
            notes.append(
                f"- FN opening `{item.show_id}` {item.start:.0f}-{item.end:.0f}s "
                f"inner ad_span {item.span_covered:.1f}/{item.span_seconds:.1f}s "
                f"({pct(item.span_coverage_frac())}); "
                f"candidate window {item.covered:.1f}/{item.duration():.1f}s"
            )
        elif not item.caught(min_frac):
            notes.append(
                f"- Opening `{item.show_id}` inner ad_span caught "
                f"({pct(item.span_coverage_frac())}) but 90s candidate window only "
                f"{pct(item.coverage_frac())}"
            )
    return notes


def fp_notes(result: ScorecardResult, min_cut: float = 8.0) -> list[str]:
    notes: list[str] = []
    for item in result.windows:
        if item.is_ad:
            continue
        if item.covered >= min_cut:
            notes.append(
                f"- FP-ish cut into not_ad `{item.show_id}` "
                f"{item.start:.0f}-{item.end:.0f}s cut {item.covered:.1f}s"
            )
    return notes


def render_scorecard_md(result: ScorecardResult) -> str:
    t = result.totals
    lines = [
        "# Production remover vs auto-gold scorecard",
        "",
        "Production Podly `AdClassifier` (scout confirm **OFF**) scored against "
        "this-audio re-judge windows on the frozen auto-gold timestamps. "
        f"{result.n_shows_scored}/{result.n_shows_catalog} episodes. "
        "Overlap scoring only; no second judge.",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| is_ad audio caught (candidate-window seconds) | {pct(t.seconds_recall())} "
        f"({t.is_ad_seconds_caught:.1f}s / {t.is_ad_seconds:.1f}s) |",
        f"| is_ad audio missed (candidate-window seconds) | {t.missed_pct():.1f}% "
        f"({t.is_ad_seconds_missed:.1f}s) |",
        f"| is_ad audio caught (judge inner ad_span seconds) | "
        f"{pct(t.span_seconds_recall())} "
        f"({t.is_ad_span_seconds_caught:.1f}s / {t.is_ad_span_seconds:.1f}s) |",
        f"| is_ad windows caught (≥50% of candidate window) | {pct(t.window_recall())} "
        f"({t.n_is_ad_caught}/{t.n_is_ad_windows}) |",
        f"| is_ad windows missed | {t.n_is_ad_missed}/{t.n_is_ad_windows} |",
        f"| opening/preroll caught (seconds overlapping t=0–{PREROLL_END_SECONDS:.0f}s) | "
        f"{pct(t.opening_seconds_recall())} "
        f"({t.opening_seconds_caught:.1f}s / {t.opening_seconds:.1f}s) |",
        f"| opening/preroll missed (t=0–{PREROLL_END_SECONDS:.0f}s) | "
        f"{t.opening_missed_pct():.1f}% "
        f"({t.opening_seconds_missed:.1f}s) |",
        f"| opening/preroll caught (inner ad_span ∩ t=0–{PREROLL_END_SECONDS:.0f}s) | "
        f"{pct(t.opening_span_seconds_recall())} "
        f"({t.opening_span_seconds_caught:.1f}s / {t.opening_span_seconds:.1f}s) |",
        f"| opening windows caught (≥50% of candidate) | "
        f"{pct(t.opening_window_recall())} "
        f"({t.n_opening_caught}/{t.n_opening_windows}) |",
        f"| opening windows caught (≥50% of inner ad_span) | "
        f"{pct(t.opening_span_window_recall())} "
        f"({t.n_opening_span_caught}/{t.n_opening_windows}) |",
        f"| t=0 windows caught (≥50% of candidate) | {pct(t.t0_window_recall())} "
        f"({t.n_t0_caught}/{t.n_t0_windows}) |",
        f"| t=0 windows caught (≥50% of inner ad_span) | "
        f"{pct(t.t0_span_window_recall())} "
        f"({t.n_t0_span_caught}/{t.n_t0_windows}) |",
        f"| not_ad seconds also cut (FP-ish) | {t.not_ad_seconds_cut:.1f}s / "
        f"{t.not_ad_seconds:.1f}s |",
        f"| start-boundary error median / p90 | {t.start_error_median:.2f}s / "
        f"{t.start_error_p90:.2f}s ({t.n_boundary_spans} inner spans) |",
        f"| end-boundary error median / p90 | {t.end_error_median:.2f}s / "
        f"{t.end_error_p90:.2f}s |",
        f"| overcut seconds (cut ∩ window, outside inner ad_span) | "
        f"{t.overcut_seconds:.1f}s |",
        f"| undercut seconds (inner ad_span left uncut) | "
        f"{t.undercut_seconds:.1f}s |",
        f"| total boundary error (overcut + undercut) | "
        f"{t.overcut_seconds + t.undercut_seconds:.1f}s |",
        f"| shows scored | {result.n_shows_scored}/{result.n_shows_catalog} |",
        f"| Gemini spend | ${result.spend_usd:.4f} (cap ${result.spend_cap_usd:.2f}) |",
        f"| `enable_bow_scout_gemini_confirm` | `False` |",
        "",
        "## Claim",
        "",
        CLAIM_LOCKED,
        "",
    ]
    if result.outliers or result.notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(result.outliers)
        if result.outliers and result.notes:
            lines.append("")
        lines.extend(f"- {note}" if not note.startswith("-") else note for note in result.notes)
        lines.append("")
    return "\n".join(lines)
