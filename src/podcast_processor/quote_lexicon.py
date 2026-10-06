"""$0 lexical refine of quote-matched cuts using word timings and pauses.

English ad-break lexicon plus pause/hole features. Show IDs are only in
TUNE_NOTES (which rule was motivated by which failure), not in the matcher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from podcast_processor.quote_anchor import QuoteMatch, WordTiming, word_index_range

# Motivated by npr_wait_wait 717 (class c): bumper then "Support comes from".
LEADIN_RE = re.compile(
    r"\b("
    r"support comes from|support for this|this episode is sponsored|"
    r"this message comes from|brought to you by|sponsored by|"
    r"today.s episode is|a word from our sponsor|paid partner|"
    r"this week on|listen now on"
    r")\b",
    re.I,
)

# Motivated by planet_money 1199 and npr_wait_wait 1883 (class d / c).
BRIDGE_RE = re.compile(
    r"\b("
    r"after the break|we.ll be (right )?back|when we come back|"
    r"stay tuned"
    r")\b",
    re.I,
)

SISTER_RE = re.compile(
    r"\b(this week on|listen now on|wherever you get your podcasts)\b",
    re.I,
)

# Motivated by npr_up_first 868 (class f): credits then a long hole.
CLOSER_RE = re.compile(
    r"\b(join us again|see you tomorrow|thanks for listening|"
    r"technical director|executive producer)\b",
    re.I,
)

SHOW_IDENT_RE = re.compile(
    r"^(okay[,.]? let.s do the show|let.s do this|"
    r"this is \w+(?:\s+\w+){0,2}(?: from \w+)?)\.?$",
    re.I,
)

CTA_RE = re.compile(
    r"(\.com|\.org|slash |promo code|\bcode \w+|donate\.|"
    r"back to the show|and we.re back|thank you\.?$)",
    re.I,
)

SENTENCE_END_RE = re.compile(r"[.!?]$")

MIN_INTRA_PAUSE_S = 0.8
MIN_LONG_HOLE_S = 12.0
MAX_HOLE_PULL_S = 25.0
MAX_FORWARD_S = 20.0
MIN_END_HOLE_S = 3.0
SHORT_END_HOLE_S = 5.0
MAX_END_HOLE_S = 15.0
PREROLL_SNAP_S = 3.0
MAX_SENTENCE_EXTEND_S = 4.0
NO_FORWARD_TRIM_BEFORE_S = 8.0
UNION_START_SLACK_S = 25.0


@dataclass(frozen=True)
class LexicalTuneNote:
    rule: str
    motivated_by: str
    class_hint: str


TUNE_NOTES = (
    LexicalTuneNote(
        "forward_trim_bridge_or_leadin",
        "npr_wait_wait 717/1883 and planet_money 1199",
        "c_overcut_show / d_split_merge",
    ),
    LexicalTuneNote(
        "union_v1_later_start",
        "npr_wait_wait 717/1883 and planet_money 1199 (unused in scored lexical_v2; same starts as forward-trim)",
        "c_overcut_show / d_split_merge",
    ),
    LexicalTuneNote(
        "pull_start_across_long_prequote_hole",
        "npr_up_first 868 (21s post-roll hole after closer)",
        "f_label_off",
    ),
    LexicalTuneNote(
        "pull_start_across_dai_bed_8_to_12s",
        "holdout npr_indicator 422 (9s wordless DAI bed before Wait Wait; now-seen)",
        "a_missed_leadin",
    ),
    LexicalTuneNote(
        "extend_end_to_silence_in_word_gap",
        "darknet_diaries 53–61 long bed (silence start); planet_money 21–24 short CTA music (silence end)",
        "b_missed_outro",
    ),
    LexicalTuneNote(
        "snap_preroll_start_to_zero",
        "darknet_diaries opening 0–2s before first word",
        "a_missed_leadin",
    ),
)


def _join(words: list[WordTiming], lo: int, hi: int) -> str:
    return " ".join(w.word for w in words[lo:hi])


def _sentence_start(words: list[WordTiming], idx: int) -> int:
    i = idx
    while i > 0 and not SENTENCE_END_RE.search(words[i - 1].word):
        i -= 1
    return i


def _sentence_end(words: list[WordTiming], idx: int) -> int:
    i = idx
    while i < len(words) and not SENTENCE_END_RE.search(words[i].word):
        i += 1
    return min(i + 1, len(words))


def _pauses_in(
    words: list[WordTiming], lo: int, hi: int, *, min_gap: float
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for i in range(lo + 1, hi):
        gap = words[i].start - words[i - 1].end
        if gap >= min_gap:
            out.append((i, gap))
    return out


def _find_span(words: list[WordTiming], lo: int, hi: int, pattern: re.Pattern[str]) -> int | None:
    for i in range(lo, hi):
        window = _join(words, i, min(hi, i + 8))
        if pattern.search(window):
            return i
    return None


def refine_quote_span(
    words: list[WordTiming],
    start: float,
    end: float,
    *,
    gaps: list[tuple[float, float]] | None = None,
    max_forward_s: float = MAX_FORWARD_S,
    min_hole_s: float = MIN_LONG_HOLE_S,
    max_hole_s: float = MAX_HOLE_PULL_S,
    pull_dai_bed: bool = True,
) -> tuple[float, float]:
    """Trim show lead-in; pull start across a pre-quote hole; eat wordless CTA tails."""
    if not words or end <= start:
        return start, end
    lo, hi = word_index_range(words, start, end)
    if hi <= lo:
        return start, end

    new_lo = lo
    limit = lo
    while limit < hi and words[limit].start < start + max_forward_s:
        limit += 1
    prefix = _join(words, lo, min(hi, lo + 10))
    is_bridge = bool(BRIDGE_RE.search(prefix))
    is_lead = bool(LEADIN_RE.search(prefix) or SISTER_RE.search(prefix))
    pauses = _pauses_in(words, lo, limit, min_gap=MIN_INTRA_PAUSE_S)

    # Do not forward-trim openings: "wherever you get your podcasts" is an
    # outro cue, and Serial-style cold opens are already the ad.
    if start >= NO_FORWARD_TRIM_BEFORE_S and pauses and (is_bridge or not is_lead):
        chosen: int | None = None
        for idx, _gap in pauses:
            after = _join(words, idx, min(hi, idx + 8))
            if LEADIN_RE.search(after):
                chosen = idx
                break
        if chosen is None:
            later = _find_span(words, lo + 1, limit, LEADIN_RE)
            if later is not None:
                chosen = later
        if chosen is None and is_bridge:
            chosen = max(pauses, key=lambda item: item[1])[0]
        if chosen is not None:
            new_lo = chosen

    start_t = words[new_lo].start
    # Pull back across a hole *before the original quote*, not a pause we just skipped.
    # Never eat a preceding show ident. Long holes (≥12s) are post-roll beds.
    # 8–12s wordless holes are DAI beds (NPR Indicator Wait Wait insert) unless
    # the previous sentence is a closer/credits line (Planet Money 1921).
    if lo > 0 and new_lo == lo:
        prev = words[lo - 1]
        hole = words[lo].start - prev.end
        prev_lo = _sentence_start(words, lo - 1)
        prev_text = _join(words, prev_lo, lo).strip()
        ident = bool(SHOW_IDENT_RE.search(prev_text))
        closer = bool(CLOSER_RE.search(prev_text))
        long_ok = min_hole_s <= hole <= max_hole_s
        dai_bed = (
            pull_dai_bed
            and 8.0 <= hole < min_hole_s
            and hole <= max_hole_s
            and not closer
        )
        if not ident and (long_ok or dai_bed):
            start_t = prev.end

    if start < PREROLL_SNAP_S and new_lo == lo:
        start_t = 0.0

    new_hi = hi
    tail = _join(words, max(new_lo, hi - 12), hi)
    cta_tail = bool(CTA_RE.search(tail))
    if cta_tail:
        sent_hi = _sentence_end(words, hi - 1)
        if words[sent_hi - 1].end <= end + MAX_SENTENCE_EXTEND_S:
            new_hi = sent_hi

    end_t = words[new_hi - 1].end if new_hi > new_lo else end
    if cta_tail and new_hi < len(words):
        next_gap = words[new_hi].start - words[new_hi - 1].end
        if MIN_END_HOLE_S <= next_gap <= MAX_END_HOLE_S:
            t0 = words[new_hi - 1].end
            t1 = words[new_hi].start
            silence_hits = [
                (float(g0), float(g1))
                for g0, g1 in (gaps or [])
                if t0 - 0.2 <= float(g0) <= t1 + 0.05 and (g1 - g0) >= 0.4
            ]
            if silence_hits:
                g0, g1 = min(silence_hits, key=lambda item: item[0])
                # Short post-CTA music: take the far edge. Long beds: stop
                # at silence start so we do not eat the show resume.
                end_t = g1 if next_gap < SHORT_END_HOLE_S else g0

    if end_t <= start_t:
        return start, end
    return start_t, end_t


def union_v1_v2_cuts(
    v2: list[tuple[float, float]],
    v1: list[tuple[float, float]],
    *,
    slack_s: float = UNION_START_SLACK_S,
) -> list[tuple[float, float]]:
    """$0 merge of cached prompt-v1 and prompt-v2 quote spans.

    Overlapping pair: take the later start (v1 often lands on the lead-in)
    within slack_s of the v2 start, and the later end (v2 often covers the
    full stacked break). Keep unmatched v1 cuts so ads are not dropped.
    Motivated by npr_wait_wait 717/1883 and planet_money 1199 vs v2 early starts.
    """
    out: list[tuple[float, float]] = []
    for start, end in v2:
        overlapping = [
            (a, b) for a, b in v1 if min(end, b) > max(start, a)
        ]
        new_start, new_end = start, end
        if overlapping:
            v1_start = min(a for a, _ in overlapping)
            if start < v1_start <= start + slack_s:
                new_start = v1_start
            v1_end = max(b for _, b in overlapping)
            if v1_end > new_end:
                new_end = v1_end
        if new_end > new_start:
            out.append((new_start, new_end))
    for start, end in v1:
        if end <= start:
            continue
        if not any(min(end, b) > max(start, a) for a, b in v2):
            out.append((start, end))
    return out



def lexical_cuts_from_matches(
    matches: list[QuoteMatch],
    words: list[WordTiming],
    gaps: list[tuple[float, float]] | None = None,
    *,
    asymmetric_end: bool = False,
    pull_dai_bed: bool = True,
) -> list[tuple[float, float]]:
    cuts: list[tuple[float, float]] = []
    for item in matches:
        if item.end <= item.start:
            continue
        start, end = refine_quote_span(
            words, item.start, item.end, gaps=gaps, pull_dai_bed=pull_dai_bed
        )
        if asymmetric_end and item.bound_end is not None and item.bound_end > end:
            end = float(item.bound_end)
        if end > start:
            cuts.append((start, end))
    return cuts
