"""Quote-anchored ad boundaries: match LLM quotes to word-level transcripts.

$0 matcher. Fail-closed: never drop an ad that the model reported.
Scout confirm stays off. Self-promo counts as an ad (policy 2026-10-05).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.72
MIN_QUOTE_TOKENS = 3
QUOTE_ANCHOR_SOURCE = "quote_anchor"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class SegmentSpan:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class QuoteAd:
    start_quote: str
    end_quote: str
    content_type: str
    confidence: float


@dataclass
class QuoteMatch:
    ad: QuoteAd
    start: float
    end: float
    method: Literal["word", "segment_fallback", "chunk_fallback"]
    score: float
    dropped: bool = False
    bound_start: float | None = None
    bound_end: float | None = None


def normalize_tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return _TOKEN_RE.findall(ascii_text.lower())


def token_edit_distance(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        curr = [i]
        for j, b in enumerate(right, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if a == b else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _window_score(quote: list[str], window: list[str]) -> float:
    if not quote:
        return 0.0
    dist = token_edit_distance(quote, window)
    denom = max(len(quote), len(window), 1)
    return 1.0 - (float(dist) / float(denom))


def match_quote_tokens(
    words: list[WordTiming],
    quote: str,
    *,
    search_from: int = 0,
    search_to: int | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[int, int, float] | None:
    """Return (start_idx, end_idx_exclusive, score) for the first-best quote hit."""
    needle = normalize_tokens(quote)
    if len(needle) < MIN_QUOTE_TOKENS:
        return None
    # Every alphanumeric piece in the word (Shopify.com → shopify, com;
    # S-I-N-C-H → s,i,n,c,h). Using only parts[0] dropped URL/spell-out
    # quotes after lexicon lock kept those as single tokens.
    tokens: list[tuple[int, str]] = []
    for idx, word in enumerate(words):
        for part in normalize_tokens(word.word):
            tokens.append((idx, part))
    lo = 0
    while lo < len(tokens) and tokens[lo][0] < search_from:
        lo += 1
    hi = len(tokens)
    if search_to is not None:
        while hi > lo and tokens[hi - 1][0] >= search_to:
            hi -= 1
    if hi - lo < MIN_QUOTE_TOKENS:
        return None

    n = len(needle)
    first = needle[0]
    # Align to the quote's first token, then take the first left-to-right
    # start that clears the threshold (best width there). Repeated phrases
    # resolve in order via search_from / match_quotes_in_order.
    for i in range(lo, hi):
        if tokens[i][1] != first:
            continue
        best_here: tuple[int, int, float] | None = None
        best_width = 0
        for width in range(max(MIN_QUOTE_TOKENS, n - 2), n + 3):
            j = i + width
            if j > hi:
                break
            window = [tok for _, tok in tokens[i:j]]
            score = _window_score(needle, window)
            if score < threshold:
                continue
            word_i = tokens[i][0]
            word_j = tokens[j - 1][0] + 1
            if (
                best_here is None
                or score > best_here[2] + 1e-9
                or (
                    abs(score - best_here[2]) <= 1e-9
                    and abs(width - n) < abs(best_width - n)
                )
            ):
                best_here = (word_i, word_j, score)
                best_width = width
        if best_here is not None:
            return best_here
    return None


def match_quotes_in_order(
    words: list[WordTiming],
    ads: list[QuoteAd],
    *,
    search_from: int = 0,
    search_to: int | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> list[tuple[QuoteAd, tuple[int, int, float] | None]]:
    cursor = search_from
    results: list[tuple[QuoteAd, tuple[int, int, float] | None]] = []
    for ad in ads:
        start_hit = match_quote_tokens(
            words,
            ad.start_quote,
            search_from=cursor,
            search_to=search_to,
            threshold=threshold,
        )
        if start_hit is None:
            results.append((ad, None))
            continue
        if normalize_tokens(ad.end_quote) == normalize_tokens(ad.start_quote):
            results.append((ad, start_hit))
            cursor = start_hit[1]
            continue
        end_hit = match_quote_tokens(
            words,
            ad.end_quote,
            search_from=start_hit[1],
            search_to=search_to,
            threshold=threshold,
        )
        if end_hit is None:
            results.append((ad, None))
            continue
        span_start = start_hit[0]
        span_end = max(end_hit[1], start_hit[1])
        score = min(start_hit[2], end_hit[2])
        results.append((ad, (span_start, span_end, score)))
        cursor = span_end
    return results


def _segment_index_for_quote(
    segments: list[SegmentSpan], quote: str, *, start_at: int = 0
) -> int | None:
    needle = normalize_tokens(quote)
    if len(needle) < MIN_QUOTE_TOKENS:
        return None
    best_i: int | None = None
    best_score = 0.0
    for i in range(start_at, len(segments)):
        hay = normalize_tokens(segments[i].text)
        if not hay:
            continue
        # Subsequence score: fraction of quote tokens found in order.
        k = 0
        for token in hay:
            if k < len(needle) and token == needle[k]:
                k += 1
        ordered = k / float(len(needle))
        bag = len(set(needle) & set(hay)) / float(len(set(needle)))
        score = max(ordered, 0.7 * bag)
        if score > best_score:
            best_score = score
            best_i = i
            if score >= 0.999:
                break
    if best_i is None or best_score < MATCH_THRESHOLD:
        return None
    return best_i


def fallback_segment_edge(
    ad: QuoteAd,
    segments: list[SegmentSpan],
    *,
    chunk_start: float,
    chunk_end: float,
) -> tuple[float, float, float] | None:
    in_chunk = [
        seg for seg in segments if seg.end > chunk_start and seg.start < chunk_end
    ]
    if not in_chunk:
        in_chunk = segments
    start_i = _segment_index_for_quote(in_chunk, ad.start_quote)
    if start_i is None:
        return None
    end_i = _segment_index_for_quote(
        in_chunk, ad.end_quote, start_at=start_i
    )
    if end_i is None:
        end_i = start_i
    start = in_chunk[start_i].start
    end = in_chunk[end_i].end
    if end <= start:
        return None
    return start, end, MATCH_THRESHOLD


def resolve_ads(
    ads: list[QuoteAd],
    words: list[WordTiming],
    segments: list[SegmentSpan],
    *,
    chunk_start: float,
    chunk_end: float,
    word_from: int = 0,
    word_to: int | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> list[QuoteMatch]:
    """Map each reported ad to a cut. Never drop: chunk span is last resort."""
    ordered = match_quotes_in_order(
        words,
        ads,
        search_from=word_from,
        search_to=word_to,
        threshold=threshold,
    )
    matches: list[QuoteMatch] = []
    for ad, hit in ordered:
        seg_bound = fallback_segment_edge(
            ad, segments, chunk_start=chunk_start, chunk_end=chunk_end
        )
        if hit is not None:
            i0, i1, score = hit
            start = words[i0].start
            end = words[i1 - 1].end
            if end > start:
                if seg_bound is not None:
                    bound_start = min(seg_bound[0], start)
                    bound_end = max(seg_bound[1], end)
                else:
                    bound_start, bound_end = start, end
                matches.append(
                    QuoteMatch(
                        ad=ad,
                        start=start,
                        end=end,
                        method="word",
                        score=score,
                        bound_start=bound_start,
                        bound_end=bound_end,
                    )
                )
                continue
            logger.warning("word match collapsed for %r-%r", ad.start_quote, ad.end_quote)
        if seg_bound is not None:
            start, end, score = seg_bound
            logger.info(
                "quote match miss; segment-edge fallback %s-%.1f-%.1f",
                ad.start_quote[:40],
                start,
                end,
            )
            matches.append(
                QuoteMatch(
                    ad=ad,
                    start=start,
                    end=end,
                    method="segment_fallback",
                    score=score,
                    bound_start=start,
                    bound_end=end,
                )
            )
            continue
        logger.warning(
            "quote unmatched; chunk fallback %.1f-%.1f for %r",
            chunk_start,
            chunk_end,
            ad.start_quote[:40],
        )
        matches.append(
            QuoteMatch(
                ad=ad,
                start=chunk_start,
                end=max(chunk_end, chunk_start + 0.5),
                method="chunk_fallback",
                score=0.0,
                bound_start=chunk_start,
                bound_end=max(chunk_end, chunk_start + 0.5),
            )
        )
    return matches


def match_rate(matches: list[QuoteMatch]) -> float:
    if not matches:
        return 1.0
    n_word = sum(1 for item in matches if item.method == "word")
    return float(n_word) / float(len(matches))


def word_index_range(
    words: list[WordTiming], start: float, end: float
) -> tuple[int, int]:
    lo = 0
    while lo < len(words) and words[lo].end <= start:
        lo += 1
    hi = lo
    while hi < len(words) and words[hi].start < end:
        hi += 1
    return lo, hi


def cuts_from_matches(matches: list[QuoteMatch]) -> list[tuple[float, float]]:
    return [(item.start, item.end) for item in matches if item.end > item.start]


def parse_quote_ads(raw: str) -> list[QuoteAd]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return []
    blob = text[start : end + 1]
    payload: object
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        try:
            payload = json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return []
    rows = payload.get("ads") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    ads: list[QuoteAd] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start_q = str(row.get("start_quote") or "").strip()
        end_q = str(row.get("end_quote") or start_q).strip()
        if not start_q:
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        ads.append(
            QuoteAd(
                start_quote=start_q,
                end_quote=end_q or start_q,
                content_type=str(row.get("content_type") or "promotional_external"),
                confidence=confidence,
            )
        )
    return ads


def keep_ad(ad: QuoteAd, min_confidence: float) -> bool:
    conf = ad.confidence
    if ad.content_type == "technical_discussion":
        conf = max(0.0, conf - 0.25)
    elif ad.content_type == "transition":
        conf = max(0.0, conf - 0.1)
    return conf >= min_confidence


def expand_outward_to_silence(
    start: float,
    end: float,
    gaps: list[tuple[float, float]],
    *,
    bound_start: float,
    bound_end: float,
    pad_seconds: float = 0.0,
) -> tuple[float, float]:
    """Grow a quote core outward to the nearest silence, never past the bound.

    Start snaps to the nearest silence_end in [bound_start, core_start].
    End snaps to the nearest silence_start in [core_end, bound_end].
    If no silence lies in that outward band, the core edge is kept (then
    clipped to the bound). pad_seconds extends the core before snapping.
    """
    lo = min(float(bound_start), float(bound_end))
    hi = max(float(bound_start), float(bound_end))
    core_start = min(max(float(start) - float(pad_seconds), lo), hi)
    core_end = min(max(float(end) + float(pad_seconds), lo), hi)
    if core_end <= core_start:
        return lo, hi

    start_hits = [
        float(gap_end)
        for gap_start, gap_end in gaps
        if lo - 1e-9 <= float(gap_end) <= core_start + 1e-9
    ]
    end_hits = [
        float(gap_start)
        for gap_start, gap_end in gaps
        if core_end - 1e-9 <= float(gap_start) <= hi + 1e-9
    ]
    new_start = min(start_hits, key=lambda t: (abs(t - core_start), t)) if start_hits else core_start
    new_end = min(end_hits, key=lambda t: (abs(t - core_end), -t)) if end_hits else core_end
    new_start = min(max(new_start, lo), core_start)
    new_end = max(min(new_end, hi), core_end)
    if new_end <= new_start:
        return core_start, core_end
    return new_start, new_end


def clip_cut_to_baseline(
    start: float,
    end: float,
    baseline: list[tuple[float, float]],
) -> tuple[float, float]:
    """Do not extend an edge past an overlapping baseline cut.

    If the quote does not overlap any baseline cut, leave it unchanged so a
    newly found ad is not dropped.
    """
    overlapping = [
        (b0, b1)
        for b0, b1 in baseline
        if min(end, b1) > max(start, b0)
    ]
    if not overlapping:
        return start, end
    bound0 = min(b0 for b0, _ in overlapping)
    bound1 = max(b1 for _, b1 in overlapping)
    return max(start, bound0), min(end, bound1)


def hybrid_cuts_from_matches(
    matches: list[QuoteMatch],
    gaps: list[tuple[float, float]],
    *,
    baseline: list[tuple[float, float]] | None = None,
    pad_seconds: float = 0.0,
) -> list[tuple[float, float]]:
    cuts: list[tuple[float, float]] = []
    for item in matches:
        if item.end <= item.start:
            continue
        bound_start = item.bound_start if item.bound_start is not None else item.start
        bound_end = item.bound_end if item.bound_end is not None else item.end
        start, end = expand_outward_to_silence(
            item.start,
            item.end,
            gaps,
            bound_start=bound_start,
            bound_end=bound_end,
            pad_seconds=pad_seconds,
        )
        if baseline:
            start, end = clip_cut_to_baseline(start, end, baseline)
        if end > start:
            cuts.append((start, end))
    return cuts
