"""Quote-anchored ad boundaries: match LLM quotes to word-level transcripts.

$0 matcher. Fail-closed: never drop an ad that the model reported.
Scout confirm stays off. Self-promo counts as an ad (policy 2026-10-05).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.72
MIN_QUOTE_TOKENS = 3
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
    tokens: list[tuple[int, str]] = []
    for idx, word in enumerate(words):
        parts = normalize_tokens(word.word)
        if parts:
            tokens.append((idx, parts[0]))
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
        if hit is not None:
            i0, i1, score = hit
            start = words[i0].start
            end = words[i1 - 1].end
            if end > start:
                matches.append(
                    QuoteMatch(
                        ad=ad, start=start, end=end, method="word", score=score
                    )
                )
                continue
            logger.warning("word match collapsed for %r-%r", ad.start_quote, ad.end_quote)
        fallback = fallback_segment_edge(
            ad, segments, chunk_start=chunk_start, chunk_end=chunk_end
        )
        if fallback is not None:
            start, end, score = fallback
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
