from __future__ import annotations

from podcast_processor.quote_anchor import (
    QuoteAd,
    SegmentSpan,
    WordTiming,
    fallback_segment_edge,
    match_quote_tokens,
    match_quotes_in_order,
    match_rate,
    normalize_tokens,
    resolve_ads,
    token_edit_distance,
)


def _words(*pairs: tuple[str, float, float]) -> list[WordTiming]:
    return [WordTiming(word=w, start=s, end=e) for w, s, e in pairs]


def test_normalize_and_edit_distance() -> None:
    assert normalize_tokens("Use Code SAVE-20!") == ["use", "code", "save", "20"]
    assert token_edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert token_edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_exact_quote_match() -> None:
    words = _words(
        ("hello", 0.0, 0.3),
        ("this", 0.3, 0.5),
        ("episode", 0.5, 0.9),
        ("is", 0.9, 1.0),
        ("sponsored", 1.0, 1.5),
        ("by", 1.5, 1.7),
        ("netsuite", 1.7, 2.2),
        ("thanks", 2.2, 2.5),
    )
    hit = match_quote_tokens(words, "this episode is sponsored by netsuite")
    assert hit is not None
    start, end, score = hit
    assert words[start].word == "this"
    assert words[end - 1].word == "netsuite"
    assert score == 1.0


def test_paraphrase_ish_quote_match() -> None:
    words = _words(
        ("this", 10.0, 10.2),
        ("episode", 10.2, 10.5),
        ("is", 10.5, 10.6),
        ("sponsored", 10.6, 11.1),
        ("by", 11.1, 11.2),
        ("acme", 11.2, 11.5),
        ("dot", 11.5, 11.6),
        ("com", 11.6, 11.8),
    )
    hit = match_quote_tokens(words, "this episode is sponsored by acme")
    assert hit is not None
    assert hit[2] >= 0.72
    assert words[hit[0]].word == "this"
    assert words[hit[1] - 1].word in {"acme", "com"}


def test_repeated_phrase_resolves_in_order() -> None:
    words = _words(
        ("first", 0.0, 0.4),
        ("use", 1.0, 1.2),
        ("code", 1.2, 1.4),
        ("save20", 1.4, 1.9),
        ("now", 2.0, 2.2),
        ("show", 30.0, 30.4),
        ("use", 50.0, 50.2),
        ("code", 50.2, 50.4),
        ("save20", 50.4, 50.9),
        ("again", 51.0, 51.3),
    )
    ads = [
        QuoteAd(
            start_quote="use code save20",
            end_quote="use code save20",
            content_type="promotional_external",
            confidence=0.9,
        ),
        QuoteAd(
            start_quote="use code save20",
            end_quote="use code save20",
            content_type="promotional_external",
            confidence=0.9,
        ),
    ]
    ordered = match_quotes_in_order(words, ads)
    assert ordered[0][1] is not None
    assert ordered[1][1] is not None
    assert words[ordered[0][1][0]].start == 1.0
    assert words[ordered[1][1][0]].start == 50.0


def test_no_match_returns_none() -> None:
    words = _words(
        ("good", 0.0, 0.3),
        ("morning", 0.3, 0.7),
        ("from", 0.7, 0.9),
        ("npr", 0.9, 1.2),
        ("news", 1.2, 1.5),
    )
    assert match_quote_tokens(words, "xyzzy plugh fnord") is None


def test_no_match_falls_back_to_segment_edge_and_never_drops() -> None:
    words = _words(
        ("hello", 0.0, 0.4),
        ("friends", 0.4, 0.8),
        ("welcome", 0.8, 1.2),
        ("back", 1.2, 1.5),
    )
    segments = [
        SegmentSpan(start=0.0, end=0.8, text="Hello friends"),
        SegmentSpan(start=0.8, end=1.5, text="welcome back to the show"),
        SegmentSpan(start=20.0, end=25.0, text="this episode is sponsored by Acme"),
        SegmentSpan(start=25.0, end=28.0, text="visit acme.com/podcast and save"),
    ]
    ad = QuoteAd(
        start_quote="this episode is sponsored by acme",
        end_quote="visit acme.com/podcast and save",
        content_type="promotional_external",
        confidence=0.95,
    )
    assert match_quote_tokens(words, ad.start_quote) is None
    fallback = fallback_segment_edge(ad, segments, chunk_start=18.0, chunk_end=30.0)
    assert fallback is not None
    assert fallback[0] == 20.0
    assert fallback[1] == 28.0

    matches = resolve_ads(
        [ad],
        words,
        segments,
        chunk_start=18.0,
        chunk_end=30.0,
    )
    assert len(matches) == 1
    assert matches[0].dropped is False
    assert matches[0].method == "segment_fallback"
    assert matches[0].start == 20.0
    assert matches[0].end == 28.0


def test_chunk_fallback_when_nothing_matches() -> None:
    words = _words(("hello", 0.0, 0.4), ("world", 0.4, 0.8))
    segments = [SegmentSpan(start=0.0, end=0.8, text="hello world")]
    ad = QuoteAd(
        start_quote="totally missing sponsor sentence here",
        end_quote="another missing closer sentence here",
        content_type="promotional_external",
        confidence=0.9,
    )
    matches = resolve_ads([ad], words, segments, chunk_start=5.0, chunk_end=12.0)
    assert len(matches) == 1
    assert matches[0].dropped is False
    assert matches[0].method == "chunk_fallback"
    assert matches[0].start == 5.0
    assert matches[0].end == 12.0
    assert match_rate(matches) == 0.0


def test_word_match_rate_is_one_when_all_hit() -> None:
    words = _words(
        ("this", 0.0, 0.2),
        ("episode", 0.2, 0.5),
        ("is", 0.5, 0.6),
        ("sponsored", 0.6, 1.0),
        ("by", 1.0, 1.1),
        ("acme", 1.1, 1.4),
        ("go", 1.4, 1.5),
        ("to", 1.5, 1.6),
        ("acme", 1.6, 1.9),
        ("dot", 1.9, 2.0),
        ("com", 2.0, 2.2),
    )
    ad = QuoteAd(
        start_quote="this episode is sponsored by acme",
        end_quote="go to acme dot com",
        content_type="promotional_external",
        confidence=0.9,
    )
    matches = resolve_ads(
        [ad], words, [], chunk_start=0.0, chunk_end=3.0
    )
    assert matches[0].method == "word"
    assert match_rate(matches) == 1.0
