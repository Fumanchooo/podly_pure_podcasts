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


def test_url_and_spellout_tokens_match_quote() -> None:
    """Lexicon lock keeps Shopify.com / S-I-N-C-H as one word; still match."""
    shop = _words(
        ("That's", 1930.89, 1931.33),
        ("Shopify.com.", 1931.33, 1932.03),
        ("Go", 1932.03, 1932.40),
        ("to", 1932.40, 1932.55),
        ("Shopify.com", 1932.55, 1933.20),
        ("today", 1933.20, 1933.60),
    )
    hit = match_quote_tokens(shop, "That's Shopify.com. Go to Shopify.com today")
    assert hit is not None
    assert shop[hit[0]].word == "That's"
    assert shop[hit[1] - 1].word == "today"
    assert hit[2] >= 0.99

    spelled = _words(
        ("that's", 2686.6, 2686.9),
        ("spelled", 2686.9, 2687.2),
        ("S-I-N-C-H", 2687.2, 2688.4),
        ("SINCH.com.", 2688.4, 2689.3),
    )
    end = match_quote_tokens(spelled, "that's spelled S-I-N-C-H SINCH.com.")
    assert end is not None
    assert spelled[end[0]].word == "that's"
    assert spelled[end[1] - 1].word == "SINCH.com."
    assert end[2] >= 0.99


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


def test_expand_outward_to_silence_stays_inside_bound() -> None:
    from podcast_processor.quote_anchor import expand_outward_to_silence

    start, end = expand_outward_to_silence(
        10.0,
        12.0,
        [(7.0, 8.0), (13.0, 14.0), (1.0, 2.0), (20.0, 21.0)],
        bound_start=6.0,
        bound_end=15.0,
    )
    assert start == 8.0
    assert end == 13.0


def test_expand_outward_no_silence_keeps_quote() -> None:
    from podcast_processor.quote_anchor import expand_outward_to_silence

    start, end = expand_outward_to_silence(
        10.0, 12.0, [], bound_start=4.0, bound_end=20.0
    )
    assert start == 10.0
    assert end == 12.0


def test_expand_outward_never_wider_than_bound() -> None:
    from podcast_processor.quote_anchor import expand_outward_to_silence

    start, end = expand_outward_to_silence(
        10.0,
        12.0,
        [(0.0, 1.0), (18.0, 19.0)],
        bound_start=9.0,
        bound_end=13.0,
    )
    assert start == 10.0
    assert end == 12.0


def test_expand_pad_then_snap_still_clipped() -> None:
    from podcast_processor.quote_anchor import expand_outward_to_silence

    start, end = expand_outward_to_silence(
        10.0,
        12.0,
        [(7.5, 8.0), (14.0, 14.5)],
        bound_start=8.5,
        bound_end=13.0,
        pad_seconds=2.0,
    )
    assert start == 8.5
    assert end == 13.0


def test_clip_cut_to_baseline_does_not_drop_new_ads() -> None:
    from podcast_processor.quote_anchor import clip_cut_to_baseline

    assert clip_cut_to_baseline(50.0, 60.0, [(0.0, 10.0)]) == (50.0, 60.0)
    assert clip_cut_to_baseline(8.0, 14.0, [(0.0, 12.0)]) == (8.0, 12.0)


def test_hybrid_uses_segment_edge_bound_from_resolve() -> None:
    words = _words(
        ("support", 8.0, 8.3),
        ("comes", 8.3, 8.6),
        ("from", 8.6, 8.9),
        ("acme", 9.0, 9.4),
        ("visit", 11.0, 11.3),
        ("acme", 11.3, 11.6),
        ("dot", 11.6, 11.8),
        ("com", 11.8, 12.0),
    )
    segments = [
        SegmentSpan(start=7.0, end=9.5, text="support comes from acme"),
        SegmentSpan(start=11.0, end=13.0, text="visit acme dot com thanks"),
    ]
    ad = QuoteAd(
        start_quote="comes from acme",
        end_quote="visit acme dot com",
        content_type="promotional_external",
        confidence=0.9,
    )
    matches = resolve_ads([ad], words, segments, chunk_start=7.0, chunk_end=13.0)
    assert matches[0].method == "word"
    assert matches[0].bound_start == 7.0
    assert matches[0].bound_end == 13.0


def test_segment_fallback_when_dai_missing_from_word_stream() -> None:
    """Word-timestamp Whisper dropped the midroll; segment transcript still has it."""
    words = _words(
        ("please.", 401.0, 401.2),
        ("Hey,", 435.98, 436.4),
        ("it's", 439.3, 439.5),
        ("Peter.", 439.7, 440.2),
        ("podcasts.", 456.0, 457.8),
    )
    segments = [
        SegmentSpan(404.36, 409.36, "This message comes from Wise, the smart way"),
        SegmentSpan(419.36, 427.36, "Keep public media public donate.npr.org"),
        SegmentSpan(439.36, 459.36, "Hey it's me Peter wherever you get your podcasts"),
    ]
    ad = QuoteAd(
        start_quote="This message comes from Wise, the smart",
        end_quote="wherever you get your podcasts",
        content_type="promotional_external",
        confidence=0.95,
    )
    matches = resolve_ads(
        [ad], words, segments, chunk_start=400.0, chunk_end=470.0,
        word_from=0, word_to=len(words),
    )
    assert matches[0].method == "segment_fallback"
    assert matches[0].start == 404.36
    assert matches[0].end == 459.36
