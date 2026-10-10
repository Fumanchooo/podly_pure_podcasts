from __future__ import annotations

from podcast_processor.quote_anchor import (
    QuoteAd,
    QuoteMatch,
    SegmentSpan,
    WordTiming,
    fallback_segment_edge,
    find_quote_matches,
    is_midstack_cta,
    match_quote_tokens,
    match_quote_with_fallbacks,
    match_quotes_in_order,
    match_rate,
    normalize_tokens,
    parse_quote_ads,
    resolve_ads,
    token_edit_distance,
    union_adjacent_matches,
    uniqueness_issues,
    uniqueness_reask_prompt,
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


def test_start_only_grows_back_across_leftover_eof_chunk() -> None:
    """Live Indicator 2026-10-07: leftover 1-seg chunk after a sister-show."""
    words = _words(
        ("This", 632.9, 633.2),
        ("election", 633.2, 633.6),
        ("season.", 633.6, 634.0),
        ("Follow", 648.7, 649.0),
        ("up", 649.0, 649.2),
        ("first.", 649.2, 654.4),
        ("Midlife", 654.4, 654.8),
        ("brings", 654.8, 655.2),
        ("changes.", 655.2, 656.5),
        ("Your", 656.5, 656.8),
        ("turn.", 656.8, 661.4),
        ("Find", 667.1, 667.4),
        ("it", 667.4, 667.6),
        ("on", 667.6, 667.8),
        ("the", 667.8, 668.0),
        ("NPR", 668.0, 668.3),
        ("app.", 668.3, 672.8),
        ("Head", 672.8, 673.1),
        ("comes", 673.1, 673.4),
        ("from", 673.4, 673.7),
        ("NPR", 673.7, 674.0),
        ("sponsor", 674.0, 674.4),
        ("AARP.", 674.4, 675.0),
    )
    segments = [
        SegmentSpan(622.4, 626.6, "It was fact checked by Sierra Wattas"),
        SegmentSpan(626.6, 632.9, "of NPR."),
        SegmentSpan(632.9, 638.2, "This election season you can streamline your news"),
        SegmentSpan(648.7, 654.4, "Follow up first on the NPR app or wherever you get your podcasts."),
        SegmentSpan(654.4, 656.5, "Midlife brings a lot of changes."),
        SegmentSpan(656.5, 661.4, "Your turn with Alas and Aubrey is a new podcast"),
        SegmentSpan(661.4, 667.1, "From health and relationships to creativity and purpose"),
        SegmentSpan(667.1, 672.8, "With your turn, find it on the NPR app or wherever you get your podcasts."),
        SegmentSpan(672.8, 675.0, "Head comes from NPR sponsor AARP."),
    ]
    ad = QuoteAd(
        start_quote="Head comes from NPR sponsor AARP",
        end_quote="Head comes from NPR sponsor AARP",
        content_type="promotional_external",
        confidence=0.95,
    )
    matches = resolve_ads(
        [ad],
        words,
        segments,
        chunk_start=672.8,
        chunk_end=675.0,
        stop_before=652.485,
    )
    assert matches[0].dropped is False
    assert matches[0].method == "word"
    assert abs(matches[0].start - 654.4) < 1e-6
    assert abs(matches[0].end - 675.0) < 1e-6
    assert matches[0].start >= 652.485


def test_start_only_back_stops_at_prior_match() -> None:
    words = _words(
        ("show", 640.0, 640.5),
        ("content.", 640.5, 641.0),
        ("This", 641.0, 641.3),
        ("week", 641.3, 641.6),
        ("on", 641.6, 641.9),
        ("Sister.", 641.9, 650.0),
        ("Head", 650.0, 650.3),
        ("comes", 650.3, 650.6),
        ("from", 650.6, 650.9),
        ("AARP.", 650.9, 652.0),
    )
    segments = [
        SegmentSpan(640.0, 641.0, "show content."),
        SegmentSpan(641.0, 650.0, "This week on Sister Show wherever you get your podcasts."),
        SegmentSpan(650.0, 652.0, "Head comes from NPR sponsor AARP."),
    ]
    ad = QuoteAd(
        start_quote="Head comes from NPR sponsor AARP",
        end_quote="Head comes from NPR sponsor AARP",
        content_type="promotional_external",
        confidence=0.95,
    )
    matches = resolve_ads(
        [ad],
        words,
        segments,
        chunk_start=650.0,
        chunk_end=652.0,
        stop_before=641.0,
    )
    assert matches[0].start == 641.0
    assert matches[0].end == 652.0


def test_start_only_grows_through_adjacent_segments() -> None:
    words = _words(
        ("this", 1921.8, 1922.2),
        ("week", 1922.2, 1922.6),
        ("on", 1922.6, 1923.0),
        ("sources", 1923.0, 1923.4),
        ("and", 1923.4, 1923.8),
        ("methods", 1923.8, 1924.2),
        ("defense", 1924.2, 1924.6),
        ("secretary", 1924.6, 1925.2),
        ("speaks.", 1925.2, 1925.7),
        ("we", 1925.8, 1926.0),
        ("are", 1926.0, 1926.2),
        ("no", 1926.2, 1926.4),
        ("longer", 1926.4, 1926.8),
        ("woke.", 1926.8, 1928.0),
    )
    segments = [
        SegmentSpan(
            1921.8,
            1925.7,
            "This week on Sources and Methods Defense Secretary speaks.",
        ),
        SegmentSpan(1925.8, 1928.0, "We are no longer woke."),
        SegmentSpan(1940.0, 1944.0, "Power is up for grabs in Wisconsin."),
    ]
    ad = QuoteAd(
        start_quote="This week on Sources and Methods Defense",
        end_quote="totally missing closer phrase here",
        content_type="educational/self_promo",
        confidence=0.95,
    )
    matches = resolve_ads([ad], words, segments, chunk_start=1920.0, chunk_end=1950.0)
    assert matches[0].dropped is False
    assert matches[0].method == "word"
    assert matches[0].start == 1921.8
    assert matches[0].end == 1928.0


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


def test_full_sentence_match_uses_first_and_last_word() -> None:
    words = _words(
        ("Support", 10.0, 10.3),
        ("for", 10.3, 10.5),
        ("this", 10.5, 10.7),
        ("podcast", 10.7, 11.1),
        ("comes", 11.1, 11.4),
        ("from", 11.4, 11.6),
        ("Acme.", 11.6, 12.0),
        ("Visit", 20.0, 20.3),
        ("acme.com/save", 20.3, 21.0),
        ("today.", 21.0, 21.4),
        ("Terms", 21.4, 21.7),
        ("apply.", 21.7, 22.0),
    )
    ads = [
        QuoteAd(
            start_quote="Support for this podcast comes from Acme.",
            end_quote="Terms apply.",
            content_type="promotional_external",
            confidence=0.95,
        )
    ]
    ordered = match_quotes_in_order(words, ads, full_sentence=True)
    assert ordered[0][1] is not None
    i0, i1, score = ordered[0][1]
    assert words[i0].word == "Support"
    assert words[i1 - 1].word == "apply."
    assert score >= 0.72


def test_full_sentence_fallback_to_short_head_on_paraphrase() -> None:
    words = _words(
        ("Support", 1.0, 1.2),
        ("for", 1.2, 1.4),
        ("this", 1.4, 1.6),
        ("podcast", 1.6, 2.0),
        ("comes", 2.0, 2.3),
        ("from", 2.3, 2.5),
        ("Acme", 2.5, 2.9),
        ("Insurance.", 2.9, 3.4),
    )
    paraphrased = (
        "Support for this podcast comes from Acme Insurance and all their "
        "wonderful partners around the country today."
    )
    hit = match_quote_with_fallbacks(words, paraphrased, tail=False)
    assert hit is not None
    assert words[hit[0]].word == "Support"


def test_stacked_sister_shows_union_to_last_sentence() -> None:
    """Planet Money 1217–1311: do not stop at the first NPR-app CTA."""
    words = _words(
        ("This", 1217.0, 1217.2),
        ("week", 1217.2, 1217.4),
        ("on", 1217.4, 1217.5),
        ("Ted", 1217.5, 1217.7),
        ("Radio", 1217.7, 1218.0),
        ("Hour.", 1218.0, 1218.4),
        ("Listen", 1255.0, 1255.3),
        ("on", 1255.3, 1255.4),
        ("the", 1255.4, 1255.5),
        ("NPR", 1255.5, 1255.8),
        ("app.", 1255.8, 1263.0),
        ("And", 1264.0, 1264.2),
        ("this", 1264.2, 1264.4),
        ("week", 1264.4, 1264.6),
        ("on", 1264.6, 1264.7),
        ("Fresh", 1264.7, 1265.0),
        ("Air.", 1265.0, 1265.4),
        ("Listen", 1305.0, 1305.3),
        ("on", 1305.3, 1305.4),
        ("the", 1305.4, 1305.5),
        ("NPR", 1305.5, 1305.8),
        ("app", 1305.8, 1306.1),
        ("or", 1306.1, 1306.2),
        ("wherever", 1306.2, 1306.6),
        ("you", 1306.6, 1306.7),
        ("get", 1306.7, 1306.9),
        ("your", 1306.9, 1307.1),
        ("podcasts.", 1307.1, 1311.0),
    )
    first = QuoteMatch(
        ad=QuoteAd(
            start_quote="This week on Ted Radio Hour.",
            end_quote="Listen on the NPR app.",
            content_type="educational/self_promo",
            confidence=0.95,
        ),
        start=1217.0,
        end=1263.0,
        method="word",
        score=0.9,
    )
    second = QuoteMatch(
        ad=QuoteAd(
            start_quote="And this week on Fresh Air.",
            end_quote="Listen on the NPR app or wherever you get your podcasts.",
            content_type="educational/self_promo",
            confidence=0.95,
        ),
        start=1264.0,
        end=1311.0,
        method="word",
        score=0.9,
    )
    unioned = union_adjacent_matches([first, second], gap_s=8.0)
    assert len(unioned) == 1
    assert unioned[0].start == 1217.0
    assert unioned[0].end == 1311.0
    assert "wherever" in unioned[0].ad.end_quote.lower()
    assert is_midstack_cta("Listen on the NPR app.")
    ads = [
        QuoteAd(
            start_quote="This week on Ted Radio Hour.",
            end_quote="Listen on the NPR app or wherever you get your podcasts.",
            content_type="educational/self_promo",
            confidence=0.95,
        )
    ]
    ordered = match_quotes_in_order(words, ads, full_sentence=True)
    assert ordered[0][1] is not None
    i0, i1, _score = ordered[0][1]
    assert words[i0].word == "This"
    assert "podcasts" in words[i1 - 1].word.lower()


def test_fallback_bound_clips_donate_npr_chunk() -> None:
    words = _words(("Give", 100.0, 100.4), ("at", 100.4, 100.6), ("donate.npr.org", 100.6, 101.2))
    segments = [
        SegmentSpan(71.0, 90.0, "Headlines from around the country continue here."),
        SegmentSpan(100.0, 102.0, "Give at donate.npr.org"),
        SegmentSpan(120.0, 155.0, "More show content after the membership ask."),
    ]
    ad = QuoteAd(
        start_quote="totally missing sponsor sentence here xyzzy",
        end_quote="also missing end quote abcdef",
        content_type="educational/self_promo",
        confidence=0.95,
    )
    matches = resolve_ads(
        [ad],
        words,
        segments,
        chunk_start=71.0,
        chunk_end=155.0,
        word_from=0,
        word_to=len(words),
        fallback_bound=(96.0, 105.0),
    )
    assert matches[0].method in {"segment_fallback", "chunk_fallback"}
    assert matches[0].start >= 96.0 - 1e-6
    assert matches[0].end <= 105.0 + 1e-6
    assert matches[0].end - matches[0].start < 30.0


def _stacked_npr_words() -> list[WordTiming]:
    return _words(
        ("This", 1217.0, 1217.2),
        ("week", 1217.2, 1217.4),
        ("on", 1217.4, 1217.5),
        ("Ted", 1217.5, 1217.7),
        ("Radio", 1217.7, 1218.0),
        ("Hour.", 1218.0, 1218.4),
        ("Listen", 1258.0, 1258.3),
        ("on", 1258.3, 1258.4),
        ("the", 1258.4, 1258.5),
        ("NPR", 1258.5, 1258.8),
        ("app.", 1258.8, 1263.0),
        ("This", 1288.0, 1288.2),
        ("is", 1288.2, 1288.3),
        ("Tony", 1288.3, 1288.6),
        ("Mosley", 1288.6, 1289.0),
        ("co-host", 1289.0, 1289.4),
        ("of", 1289.4, 1289.5),
        ("Fresh", 1289.5, 1289.8),
        ("Air.", 1289.8, 1290.2),
        ("Listen", 1309.0, 1309.2),
        ("on", 1309.2, 1309.3),
        ("the", 1309.3, 1309.4),
        ("NPR", 1309.4, 1309.7),
        ("app", 1309.7, 1310.0),
        ("or", 1310.0, 1310.1),
        ("wherever", 1310.1, 1310.4),
        ("you", 1310.4, 1310.5),
        ("get", 1310.5, 1310.7),
        ("Fresh", 1310.7, 1310.9),
        ("Air.", 1310.9, 1311.2),
    )


def test_repeated_cta_is_not_unique_fresh_air_closer_is() -> None:
    words = _stacked_npr_words()
    cta_hits = find_quote_matches(words, "listen on the NPR app")
    assert len(cta_hits) >= 2
    unique_end = find_quote_matches(words, "wherever you get Fresh Air")
    assert len(unique_end) == 1
    assert words[unique_end[0][1] - 1].word.lower().startswith("air")
    unique_start = find_quote_matches(words, "This week on Ted Radio Hour")
    assert len(unique_start) == 1
    issues = uniqueness_issues(
        words,
        [
            QuoteAd(
                start_quote="This week on Ted Radio Hour",
                end_quote="listen on the NPR app",
                content_type="educational/self_promo",
                confidence=0.95,
            )
        ],
    )
    assert any(item.field == "end_anchor" and item.n_matches >= 2 for item in issues)
    prompt = uniqueness_reask_prompt(issues)
    assert "listen on the NPR app" in prompt
    assert "matched" in prompt
    ads = parse_quote_ads(
        '{"ads":[{"start_anchor":"This week on Ted Radio Hour",'
        '"end_anchor":"wherever you get Fresh Air",'
        '"content_type":"educational/self_promo","confidence":0.95}]}'
    )
    assert ads[0].start_quote.startswith("This week")
    assert "Fresh Air" in ads[0].end_quote
    matched = resolve_ads(
        ads,
        words,
        [
            SegmentSpan(1217.0, 1263.0, "This week on Ted Radio Hour. Listen on the NPR app."),
            SegmentSpan(1288.0, 1311.2, "This is Tony Mosley co-host of Fresh Air. Listen on the NPR app or wherever you get Fresh Air."),
        ],
        chunk_start=1217.0,
        chunk_end=1311.2,
        unique_anchors=True,
    )
    assert matched
    assert matched[0].start <= 1218.0
    assert matched[0].end >= 1310.5


def test_first_last_snaps_repeated_cta_to_last_in_window() -> None:
    """Sentence-variant 1263 CTA: last occurrence is Fresh Air ~1311."""
    words = _stacked_npr_words()
    ad = QuoteAd(
        start_quote="This week on Ted Radio Hour",
        end_quote="listen on the NPR app",
        content_type="educational/self_promo",
        confidence=0.95,
    )
    first_best = resolve_ads(
        [ad],
        words,
        [
            SegmentSpan(1217.0, 1263.0, "This week on Ted Radio Hour. Listen on the NPR app."),
            SegmentSpan(1288.0, 1311.2, "This is Tony Mosley co-host of Fresh Air. Listen on the NPR app."),
        ],
        chunk_start=1217.0,
        chunk_end=1311.2,
    )
    assert first_best[0].end <= 1263.5
    matched = resolve_ads(
        [ad],
        words,
        [
            SegmentSpan(1217.0, 1263.0, "This week on Ted Radio Hour. Listen on the NPR app."),
            SegmentSpan(1288.0, 1311.2, "This is Tony Mosley co-host of Fresh Air. Listen on the NPR app."),
        ],
        chunk_start=1217.0,
        chunk_end=1311.2,
        first_last=True,
        window_bound=(1217.0, 1311.2),
    )
    assert matched[0].start <= 1218.0
    assert matched[0].end >= 1309.5
    assert matched[0].end <= 1311.2 + 1e-6
    assert matched[0].start < matched[0].end


def test_first_last_start_is_first_occurrence() -> None:
    words = _words(
        ("support", 10.0, 10.2),
        ("comes", 10.2, 10.4),
        ("from", 10.4, 10.6),
        ("acme", 10.6, 11.0),
        ("show", 20.0, 20.4),
        ("support", 30.0, 30.2),
        ("comes", 30.2, 30.4),
        ("from", 30.4, 30.6),
        ("acme", 30.6, 31.0),
        ("visit", 40.0, 40.2),
        ("acme", 40.2, 40.5),
        ("com", 40.5, 41.0),
    )
    ad = QuoteAd(
        start_quote="support comes from acme",
        end_quote="visit acme com",
        content_type="promotional_external",
        confidence=0.9,
    )
    matched = resolve_ads(
        [ad],
        words,
        [],
        chunk_start=10.0,
        chunk_end=41.0,
        first_last=True,
        window_bound=(10.0, 41.0),
    )
    assert matched[0].start == 10.0
    assert matched[0].end == 41.0


def test_first_last_never_past_window_and_skips_pad_repeat() -> None:
    """A CTA in padding after the spotted window must not pull the end."""
    words = _stacked_npr_words() + _words(
        ("Listen", 1400.0, 1400.2),
        ("on", 1400.2, 1400.3),
        ("the", 1400.3, 1400.4),
        ("NPR", 1400.4, 1400.7),
        ("app.", 1400.7, 1401.0),
    )
    ad = QuoteAd(
        start_quote="This week on Ted Radio Hour",
        end_quote="listen on the NPR app",
        content_type="educational/self_promo",
        confidence=0.95,
    )
    matched = resolve_ads(
        [ad],
        words,
        [],
        chunk_start=1217.0,
        chunk_end=1311.2,
        first_last=True,
        window_bound=(1217.0, 1311.2),
    )
    assert matched[0].start <= 1218.0
    assert matched[0].end >= 1309.5
    assert matched[0].end <= 1311.2 + 1e-6
    from podcast_processor.quote_anchor import expand_outward_to_silence

    snapped = expand_outward_to_silence(
        matched[0].start,
        matched[0].end,
        [(1216.5, 1217.0), (1311.2, 1312.0)],
        bound_start=matched[0].bound_start or matched[0].start,
        bound_end=matched[0].bound_end or matched[0].end,
    )
    assert snapped[0] >= 1217.0 - 1e-6
    assert snapped[1] <= 1311.2 + 1e-6
    assert snapped[0] < snapped[1]
