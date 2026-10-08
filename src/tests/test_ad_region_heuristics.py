from __future__ import annotations

import inspect

from podcast_processor.experiments.ad_region_heuristics import (
    lexical_spans,
    repeated_ngram_spans,
    score_flags_against_gold,
    silence_bracket_spans,
)
from podcast_processor.quote_anchor import SegmentSpan
from podcast_processor.quote_anchor_classifier import QuoteAnchorClassifier


def test_heuristics_not_imported_by_cutter() -> None:
    source = inspect.getsource(QuoteAnchorClassifier)
    assert "ad_region_heuristics" not in source
    from podcast_processor import audio_processor as audio_mod
    from podcast_processor import quote_anchor as quote_mod
    from podcast_processor import quote_anchor_classifier as clf_mod

    assert "ad_region_heuristics" not in inspect.getsource(audio_mod)
    assert "ad_region_heuristics" not in inspect.getsource(quote_mod)
    assert "ad_region_heuristics" not in inspect.getsource(clf_mod)


def test_lexical_flags_promo_and_support() -> None:
    segs = [
        SegmentSpan(0.0, 5.0, "Good morning from the newsroom."),
        SegmentSpan(10.0, 14.0, "Support for this podcast comes from Acme."),
        SegmentSpan(14.0, 18.0, "Use promo code SAVE20 at acme.com/podcast. Terms apply."),
        SegmentSpan(40.0, 45.0, "Back to the story about interest rates."),
    ]
    hits = lexical_spans(segs, duration=50.0, pad_s=1.0)
    assert hits["support_comes_from"]
    assert hits["promo_code"]
    assert hits["terms_apply"]
    assert hits["lexical_any"]
    # Show content should not be the only flagged span
    assert any(a >= 9.0 for a, _ in hits["lexical_any"])


def test_silence_brackets_and_ngram_repeat() -> None:
    segs = [
        SegmentSpan(0.0, 8.0, "hello show"),
        SegmentSpan(12.0, 30.0, "this episode is brought to you by acme today"),
        SegmentSpan(40.0, 50.0, "story continues"),
    ]
    gaps = [(8.0, 12.0), (30.0, 34.0)]
    brackets = silence_bracket_spans(segs, gaps, min_silence_s=0.8)
    assert brackets
    assert brackets[0][0] == 12.0
    by_show = {
        "npr_up_first": [
            SegmentSpan(1.0, 4.0, "support for npr comes from acme insurance"),
        ],
        "npr_indicator": [
            SegmentSpan(10.0, 14.0, "support for npr comes from acme insurance"),
        ],
    }
    repeated = repeated_ngram_spans(by_show, n=5, pad_s=0.0)
    assert repeated["npr_up_first"]
    assert repeated["npr_indicator"]


def test_score_flags_recall_precision() -> None:
    windows = [
        {
            "show_id": "demo",
            "start": 0.0,
            "end": 20.0,
            "is_ad": True,
            "ad_spans": [{"start": 10.0, "end": 18.0}],
        },
        {"show_id": "demo", "start": 40.0, "end": 50.0, "is_ad": False},
    ]
    score = score_flags_against_gold(windows, {"demo": [(10.0, 18.0)]})
    assert score.recall == 1.0
    assert score.not_ad_seconds_flagged == 0.0
    assert score.precision == 1.0
    leak = score_flags_against_gold(windows, {"demo": [(40.0, 50.0)]})
    assert leak.recall == 0.0
    assert leak.not_ad_seconds_flagged == 10.0
