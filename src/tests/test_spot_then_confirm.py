"""Spot-then-confirm: one LLM call per candidate, edge extend, flag default off."""

from __future__ import annotations

import json

from flask import Flask

from app.extensions import db
from app.models import Feed, Post, TranscriptSegment
from podcast_processor.experiments.prod_vs_auto_gold import force_local_writer
from podcast_processor.quote_anchor import WordTiming
from podcast_processor.quote_anchor_classifier import (
    QuoteAnchorClassifier,
    spot_then_confirm_enabled,
)
from shared import defaults as DEFAULTS
from shared.config import TestWhisperConfig as WhisperTestConfig
from shared.test_utils import create_standard_test_config


def _spot_config(*, pad: float = 50.0):
    config = create_standard_test_config()
    return config.model_copy(
        update={
            "enable_spot_then_confirm": True,
            "enable_quote_anchor_boundaries": True,
            "whisper": WhisperTestConfig(),
        }
    ), pad


def _seed(title: str, rows: list[tuple[float, float, str]]) -> tuple[Post, list[TranscriptSegment]]:
    feed = Feed(title="Test Feed", rss_url="http://example.com/rss.xml")
    db.session.add(feed)
    db.session.commit()
    post = Post(
        feed_id=feed.id,
        title=title,
        guid="spot-guid",
        download_url="http://example.com/audio.mp3",
        unprocessed_audio_path="missing.mp3",
    )
    db.session.add(post)
    db.session.commit()
    segments = [
        TranscriptSegment(
            post_id=post.id,
            sequence_num=i,
            start_time=a,
            end_time=b,
            text=text,
        )
        for i, (a, b, text) in enumerate(rows)
    ]
    db.session.add_all(segments)
    db.session.commit()
    return post, segments


def _words_from_rows(rows: list[tuple[float, float, str]]) -> list[WordTiming]:
    words: list[WordTiming] = []
    for start, end, text in rows:
        parts = text.split()
        if not parts:
            continue
        dt = max(0.05, (end - start) / len(parts))
        t = start
        for part in parts:
            words.append(WordTiming(word=part, start=t, end=min(end, t + dt)))
            t += dt
    return words


def test_spot_then_confirm_default_off() -> None:
    assert DEFAULTS.ENABLE_SPOT_THEN_CONFIRM is False
    config = create_standard_test_config()
    assert spot_then_confirm_enabled(config) is False


def test_one_llm_call_per_candidate_not_whole_episode(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 8.0, "Good morning everyone welcome back to the show."),
            (8.0, 16.0, "This episode is sponsored by NetSuite. Get the guide at netsuite.com slash darknet."),
            (16.0, 24.0, "And we are back with the story from last week."),
        ]
        post, segments = _seed("Darknet", rows)
        calls: list[str] = []

        def complete(_system: str, user: str) -> str:
            calls.append(user)
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_quote": "This episode is sponsored by NetSuite",
                            "end_quote": "guide at netsuite.com slash darknet",
                            "content_type": "promotional_external",
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        config, pad = _spot_config(pad=8.0)
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=pad,
        )
        cuts = classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert len(classifier.last_candidates) >= 1
        assert classifier.last_llm_calls == len(calls)
        assert classifier.last_llm_calls == len(classifier.last_candidates) + classifier.last_extends
        assert cuts
        assert any("CANDIDATE" in user for user in calls)


def test_confirm_not_ad_returns_no_cut(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 6.0, "This episode is sponsored by NetSuite in a case study."),
            (6.0, 12.0, "The engineers at the company built a database."),
        ]
        post, segments = _seed("Tech", rows)

        def complete(_system: str, _user: str) -> str:
            return '{"ads":[]}'

        config, pad = _spot_config(pad=4.0)
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=pad,
        )
        cuts = classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert classifier.last_llm_calls >= 1
        assert cuts == []


def test_padding_edge_extends_once(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 80.0, "The hosts talk about the news of the day in great detail here for a long while."),
            (80.0, 90.0, "Ordinary conversation continues without any sponsor mention at all."),
            (90.0, 100.0, "Alpha bravo charlie delta echo foxtrot continue the story."),
            (100.0, 110.0, "This episode is sponsored by NetSuite. Visit netsuite.com slash demo."),
            (110.0, 120.0, "Back to the interview with more of the conversation today."),
        ]
        post, segments = _seed("Extend", rows)
        calls = {"n": 0}

        def complete(_system: str, user: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                # Quote the padding edge so the matcher sits on the window start.
                return json.dumps(
                    {
                        "ads": [
                            {
                                "start_quote": "Alpha bravo charlie delta echo",
                                "end_quote": "Visit netsuite.com slash demo",
                                "content_type": "promotional_external",
                                "confidence": 0.95,
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_quote": "This episode is sponsored by NetSuite",
                            "end_quote": "Visit netsuite.com slash demo",
                            "content_type": "promotional_external",
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        config, pad = _spot_config(pad=9.0)
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=pad,
        )
        classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert classifier.last_extends == 1
        assert classifier.last_llm_calls >= 2


def test_spot_v2_flags_default_off() -> None:
    assert DEFAULTS.ENABLE_QUOTE_FULL_SENTENCE_QUOTES is False
    assert DEFAULTS.ENABLE_SPOT_MERGE_CALLS is False
    assert DEFAULTS.ENABLE_RICHER_SPOTTER is False
    assert DEFAULTS.ENABLE_QUOTE_FALLBACK_BOUND is False
    assert DEFAULTS.ENABLE_QUOTE_UNIQUE_ANCHORS is False


def test_merge_calls_one_llm_for_nearby_spots(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 6.0, "This week on Ted Radio Hour listen on the NPR app."),
            (8.0, 16.0, "And this week on Fresh Air listen on the NPR app or wherever you get your podcasts."),
            (20.0, 24.0, "After the break we continue the story."),
        ]
        post, segments = _seed("Planet Money", rows)
        calls: list[str] = []

        def complete(_system: str, user: str) -> str:
            calls.append(user)
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_quote": "This week on Ted Radio Hour listen on the NPR app.",
                            "end_quote": "Listen on the NPR app or wherever you get your podcasts.",
                            "content_type": "educational/self_promo",
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        config = create_standard_test_config().model_copy(
            update={
                "enable_spot_then_confirm": True,
                "enable_quote_anchor_boundaries": True,
                "enable_spot_merge_calls": True,
                "enable_quote_full_sentence_quotes": True,
                "whisper": WhisperTestConfig(),
            }
        )
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=4.0,
        )
        cuts = classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert classifier.last_llm_calls == 1 + classifier.last_extends
        assert len(classifier.last_candidates) == 1
        assert cuts
        assert cuts[0][0] <= 1.0
        assert cuts[0][1] >= 14.0


def test_midstack_cta_extends_once(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 5.0, "This week on Ted Radio Hour."),
            (6.0, 10.0, "Listen on the NPR app."),
            (12.0, 16.0, "And this week on Fresh Air."),
            (17.0, 24.0, "Listen on the NPR app or wherever you get your podcasts."),
            (80.0, 90.0, "The hosts continue the interview with more details."),
        ]
        post, segments = _seed("Stacked", rows)
        calls = {"n": 0}

        def complete(_system: str, user: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                return json.dumps(
                    {
                        "ads": [
                            {
                                "start_quote": "This week on Ted Radio Hour.",
                                "end_quote": "Listen on the NPR app.",
                                "content_type": "educational/self_promo",
                                "confidence": 0.95,
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_quote": "This week on Ted Radio Hour.",
                            "end_quote": "Listen on the NPR app or wherever you get your podcasts.",
                            "content_type": "educational/self_promo",
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        config = create_standard_test_config().model_copy(
            update={
                "enable_spot_then_confirm": True,
                "enable_quote_anchor_boundaries": True,
                "enable_quote_full_sentence_quotes": True,
                "whisper": WhisperTestConfig(),
            }
        )
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=3.0,
        )
        classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert classifier.last_extends == 1
        assert calls["n"] >= 2
        assert classifier.last_cuts
        assert classifier.last_cuts[0][1] >= 20.0
        assert classifier.last_cuts
        assert classifier.last_cuts[0][1] >= 20.0


def test_unique_anchor_reasks_then_uses_fresh_air_closer(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 6.0, "This week on Ted Radio Hour listen on the NPR app."),
            (8.0, 16.0, "This is Tony Mosley co-host of Fresh Air."),
            (17.0, 24.0, "Listen on the NPR app or wherever you get Fresh Air."),
            (80.0, 90.0, "The hosts continue the interview with more details."),
        ]
        post, segments = _seed("Planet Money", rows)
        calls: list[str] = []

        def complete(_system: str, user: str) -> str:
            calls.append(user)
            if "PREVIOUS ANCHORS REJECTED" in user:
                return json.dumps(
                    {
                        "ads": [
                            {
                                "start_anchor": "This week on Ted Radio Hour",
                                "end_anchor": "wherever you get Fresh Air",
                                "content_type": "educational/self_promo",
                                "confidence": 0.95,
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_anchor": "This week on Ted Radio Hour",
                            "end_anchor": "listen on the NPR app",
                            "content_type": "educational/self_promo",
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        config = create_standard_test_config().model_copy(
            update={
                "enable_spot_then_confirm": True,
                "enable_quote_anchor_boundaries": True,
                "enable_quote_unique_anchors": True,
                "whisper": WhisperTestConfig(),
            }
        )
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=4.0,
        )
        cuts = classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert classifier.last_reasks >= 1
        assert any("PREVIOUS ANCHORS REJECTED" in user for user in calls)
        assert classifier.last_unique_ok >= 1
        assert cuts
        assert cuts[0][0] <= 1.0
        assert cuts[0][1] >= 22.0


def test_unique_anchor_fallback_after_failed_reask(app: Flask) -> None:
    force_local_writer()
    with app.app_context():
        rows = [
            (0.0, 6.0, "This week on Ted Radio Hour listen on the NPR app."),
            (8.0, 16.0, "And this week on Fresh Air listen on the NPR app."),
            (80.0, 90.0, "The hosts continue the interview with more details."),
        ]
        post, segments = _seed("Stacked", rows)

        def complete(_system: str, _user: str) -> str:
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_anchor": "This week on Ted Radio Hour",
                            "end_anchor": "listen on the NPR app",
                            "content_type": "educational/self_promo",
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        config = create_standard_test_config().model_copy(
            update={
                "enable_spot_then_confirm": True,
                "enable_quote_anchor_boundaries": True,
                "enable_quote_unique_anchors": True,
                "whisper": WhisperTestConfig(),
            }
        )
        classifier = QuoteAnchorClassifier(
            config=config,
            db_session=db.session,
            complete_fn=complete,
            spot_pad_s=4.0,
        )
        classifier.classify(
            transcript_segments=segments,
            post=post,
            words=_words_from_rows(rows),
        )
        assert classifier.last_reasks >= 1
        assert classifier.last_unique_fallbacks >= 1
