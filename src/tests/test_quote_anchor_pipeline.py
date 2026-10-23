"""Quote-anchor production path: flag, cutter, fail-closed classify."""

from __future__ import annotations

import inspect
import json

import pytest
from flask import Flask

from app.extensions import db
from app.models import Feed, Identification, ModelCall, Post, TranscriptSegment
from podcast_processor.audio_processor import AudioProcessor
from podcast_processor.podcast_processor import (
    PodcastProcessor,
    quote_anchor_boundaries_enabled,
    quote_anchor_path_enabled,
    spot_then_confirm_enabled,
)
from podcast_processor.quote_anchor import (
    QUOTE_ANCHOR_SOURCE,
    WordTiming,
    keep_ad,
    parse_quote_ads,
)
from podcast_processor.quote_anchor_classifier import QuoteAnchorClassifier
from shared import defaults as DEFAULTS
from shared.config import Config
from shared.config import TestWhisperConfig as WhisperTestConfig
from shared.local_writer import force_local_writer
from shared.test_utils import create_standard_test_config


@pytest.fixture(autouse=True)
def _local_writer() -> None:
    force_local_writer()


def _quote_config() -> Config:
    config = create_standard_test_config()
    return config.model_copy(
        update={
            "enable_quote_anchor_boundaries": True,
            "whisper": WhisperTestConfig(),
        }
    )


def _seed_post(title: str = "Darknet") -> tuple[Post, list[TranscriptSegment]]:
    feed = Feed(title="Test Feed", rss_url="http://example.com/rss.xml")
    db.session.add(feed)
    db.session.commit()
    post = Post(
        feed_id=feed.id,
        title=title,
        guid="qa-guid",
        download_url="http://example.com/audio.mp3",
        unprocessed_audio_path="missing.mp3",
    )
    db.session.add(post)
    db.session.commit()
    segments = [
        TranscriptSegment(
            post_id=post.id,
            sequence_num=0,
            start_time=0.0,
            end_time=2.0,
            text="Good morning everyone.",
        ),
        TranscriptSegment(
            post_id=post.id,
            sequence_num=1,
            start_time=2.0,
            end_time=8.0,
            text="This episode is sponsored by NetSuite. Get the guide at netsuite.com/darknet.",
        ),
        TranscriptSegment(
            post_id=post.id,
            sequence_num=2,
            start_time=8.0,
            end_time=12.0,
            text="And we are back with the story.",
        ),
    ]
    db.session.add_all(segments)
    db.session.commit()
    return post, segments


def test_quote_anchor_flag_defaults_off() -> None:
    assert DEFAULTS.ENABLE_QUOTE_ANCHOR_BOUNDARIES is False
    assert DEFAULTS.ENABLE_SPOT_THEN_CONFIRM is False
    assert DEFAULTS.ENABLE_QUOTE_RESUME_TRIM is False
    assert DEFAULTS.ENABLE_QUOTE_BACKFILL_LEADIN is False
    assert DEFAULTS.ENABLE_QUOTE_CLOSER_TAIL is False
    assert DEFAULTS.ENABLE_SPOT_WEAK_DROP is False
    assert DEFAULTS.ENABLE_SNAP_EARLIER_START is False
    assert DEFAULTS.ENABLE_SHORT_MIN_AD_SEGMENT is False
    assert DEFAULTS.ENABLE_SPOT_MINUTE_PAD is False
    assert DEFAULTS.ENABLE_SPOT_CONFIRM_UNION is False
    assert DEFAULTS.ENABLE_SPOT_WIDE_PAD is False
    assert DEFAULTS.ENABLE_SPOT_BRIDGE_WINDOWS is False
    assert DEFAULTS.ENABLE_SPOT_OPENING_ISOLATE is False
    assert DEFAULTS.ENABLE_QUOTE_PROMPT_FULL_BREAK is False
    config = create_standard_test_config()
    assert config.enable_quote_anchor_boundaries is False
    assert config.enable_spot_then_confirm is False
    assert quote_anchor_boundaries_enabled(config) is False
    assert spot_then_confirm_enabled(config) is False
    assert quote_anchor_path_enabled(config) is False


def test_quote_anchor_flag_can_be_enabled() -> None:
    enabled = create_standard_test_config().model_copy(
        update={"enable_quote_anchor_boundaries": True}
    )
    assert quote_anchor_boundaries_enabled(enabled) is True
    assert create_standard_test_config().enable_quote_anchor_boundaries is False


def test_classify_uses_quote_anchor_when_flag_on() -> None:
    source = inspect.getsource(PodcastProcessor._classify_ad_segments)
    assert "quote_anchor_path_enabled" in source
    assert "self.quote_anchor_classifier.classify" in source
    assert "self.ad_classifier.classify" in source


def test_parse_quote_ads_and_keep() -> None:
    ads = parse_quote_ads(
        '{"ads":[{"start_quote":"this episode is sponsored by netsuite",'
        '"end_quote":"guide at netsuite.com/darknet",'
        '"content_type":"promotional_external","confidence":0.95}]}'
    )
    assert len(ads) == 1
    assert keep_ad(ads[0], 0.8) is True
    tech = parse_quote_ads(
        '{"ads":[{"start_quote":"for example acme corp built",'
        '"end_quote":"the case study ended there",'
        '"content_type":"technical_discussion","confidence":0.9}]}'
    )
    assert keep_ad(tech[0], 0.8) is False


def test_get_ad_segments_uses_quote_cuts_when_flag_on(app: Flask) -> None:
    with app.app_context():
        post, _segments = _seed_post()
        post.refined_ad_boundaries = [
            {
                "source": QUOTE_ANCHOR_SOURCE,
                "empty": False,
                "orig_start": 0.0,
                "orig_end": 0.0,
                "refined_start": 0.0,
                "refined_end": 0.0,
            },
            {
                "source": QUOTE_ANCHOR_SOURCE,
                "orig_start": 2.0,
                "orig_end": 8.0,
                "refined_start": 1.5,
                "refined_end": 8.2,
                "empty": False,
            },
        ]
        db.session.commit()
        processor = AudioProcessor(config=_quote_config(), db_session=db.session)
        cuts = processor.get_ad_segments(post)
        assert cuts == [(1.5, 8.2)]


def test_get_ad_segments_ignores_quote_payload_when_flag_off(app: Flask) -> None:
    with app.app_context():
        post, segments = _seed_post()
        post.refined_ad_boundaries = [
            {
                "source": QUOTE_ANCHOR_SOURCE,
                "orig_start": 2.0,
                "orig_end": 8.0,
                "refined_start": 1.5,
                "refined_end": 8.2,
                "empty": False,
            }
        ]
        model_call = ModelCall(
            post_id=post.id,
            first_segment_sequence_num=0,
            last_segment_sequence_num=2,
            model_name="test",
            prompt="x",
            response='{"ads":[]}',
            status="success",
        )
        db.session.add(model_call)
        db.session.commit()
        ident = Identification(
            transcript_segment_id=segments[1].id,
            model_call_id=model_call.id,
            label="ad",
            confidence=0.9,
        )
        db.session.add(ident)
        db.session.commit()
        processor = AudioProcessor(
            config=create_standard_test_config(), db_session=db.session
        )
        cuts = processor.get_ad_segments(post)
        assert cuts == [(2.0, 8.0)]


def test_classifier_fail_closed_segment_fallback_never_drops(app: Flask) -> None:
    with app.app_context():
        post, segments = _seed_post()
        words = [
            WordTiming("Good", 0.0, 0.3),
            WordTiming("morning", 0.3, 0.8),
            WordTiming("everyone.", 0.8, 1.4),
            WordTiming("This", 2.0, 2.2),
            WordTiming("episode", 2.2, 2.6),
            WordTiming("is", 2.6, 2.7),
            WordTiming("sponsored", 2.7, 3.3),
            WordTiming("by", 3.3, 3.5),
            WordTiming("NetSuite.", 3.5, 4.2),
            WordTiming("Get", 4.3, 4.5),
            WordTiming("the", 4.5, 4.6),
            WordTiming("guide", 4.6, 4.9),
            WordTiming("at", 4.9, 5.0),
            WordTiming("netsuite.com/darknet.", 5.0, 6.5),
            WordTiming("And", 8.0, 8.2),
            WordTiming("we", 8.2, 8.3),
            WordTiming("are", 8.3, 8.4),
            WordTiming("back", 8.4, 8.7),
            WordTiming("with", 8.7, 8.9),
            WordTiming("the", 8.9, 9.0),
            WordTiming("story.", 9.0, 9.5),
        ]

        def complete(_system: str, _user: str) -> str:
            return json.dumps(
                {
                    "ads": [
                        {
                            "start_quote": "this episode is sponsored by netsuite",
                            "end_quote": "get the guide at",
                            "content_type": "promotional_external",
                            "confidence": 0.95,
                        },
                        {
                            "start_quote": "totally invented wording xyzzy",
                            "end_quote": "also invented plugh plover",
                            "content_type": "promotional_external",
                            "confidence": 0.9,
                        },
                    ]
                }
            )

        classifier = QuoteAnchorClassifier(
            config=_quote_config(),
            db_session=db.session,
            complete_fn=complete,
        )
        cuts = classifier.classify(transcript_segments=segments, post=post, words=words)
        assert len(classifier.last_matches) == 2
        assert not any(item.dropped for item in classifier.last_matches)
        methods = {item.method for item in classifier.last_matches}
        assert "word" in methods
        assert "segment_fallback" in methods or "chunk_fallback" in methods
        assert len(cuts) == 2
        processor = AudioProcessor(config=_quote_config(), db_session=db.session)
        stored = processor.get_ad_segments(post)
        assert len(stored) == 2
        n_idents = (
            db.session.query(Identification)
            .join(TranscriptSegment)
            .filter(TranscriptSegment.post_id == post.id, Identification.label == "ad")
            .count()
        )
        assert n_idents >= 1


def test_classifier_empty_ads_stores_quote_header(app: Flask) -> None:
    with app.app_context():
        post, segments = _seed_post()

        def complete(_system: str, _user: str) -> str:
            return '{"ads":[]}'

        classifier = QuoteAnchorClassifier(
            config=_quote_config(),
            db_session=db.session,
            complete_fn=complete,
        )
        cuts = classifier.classify(transcript_segments=segments, post=post, words=[])
        assert cuts == []
        processor = AudioProcessor(config=_quote_config(), db_session=db.session)
        assert processor.get_ad_segments(post) == []
        payload = post.refined_ad_boundaries
        assert isinstance(payload, list)
        assert payload[0]["source"] == QUOTE_ANCHOR_SOURCE
        assert payload[0]["empty"] is True
