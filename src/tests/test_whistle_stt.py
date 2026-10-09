"""Cactus Whistle STT: chunk/stitch, token-cap flag, default off."""

from __future__ import annotations

from pathlib import Path

from podcast_processor.quote_anchor import WordTiming
from podcast_processor.quote_anchor_classifier import (
    whistle_stt_enabled,
    word_cache_path_for_audio,
)
from podcast_processor.whistle_transcribe import (
    WHISTLE_BACKEND,
    clip_truncated,
    payload_is_whistle,
    sponsor_keywords,
    stitch_chunk_words,
    transcribe_audio,
)
from shared import defaults as DEFAULTS
from shared.config import Config
from shared.test_utils import create_standard_test_config


def test_whistle_flag_defaults_off() -> None:
    assert DEFAULTS.ENABLE_WHISTLE_STT is False
    config = create_standard_test_config()
    assert config.enable_whistle_stt is False
    assert whistle_stt_enabled(config) is False
    assert Config.model_fields["enable_whistle_stt"].default is False


def test_whistle_cache_path_is_distinct_from_whisper() -> None:
    audio = Path("/tmp/ep.mp3")
    assert word_cache_path_for_audio(audio).name == "ep.mp3.words.json"
    assert (
        word_cache_path_for_audio(audio, backend="whistle").name
        == "ep.mp3.whistle.json"
    )


def test_sponsor_keywords_cap_and_lexicon() -> None:
    keys = sponsor_keywords(limit=8)
    assert 1 <= len(keys) <= 8
    folded = {k.lower() for k in keys}
    assert "shopify" in folded or "netsuite" in folded


def test_stitch_dedupes_overlap_at_seam() -> None:
    left = [
        {"word": "hello", "start": 26.0, "end": 26.2},
        {"word": "sponsored", "start": 27.0, "end": 27.4},
        {"word": "by", "start": 27.4, "end": 27.6},
    ]
    right = [
        {"word": "sponsored", "start": 27.05, "end": 27.4},
        {"word": "by", "start": 27.4, "end": 27.6},
        {"word": "NetSuite", "start": 27.8, "end": 28.2},
    ]
    words, gaps = stitch_chunk_words(
        [(0.0, 28.0, left), (26.5, 54.5, right)], overlap_s=1.5
    )
    labels = [w["word"] for w in words]
    assert labels == ["hello", "sponsored", "by", "NetSuite"]
    assert gaps == []


def test_stitch_logs_seam_gap() -> None:
    left = [{"word": "before", "start": 26.0, "end": 26.3}]
    right = [{"word": "after", "start": 28.2, "end": 28.5}]
    _words, gaps = stitch_chunk_words(
        [(0.0, 28.0, left), (26.5, 54.5, right)], overlap_s=1.5
    )
    assert gaps
    assert gaps[0]["gap_s"] >= 0.4


def test_clip_truncated_near_token_cap() -> None:
    assert clip_truncated(
        n_words=300, last_word_end=27.5, clip_start=0.0, clip_end=28.0
    )
    assert not clip_truncated(
        n_words=12, last_word_end=5.0, clip_start=0.0, clip_end=28.0
    )


def test_transcribe_audio_chunks_and_offsets() -> None:
    import numpy as np

    import podcast_processor.whistle_transcribe as wt

    calls: list[int] = []

    def fake_transcribe(samples: object, **_kwargs: object) -> dict:
        n = len(samples)  # type: ignore[arg-type]
        calls.append(n)
        dur = n / float(wt.SAMPLE_RATE)
        words = [
            {
                "word": f"start{len(calls)}",
                "start": 0.1,
                "end": 0.3,
                "probability": 0.9,
            },
            {
                "word": f"end{len(calls)}",
                "start": max(0.5, dur - 0.6),
                "end": max(0.7, dur - 0.2),
                "probability": 0.9,
            },
        ]
        return {"text": " ".join(w["word"] for w in words), "words": words}

    audio = np.zeros(int(40 * wt.SAMPLE_RATE), dtype=np.float32)
    result = transcribe_audio(
        Path("dummy.wav"), samples=audio, transcribe_fn=fake_transcribe
    )
    assert len(calls) >= 2
    assert all(n <= int(30 * wt.SAMPLE_RATE) for n in calls)
    assert result.words[0]["word"] == "start1"
    assert result.words[-1]["start"] >= 30.0
    payload = result.payload(audio_path="dummy.wav")
    assert payload_is_whistle(payload)
    assert payload["backend"] == WHISTLE_BACKEND


def test_whistle_payload_roundtrip() -> None:
    from podcast_processor.experiments.word_transcripts import words_from_payload

    payload = {
        "backend": "whistle",
        "words": [{"word": "NetSuite", "start": 1.0, "end": 1.4}],
    }
    assert payload_is_whistle(payload)
    assert words_from_payload(payload) == [
        WordTiming(word="NetSuite", start=1.0, end=1.4)
    ]
