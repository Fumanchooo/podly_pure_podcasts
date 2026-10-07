from __future__ import annotations

from podcast_processor.experiments.word_transcripts import (
    ALIGNED_BACKEND,
    group_segment_rows,
    payload_from_aligned_words,
    payload_is_aligned,
)


def test_group_segment_rows_packs_until_max_span() -> None:
    rows = [(0.0, 5.0, "a"), (5.0, 10.0, "b"), (20.0, 25.0, "c")]
    groups = group_segment_rows(rows, max_span_s=28.0)
    assert groups == [rows]


def test_group_segment_rows_splits_when_span_exceeds() -> None:
    rows = [(0.0, 5.0, "a"), (5.0, 20.0, "b"), (20.0, 40.0, "c")]
    groups = group_segment_rows(rows, max_span_s=28.0)
    assert groups == [[rows[0], rows[1]], [rows[2]]]


def test_group_segment_rows_isolates_overlong_segment() -> None:
    rows = [(0.0, 4.0, "a"), (4.0, 40.0, "long"), (41.0, 45.0, "c")]
    groups = group_segment_rows(rows, max_span_s=28.0)
    assert groups == [[rows[0]], [rows[1]], [rows[2]]]


def test_lexicon_lock_keeps_segment_tokens_when_dtw_glues() -> None:
    from podcast_processor.experiments.word_transcripts import lock_words_to_segment_lexicon

    rows = [(864.6, 865.8, "Let's join us again tomorrow.")]
    dtw = [
        {"word": "Let'sjoin", "start": 864.61, "end": 864.95},
        {"word": "us", "start": 864.95, "end": 865.03},
        {"word": "again", "start": 865.03, "end": 865.17},
        {"word": "tomorrow.", "start": 865.17, "end": 865.39},
    ]
    locked = lock_words_to_segment_lexicon(dtw, rows)
    labels = [row["word"].lower() for row in locked]
    assert "join" in labels
    assert "letsjoin" not in {item.replace("'", "") for item in labels}


def test_payload_is_aligned_rejects_decoder_word_timestamps() -> None:
    bad = {"word_timestamps": True, "words": [{"word": "hi", "start": 0, "end": 0.2}]}
    good = payload_from_aligned_words(
        [{"word": "Wise", "start": 404.4, "end": 404.8}],
        [(404.36, 419.36, "This message comes from Wise")],
        model_name="base",
        audio_path="x.mp3",
        align_s=1.2,
    )
    assert payload_is_aligned(bad) is False
    assert payload_is_aligned(good) is True
    assert good["backend"] == ALIGNED_BACKEND
    assert good["word_timestamps"] is False
