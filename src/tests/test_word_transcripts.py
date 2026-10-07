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


def test_lexicon_mismatch_times_on_dtw_span_not_full_segment() -> None:
    from podcast_processor.experiments.word_transcripts import lexicon_words_for_segment

    dtw = [
        {"word": "Thisepisode", "start": 475.2, "end": 476.0},
        {"word": "Locker.", "start": 477.1, "end": 477.8},
    ]
    locked = lexicon_words_for_segment(
        465.84,
        478.0,
        "This episode is sponsored by Threat Locker.",
        dtw_words=dtw,
    )
    assert locked[0]["word"] == "This"
    assert locked[0]["start"] >= 474.5
    assert locked[-1]["end"] <= 478.05


def test_reanchor_packs_leading_bed_stretch() -> None:
    from podcast_processor.experiments.word_transcripts import reanchor_words_to_speech
    from podcast_processor.quote_anchor import WordTiming

    ident = [
        WordTiming("This", 457.44, 457.75),
        WordTiming("is", 457.75, 458.31),
        WordTiming("Darknet", 458.31, 458.73),
        WordTiming("Diaries.", 458.73, 459.27),
    ]
    # Linear interpol across 465.84–478 music+speech.
    stretched = [
        WordTiming("This", 465.84, 467.58),
        WordTiming("episode", 467.58, 469.31),
        WordTiming("is", 469.31, 471.05),
        WordTiming("sponsored", 471.05, 472.79),
        WordTiming("by", 472.79, 474.53),
        WordTiming("Threat", 474.53, 476.26),
        WordTiming("Locker.", 476.26, 478.00),
    ]
    segs = [
        (457.44, 465.68, "This is Darknet Diaries."),
        (465.84, 478.00, "This episode is sponsored by Threat Locker."),
    ]
    out = reanchor_words_to_speech(ident + stretched, segs, gaps=[])
    locker = [w for w in out if w.start >= 464]
    assert locker[0].word == "This"
    assert locker[0].start >= 474.0
    assert locker[-1].end <= 478.05


def test_reanchor_restores_intra_segment_silence_pause() -> None:
    from podcast_processor.experiments.word_transcripts import reanchor_words_to_speech
    from podcast_processor.quote_anchor import WordTiming

    words = [
        WordTiming("Plus,", 1208.36, 1209.15),
        WordTiming("we", 1209.15, 1209.94),
        WordTiming("go", 1209.94, 1210.73),
        WordTiming("back", 1210.73, 1211.52),
        WordTiming("to", 1211.52, 1212.31),
        WordTiming("our", 1212.31, 1213.10),
        WordTiming("former", 1213.10, 1213.90),
        WordTiming("chief", 1213.90, 1214.69),
        WordTiming("actuary", 1214.69, 1215.48),
        WordTiming("for", 1215.48, 1216.27),
        WordTiming("a", 1216.27, 1217.06),
        WordTiming("reality", 1217.06, 1217.85),
        WordTiming("check.", 1217.85, 1218.64),
        WordTiming("This", 1218.64, 1218.99),
    ]
    segs = [
        (1208.36, 1218.64, "Plus we go back to our former chief actuary for a reality check."),
        (1218.64, 1223.52, "This election season"),
    ]
    gaps = [(1215.38, 1217.08)]
    out = reanchor_words_to_speech(words, segs, gaps)
    plus = [w for w in out if w.word != "This"]
    nxt = [w for w in out if w.word == "This"][0]
    assert plus[-1].word == "check."
    assert plus[-1].end <= 1215.45
    assert nxt.start - plus[-1].end >= 0.8


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
