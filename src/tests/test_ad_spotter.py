"""$0 lexicon spotter: cues, sponsor names, Whisper variants, clustering."""

from __future__ import annotations

from podcast_processor.ad_spotter import (
    CandidateBreak,
    SpotHit,
    corroborate_candidates,
    drop_weak_candidates,
    gold_misses,
    intra_episode_ngrams,
    is_weak_candidate,
    merge_candidates_for_calls,
    names_in_text,
    repeated_ngrams,
    spot_candidates,
    spot_hits,
)
from podcast_processor.quote_anchor import WordTiming
from shared import defaults as DEFAULTS


def _words(text: str, t0: float = 0.0, dt: float = 0.2) -> list[WordTiming]:
    words: list[WordTiming] = []
    t = t0
    for raw in text.split():
        end = t + dt
        words.append(WordTiming(word=raw, start=t, end=end))
        t = end
    return words


def test_spot_flag_defaults_off() -> None:
    assert DEFAULTS.ENABLE_SPOT_THEN_CONFIRM is False


def test_spots_donate_and_support_membership() -> None:
    words = _words(
        "Your support empowers us. Give at donate.npr.org and thank you."
    )
    hits = spot_hits(words)
    kinds = {h.kind for h in hits}
    cues = {h.cue for h in hits}
    assert "cue" in kinds or "url" in kinds or "sponsor" in kinds
    assert any(
        "support" in c or "donate" in c or c in {"url", "tld"} for c in cues
    )


def test_spots_this_week_on_sister_show() -> None:
    words = _words(
        "This week on Sources and Methods Defense Secretary Pete Hegseth"
    )
    hits = spot_hits(words)
    assert any(
        h.kind in {"cue", "sponsor"}
        and ("this week on" in h.cue or "sources" in h.cue)
        for h in hits
    )


def test_spots_netsuite_and_spoken_url() -> None:
    words = _words(
        "This episode is sponsored by NetSuite. Get the guide at netsuite.com slash darknet."
    )
    hits = spot_hits(words)
    kinds = {h.kind for h in hits}
    assert "cue" in kinds
    assert "sponsor" in kinds or "url" in kinds


def test_spots_spelled_doppel() -> None:
    words = _words("Learn more at doppel.com that's spelled D-O-P-P-E-L Doppel.com.")
    hits = spot_hits(words)
    assert any(h.kind == "sponsor" and "doppel" in h.cue for h in hits)


def test_spots_listen_to_sister_show() -> None:
    words = _words("Listen to So Supernatural wherever you get your podcasts.")
    hits = spot_hits(words)
    assert any(
        "listen" in h.cue or "supernatural" in h.cue or "wherever" in h.cue
        for h in hits
    )


def test_cluster_nearby_hits() -> None:
    words = _words(
        "Support comes from NetSuite. Visit netsuite.com slash darknet. And we're back.",
        t0=100.0,
    )
    cands, stats = spot_candidates(words)
    assert stats.n_hits >= 2
    assert len(cands) == 1
    assert cands[0].end > cands[0].start


def test_gold_misses_unrecoverable() -> None:
    words = _words("Good morning everyone. Here's the news from Washington.")
    cands, _ = spot_candidates(words)
    missed = gold_misses(
        [(0.0, 8.0, "npr_up_first", "Good morning everyone. Here's the news.")],
        cands,
    )
    assert len(missed) == 1
    assert missed[0]["show_id"] == "npr_up_first"


def test_repeated_ngrams_across_episodes() -> None:
    a = _words("This week on Sources and Methods more news today folks")
    b = _words("Later this week on Sources and Methods we talk again")
    phrases = repeated_ngrams({"ep1": a, "ep2": b}, n=4, min_episodes=2)
    assert any("this week on sources" in p for p in phrases)


def test_intra_episode_ngrams_catch_repeated_ad_read() -> None:
    words = _words(
        "Visit donate npr org to support public media. Later visit donate npr org again."
    )
    phrases = intra_episode_ngrams(words, n=4, min_repeats=2)
    assert any("donate" in p and "npr" in p for p in phrases)


def test_drop_lone_company_name_keeps_sponsored_read() -> None:
    weak = _words("The engineers at Microsoft built a database.")
    strong = _words("This episode is sponsored by NetSuite. Visit netsuite.com slash demo.")
    weak_cands, _ = spot_candidates(weak)
    strong_cands, _ = spot_candidates(strong)
    assert weak_cands
    assert all(is_weak_candidate(c) for c in weak_cands)
    assert drop_weak_candidates(weak_cands) == []
    kept = drop_weak_candidates(strong_cands)
    assert kept
    assert not any(is_weak_candidate(c) for c in kept)


def test_merge_candidates_close_together() -> None:
    words = _words(
        "This week on Ted Radio Hour listen on the NPR app. "
        "And this week on Fresh Air listen on the NPR app or wherever you get your podcasts."
    )
    cands, _ = spot_candidates(words, gap_s=0.05, max_span_s=5.0)
    # Force two clusters then merge.
    if len(cands) < 2:
        from podcast_processor.ad_spotter import CandidateBreak, SpotHit

        cands = [
            CandidateBreak(
                start=0.0,
                end=1.0,
                hits=(SpotHit(0.0, 1.0, "cue", "this week on", 1.2),),
                kinds=("cue",),
                weight=1.2,
            ),
            CandidateBreak(
                start=8.0,
                end=10.0,
                hits=(SpotHit(8.0, 10.0, "cue", "listen on the npr app", 1.2),),
                kinds=("cue",),
                weight=1.2,
            ),
        ]
    merged = merge_candidates_for_calls(cands, gap_s=18.0)
    assert len(merged) == 1
    assert merged[0].end >= merged[0].start


def test_exclude_names_drops_leaked_sponsors() -> None:
    words = _words("This episode is sponsored by NetSuite today.")
    hits = spot_hits(words, exclude_names=("netsuite", "NetSuite"))
    assert not any(h.kind == "sponsor" and "netsuite" in h.cue for h in hits)
    mentioned = names_in_text("visit netsuite.com slash darknet")
    assert any("netsuite" in n.lower() for n in mentioned)


def test_loudness_corroborates_but_does_not_spawn_candidates() -> None:
    lexical = [
        CandidateBreak(
            start=10.0,
            end=12.0,
            hits=(SpotHit(10.0, 12.0, "sponsor", "microsoft", 1.6),),
            kinds=("sponsor",),
            weight=1.6,
        )
    ]
    audio = [
        SpotHit(10.5, 14.0, "loudness", "rms_step", 1.15),
        SpotHit(400.0, 410.0, "speaker", "voice_change", 1.25),
    ]
    out = corroborate_candidates(lexical, audio, near_s=8.0)
    assert len(out) == 1
    assert "loudness" in out[0].kinds
    assert "speaker" not in out[0].kinds
    assert not is_weak_candidate(out[0])
