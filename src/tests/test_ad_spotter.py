"""$0 lexicon spotter: cues, sponsor names, Whisper variants, clustering."""

from __future__ import annotations

from podcast_processor.ad_spotter import (
    gold_misses,
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
    assert DEFAULTS.ENABLE_SPOT_WEAK_DROP is False


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
