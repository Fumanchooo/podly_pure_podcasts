"""Algo-theories flags: each default off; matcher-side $0 refinements."""

from __future__ import annotations

from podcast_processor.ad_spotter import (
    drop_weak_candidates,
    is_weak_candidate,
    spot_candidates,
    spot_hits,
)
from podcast_processor.boundary_snap import SilenceGap, snap_time_to_silence
from podcast_processor.quote_anchor import WordTiming
from podcast_processor.quote_anchor_classifier import (
    quote_backfill_leadin_enabled,
    quote_closer_tail_enabled,
    quote_resume_trim_enabled,
    snap_earlier_start_enabled,
    spot_weak_drop_enabled,
)
from podcast_processor.quote_lexicon import refine_quote_span
from shared import defaults as DEFAULTS
from shared.config import Config
from shared.test_utils import create_standard_test_config


def _words(*pairs: tuple[str, float, float]) -> list[WordTiming]:
    return [WordTiming(word=w, start=s, end=e) for w, s, e in pairs]


def test_algo_theory_flags_default_off() -> None:
    assert DEFAULTS.ENABLE_QUOTE_RESUME_TRIM is False
    assert DEFAULTS.ENABLE_QUOTE_BACKFILL_LEADIN is False
    assert DEFAULTS.ENABLE_QUOTE_CLOSER_TAIL is False
    assert DEFAULTS.ENABLE_SPOT_WEAK_DROP is False
    assert DEFAULTS.ENABLE_SNAP_EARLIER_START is False
    config = create_standard_test_config()
    assert config.enable_quote_resume_trim is False
    assert config.enable_quote_backfill_leadin is False
    assert config.enable_quote_closer_tail is False
    assert config.enable_spot_weak_drop is False
    assert config.enable_snap_earlier_start is False
    assert quote_resume_trim_enabled(config) is False
    assert quote_backfill_leadin_enabled(config) is False
    assert quote_closer_tail_enabled(config) is False
    assert spot_weak_drop_enabled(config) is False
    assert snap_earlier_start_enabled(config) is False
    assert Config.model_fields["enable_quote_resume_trim"].default is False
    assert Config.model_fields["enable_quote_backfill_leadin"].default is False
    assert Config.model_fields["enable_quote_closer_tail"].default is False
    assert Config.model_fields["enable_spot_weak_drop"].default is False
    assert Config.model_fields["enable_snap_earlier_start"].default is False


def test_resume_trim_drops_okay_lets_do_the_show() -> None:
    words = _words(
        ("Get", 40.0, 40.2),
        ("the", 40.2, 40.4),
        ("watch", 40.4, 40.7),
        ("at", 40.7, 40.9),
        ("withings.com.", 40.9, 41.5),
        ("Okay,", 42.0, 42.3),
        ("let's", 42.3, 42.5),
        ("do", 42.5, 42.7),
        ("the", 42.7, 42.8),
        ("show.", 42.8, 43.2),
    )
    off = refine_quote_span(words, 40.0, 43.2)
    assert off[1] == 43.2
    on = refine_quote_span(words, 40.0, 43.2, resume_trim=True)
    assert abs(on[1] - 41.5) < 1e-6


def test_resume_trim_drops_okay_this_is_the_epic() -> None:
    words = _words(
        ("Learn", 590.0, 590.3),
        ("more", 590.3, 590.5),
        ("at", 590.5, 590.7),
        ("doppel.com.", 590.7, 591.9),
        ("Okay,", 595.3, 595.5),
        ("this", 595.5, 595.7),
        ("is", 595.7, 595.9),
        ("the", 595.9, 596.1),
        ("epic", 596.1, 596.4),
        ("story", 596.4, 596.7),
        ("of", 596.7, 596.9),
        ("those.", 596.9, 597.2),
    )
    on = refine_quote_span(words, 590.0, 597.2, resume_trim=True)
    assert abs(on[1] - 591.9) < 1e-6


def test_resume_trim_keeps_this_is_planet_money() -> None:
    words = _words(
        ("Give", 20.0, 20.3),
        ("at", 20.3, 20.5),
        ("donate.npr.org.", 20.5, 21.2),
        ("This", 21.3, 21.5),
        ("is", 21.5, 21.7),
        ("Planet", 21.7, 22.0),
        ("Money.", 22.0, 22.4),
    )
    on = refine_quote_span(words, 20.0, 22.4, resume_trim=True)
    assert abs(on[1] - 22.4) < 1e-6


def test_backfill_leadin_grows_to_pause() -> None:
    words = _words(
        ("From", 1886.0, 1886.3),
        ("NPO.", 1886.3, 1887.4),
        ("Austin", 1890.1, 1890.4),
        ("Tice", 1890.4, 1890.7),
        ("is", 1890.7, 1890.9),
        ("an", 1890.9, 1891.1),
        ("American", 1891.1, 1891.5),
        ("journalist.", 1891.5, 1892.0),
    )
    off = refine_quote_span(words, 1890.4, 1892.0)
    assert abs(off[0] - 1890.4) < 1e-6
    on = refine_quote_span(words, 1890.4, 1892.0, backfill_leadin=True)
    assert abs(on[0] - 1890.1) < 1e-6


def test_closer_tail_includes_this_is_planet_money() -> None:
    words = _words(
        ("Give", 20.0, 20.3),
        ("at", 20.3, 20.5),
        ("donate.npr.org.", 20.5, 24.13),
        ("This", 24.20, 24.4),
        ("is", 24.4, 24.6),
        ("Planet", 24.6, 24.9),
        ("Money.", 24.9, 25.2),
    )
    off = refine_quote_span(words, 20.0, 24.13)
    assert abs(off[1] - 24.13) < 1e-6
    on = refine_quote_span(words, 20.0, 24.13, closer_tail=True)
    assert abs(on[1] - 25.2) < 1e-6


def test_closer_tail_does_not_cross_pause_into_news() -> None:
    words = _words(
        ("thank", 22.0, 22.4),
        ("you.", 22.4, 23.84),
        ("A", 25.75, 26.0),
        ("flight", 26.0, 26.3),
        ("bound", 26.3, 26.6),
        ("for", 26.6, 26.8),
        ("Israel.", 26.8, 27.2),
    )
    on = refine_quote_span(words, 20.0, 23.84, closer_tail=True)
    assert abs(on[1] - 23.84) < 1e-6


def _text_words(text: str, t0: float = 0.0, dt: float = 0.2) -> list[WordTiming]:
    words: list[WordTiming] = []
    t = t0
    for raw in text.split():
        words.append(WordTiming(word=raw, start=t, end=t + dt))
        t += dt
    return words


def test_weak_drop_keeps_url_drops_name_only() -> None:
    name_only = _text_words(
        "NetSuite announced a new product today in Cupertino California."
    )
    url = _text_words(
        "Go to netsuite.com slash darknet and donate whatever you can today."
    )
    weak, _ = spot_candidates(name_only, drop_weak=True)
    strong, _ = spot_candidates(url, drop_weak=True)
    raw_name, _ = spot_candidates(name_only, drop_weak=False)
    assert all(not is_weak_candidate(c) for c in drop_weak_candidates(raw_name) or [])
    # URL/slash hits survive; a lone folded company name does not.
    assert len(strong) >= 1
    kinds = {h.kind for h in spot_hits(url)}
    assert "url" in kinds or "cue" in kinds
    if weak:
        assert all(not is_weak_candidate(c) for c in weak)


def test_earlier_start_does_not_snap_start_later() -> None:
    gaps = [SilenceGap(start=8.8, end=9.0)]
    nearest = snap_time_to_silence(8.7, gaps, radius=2.5, kind="start")
    earlier = snap_time_to_silence(
        8.7, gaps, radius=2.5, kind="start", prefer="earlier_start"
    )
    assert nearest == 9.0
    assert earlier == 8.7
    # End snap is unchanged (still nearest).
    end_gaps = [SilenceGap(start=20.2, end=20.6)]
    assert (
        snap_time_to_silence(
            20.0, end_gaps, radius=2.5, kind="end", prefer="earlier_start"
        )
        == 20.2
    )
