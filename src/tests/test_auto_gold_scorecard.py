"""Unit tests for production-vs-auto-gold overlap scoring."""

from __future__ import annotations

import json
from pathlib import Path

import feedparser

from podcast_processor.experiments.auto_gold.downloader import (
    normalize_episode_title,
    pick_audio_entry_by_title,
)
from podcast_processor.experiments.auto_gold.scorecard import (
    CLAIM_LOCKED,
    covered_seconds,
    render_scorecard_md,
    score_windows,
)


def test_catalog_has_twelve_shows_and_fifty_nine_windows() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "experiments"
        / "auto_gold"
        / "scorecard_catalog.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    assert len(episodes) == 12
    n_windows = sum(len(row["windows"]) for row in episodes)
    assert n_windows == 59
    genres = {row["genre"] for row in episodes}
    assert genres == {"news", "comedy", "true_crime", "tech", "sports", "finance"}


def test_title_normalize_curly_quotes() -> None:
    assert normalize_episode_title('MURDERED: Dorothy “Toby” Tate') == (
        normalize_episode_title('MURDERED: Dorothy "Toby" Tate')
    )


def test_pick_pinned_title_not_latest() -> None:
    xml = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Newer episode</title>
      <enclosure url="https://ex.com/new.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Who’s gonna pay for your Social Security?</title>
      <enclosure url="https://ex.com/old.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>""".encode()
    parsed = feedparser.parse(xml)
    entry = pick_audio_entry_by_title(
        parsed, "Who's gonna pay for your Social Security?"
    )
    assert entry is not None
    href = str(entry.enclosures[0].href)
    assert "old.mp3" in href


def test_seconds_recall_and_opening_fn() -> None:
    rows = [
        {
            "show_id": "s1",
            "start": 0.0,
            "end": 90.0,
            "is_ad": True,
            "sources": ["preroll_always"],
        },
        {
            "show_id": "s1",
            "start": 400.0,
            "end": 430.0,
            "is_ad": True,
            "sources": ["dsp_silence"],
        },
        {
            "show_id": "s1",
            "start": 100.0,
            "end": 130.0,
            "is_ad": False,
            "sources": ["dai_probe"],
        },
    ]
    cuts = {"s1": [(0.0, 80.0), (110.0, 125.0)]}
    result = score_windows(rows, cuts)
    t = result.totals
    assert t.n_is_ad_windows == 2
    assert t.n_opening_windows == 1
    assert t.n_t0_windows == 1
    assert t.is_ad_seconds_caught == 80.0
    assert t.is_ad_seconds_missed == 40.0
    assert t.n_is_ad_caught == 1
    assert t.n_is_ad_missed == 1
    assert t.n_opening_caught == 1
    assert t.not_ad_seconds_cut == 15.0
    assert covered_seconds(0.0, 90.0, cuts["s1"]) == 80.0
    empty = covered_seconds(0.0, 90.0, [])
    assert empty == 0.0
    assert isinstance(empty, float)
    overlapping = covered_seconds(0.0, 100.0, [(0.0, 50.0), (40.0, 70.0)])
    assert overlapping == 70.0


def test_opening_seconds_clip_to_preroll() -> None:
    rows = [
        {
            "show_id": "s1",
            "start": 0.0,
            "end": 122.0,
            "is_ad": True,
            "sources": ["preroll_always"],
            "ad_spans": [{"start": 0.0, "end": 59.7}],
        }
    ]
    result = score_windows(rows, {"s1": [(0.0, 54.0)]})
    t = result.totals
    assert t.opening_seconds == 90.0
    assert t.opening_seconds_caught == 54.0
    assert t.opening_span_seconds == 59.7
    assert t.opening_span_seconds_caught == 54.0
    assert t.n_opening_caught == 0
    assert t.n_opening_span_caught == 1
    assert t.is_ad_seconds == 122.0


def test_scorecard_markdown_leads_with_metrics_and_locked_claim() -> None:
    rows = [
        {
            "show_id": "s1",
            "start": 0.0,
            "end": 10.0,
            "is_ad": True,
            "sources": ["preroll_always"],
        }
    ]
    result = score_windows(rows, {"s1": [(0.0, 10.0)]})
    result.spend_usd = 0.0412
    md = render_scorecard_md(result)
    assert md.startswith("# Production remover vs auto-gold scorecard")
    assert "| is_ad audio caught (candidate-window seconds) |" in md
    assert CLAIM_LOCKED in md
    assert "$0.0412" in md
    assert "`False`" in md


def test_inner_ad_span_recall_does_not_require_full_90s_window() -> None:
    rows = [
        {
            "show_id": "s1",
            "start": 0.0,
            "end": 90.0,
            "is_ad": True,
            "sources": ["preroll_always"],
            "ad_spans": [{"start": 0.0, "end": 40.0}],
        }
    ]
    result = score_windows(rows, {"s1": [(0.0, 40.0)]})
    t = result.totals
    assert t.n_opening_caught == 0
    assert t.n_opening_span_caught == 1
    assert t.opening_span_seconds_caught == 40.0
    assert t.opening_span_seconds == 40.0
