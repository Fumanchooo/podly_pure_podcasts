#!/usr/bin/env python3
"""Score quote-anchor through the integrated AudioProcessor cutter.

Uses cached word timestamps + prompt_v2 llm_cache_v2 ($0 Gemini). Scout off.
Does not retune constants.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from podcast_processor.experiments.auto_gold.scorecard import (
    CLAIM_LOCKED,
    render_scorecard_md,
    score_windows,
)
from podcast_processor.experiments.prod_vs_auto_gold import (
    CAKE_ENV_LOCAL,
    SpendTracker,
    force_local_writer,
    load_env_file,
    make_flask,
    production_config,
    run_quote_anchor_production_cut,
)
from podcast_processor.experiments.quote_anchor_eval import (
    DEFAULT_SILENCE_CACHE,
    TokenTotals,
    baseline_tokens_from_sqlite,
    load_json,
    load_silence_gaps,
)
from podcast_processor.transcribe import LocalWhisperTranscriber
from shared import defaults as DEFAULTS
from shared.env import gemini_api_key

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
POLICY_PROD = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/production"
AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
SQLITE = AUDIO / "scorecard.sqlite"
RUN = REPO / "docs/experiments/auto_gold/runs/quote-anchor"
WORDS = RUN / "words"
LLM_CACHE = RUN / "llm_cache_v2"
SILENCE = REPO / DEFAULT_SILENCE_CACHE
OUT = RUN / "pipeline-v2"

GATES_INNER = 0.965
GATES_NOT_AD = 2.7
GATES_EDGE = 26.4


def _load_keys() -> None:
    load_env_file(CAKE_ENV_LOCAL)
    load_env_file(Path("/home/jdr/cursor/.env"))
    key = gemini_api_key() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("LLM_API_KEY", key)


def _baseline_snapped() -> dict[str, list[tuple[float, float]]]:
    from podcast_processor.boundary_snap import refine_cuts

    downloads = load_json(AUDIO / "downloads.json")
    snapped: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(POLICY_PROD.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        sid = str(row.get("show_id") or path.stem)
        if sid in {"bill_simmons", "marketplace"}:
            continue
        cuts = [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        audio = Path(str(downloads[sid]["audio_path"]))
        if not audio.is_absolute():
            audio = REPO / audio
        gaps = load_silence_gaps(sid, audio, SILENCE)
        snapped[sid] = refine_cuts(cuts, audio_path=audio, gaps=gaps, radius=2.0)
    return snapped


def _metrics(totals, *, n_shows: int) -> dict:
    inner = totals.span_seconds_recall()
    opening = f"{totals.n_opening_span_caught}/{totals.n_opening_windows}"
    edge = totals.overcut_seconds + totals.undercut_seconds
    return {
        "inner": inner,
        "opening": opening,
        "not_ad": totals.not_ad_seconds_cut,
        "overcut": totals.overcut_seconds,
        "undercut": totals.undercut_seconds,
        "edge": edge,
        "n_shows": n_shows,
        "gates": {
            "inner": inner + 1e-12 >= GATES_INNER,
            "opening": opening == "7/7",
            "not_ad": totals.not_ad_seconds_cut <= GATES_NOT_AD + 1e-12,
            "edge": edge <= GATES_EDGE + 1e-12,
        },
    }


def main() -> int:
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM:
        print("scout must stay false", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["PODLY_WRITER_LOCAL_FALLBACK"] = "1"
    _load_keys()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_json(LABELS)
    windows = list(labels.get("windows") or [])
    downloads = load_json(AUDIO / "downloads.json")
    scored = [
        sid
        for sid, row in downloads.items()
        if row.get("audio_path") and sid not in {"bill_simmons", "marketplace"}
    ]

    def _size(sid: str) -> int:
        audio = Path(str(downloads[sid]["audio_path"]))
        if not audio.is_absolute():
            audio = REPO / audio
        return audio.stat().st_size if audio.is_file() else 1 << 62

    scored.sort(key=_size)
    baseline_cuts = _baseline_snapped()
    baseline = score_windows(
        [row for row in windows if not row.get("skipped") and row.get("show_id") in baseline_cuts],
        baseline_cuts,
    )
    config = production_config(enable_quote_anchor_boundaries=True)
    app = make_flask(OUT / "scorecard.sqlite")
    force_local_writer()
    tracker = SpendTracker()
    transcriber = LocalWhisperTranscriber(logging.getLogger("pipeline"), "base")
    production: dict[str, dict] = {}
    tokens = TokenTotals()
    walls: dict[str, float] = {}
    t0 = time.time()
    with app.app_context():
        for show_id in scored:
            audio = Path(str(downloads[show_id]["audio_path"]))
            if not audio.is_absolute():
                audio = REPO / audio
            title = str(downloads[show_id].get("resolved_title") or show_id)
            started = time.time()
            payload = run_quote_anchor_production_cut(
                show_id=show_id,
                title=title,
                audio_path=audio,
                output_dir=OUT,
                config=config,
                app=app,
                transcriber=transcriber,
                tracker=tracker,
                words_cache=WORDS / f"{show_id}.json",
                llm_cache_dir=LLM_CACHE,
                skip_audio_clip=True,
            )
            walls[show_id] = time.time() - started
            production[show_id] = payload
            tokens.input_tokens += int(payload.get("input_tokens") or 0)
            tokens.output_tokens += int(payload.get("output_tokens") or 0)
            tokens.calls += 1
            tokens.n_episodes += 1
            logging.info(
                "%s wall=%.1fs quotes=%s word=%s seg_fb=%s cuts=%s",
                show_id,
                walls[show_id],
                payload.get("n_quotes"),
                payload.get("n_word"),
                payload.get("n_segment_fallback"),
                len(payload.get("cut_segments_s") or []),
            )
    qa_cuts = {
        sid: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        for sid, row in production.items()
    }
    quote = score_windows(
        [row for row in windows if not row.get("skipped") and row.get("show_id") in qa_cuts],
        qa_cuts,
    )
    btok = baseline_tokens_from_sqlite(SQLITE)
    pin, pout = tokens.per_episode(max(1, len(production)))
    bpin, bpout = btok.per_episode()
    mean_wall = sum(walls.values()) / max(1, len(walls))
    summary = {
        "gold_family": "general_podcast_ads",
        "claim": CLAIM_LOCKED,
        "scout": False,
        "integrated_pipeline": True,
        "baseline": _metrics(baseline.totals, n_shows=len(baseline_cuts)),
        "quote_anchor": _metrics(quote.totals, n_shows=len(production)),
        "tokens_per_episode": {"quote_anchor": pin, "baseline": bpin},
        "output_tokens_per_episode": {"quote_anchor": pout, "baseline": bpout},
        "wall_s_per_episode": mean_wall,
        "wall_s_by_show": walls,
        "n_quotes": sum(int(row.get("n_quotes") or 0) for row in production.values()),
        "n_word": sum(int(row.get("n_word") or 0) for row in production.values()),
        "n_segment_fallback": sum(
            int(row.get("n_segment_fallback") or 0) for row in production.values()
        ),
        "n_chunk_fallback": sum(
            int(row.get("n_chunk_fallback") or 0) for row in production.values()
        ),
        "n_dropped": sum(int(row.get("n_dropped") or 0) for row in production.values()),
        "elapsed_s": time.time() - t0,
    }
    qa_gates = summary["quote_anchor"]["gates"]
    tok_ok = pin <= bpin + 1e-9
    summary["original10_pass"] = all(qa_gates.values()) and tok_ok
    summary["tokens_ok"] = tok_ok
    (OUT / "score.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    quote.notes = [
        "Integrated quote-anchor (prompt_v2 + lexical_v2 + AudioProcessor merge/snap).",
        "Cached words + llm_cache_v2. Scout off. general_podcast_ads LOCKED.",
    ]
    (OUT / "SCORECARD.md").write_text(render_scorecard_md(quote), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["original10_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
