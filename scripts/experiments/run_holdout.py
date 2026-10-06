#!/usr/bin/env python3
"""Fresh-show holdout: label with auto-gold judge, score baseline vs quote-anchor.

Does not retune lexical constants. Scout off. Gemini UTC-day cap $0.50.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from podcast_processor.experiments.auto_gold.audio_ops import probe_duration_seconds
from podcast_processor.experiments.auto_gold.constants import (
    DEFAULT_GEMINI_MODEL,
    GOLD_FAMILY,
)
from podcast_processor.experiments.auto_gold.pipeline import AutoGoldConfig, _run_show
from podcast_processor.experiments.auto_gold.scorecard import CLAIM_LOCKED, score_windows
from podcast_processor.experiments.auto_gold.types import ShowSpec
from podcast_processor.experiments.auto_gold.whisper_chunks import ChunkWhisper
from podcast_processor.experiments.auto_gold.judge import GoldJudge
from podcast_processor.experiments.prod_vs_auto_gold import (
    CAKE_ENV_LOCAL,
    CachedLocalWhisper,
    SpendTracker,
    force_local_writer,
    load_env_file,
    make_flask,
    production_config,
    run_production_cut,
    run_quote_anchor_production_cut,
)
from podcast_processor.experiments.quote_anchor_eval import load_silence_gaps
from shared import defaults as DEFAULTS
from shared.env import gemini_api_key

REPO = Path(__file__).resolve().parents[2]
HOLDOUT_SHOWS = REPO / "docs/experiments/auto_gold/holdout_shows.json"
OUT = REPO / "docs/experiments/auto_gold/runs/quote-anchor/holdout"
WORDS = OUT / "words"
LLM_CACHE = OUT / "llm_cache_v2"
SILENCE = REPO / "docs/experiments/auto_gold/runs/boundary-snap-eval/silence_cache"

JUDGE_CAP = 0.20
PIPELINE_CAP = 0.28  # leftover room under $0.50


def _load_keys() -> None:
    load_env_file(CAKE_ENV_LOCAL)
    load_env_file(Path("/home/jdr/cursor/.env"))
    key = gemini_api_key() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("LLM_API_KEY", key)
        os.environ.setdefault("LLM_MODEL", DEFAULT_GEMINI_MODEL)


def load_holdout_shows(path: Path) -> list[ShowSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gold_family") != GOLD_FAMILY:
        raise ValueError("holdout gold_family lock broken")
    shows: list[ShowSpec] = []
    for row in payload["shows"]:
        shows.append(
            ShowSpec(
                show_id=str(row["id"]).strip(),
                title=str(row["title"]).strip(),
                publisher=str(row.get("publisher") or "").strip(),
                genre=str(row.get("genre") or "").strip(),
                ad_mechanism=str(row.get("ad_mechanism") or "").strip(),
                rss_url=str(row["rss_url"]).strip(),
                notes=str(row.get("notes") or "").strip(),
            )
        )
    return shows


def windows_from_labels(results) -> list[dict]:
    windows: list[dict] = []
    for show in results:
        for chunk, label in zip(show.candidates, show.labels, strict=False):
            windows.append(
                {
                    "show_id": show.show.show_id,
                    "start": float(chunk.start),
                    "end": float(chunk.end),
                    "sources": list(chunk.sources),
                    "is_ad": bool(label.is_ad) and not label.skipped,
                    "skipped": bool(label.skipped),
                    "skip_reason": label.skip_reason,
                    "content_type": label.content_type,
                    "confidence": label.confidence,
                    "ad_spans": label.ad_spans,
                    "usd": label.usd,
                }
            )
    return windows


def holdout_ok(baseline, quote) -> dict:
    b_inner = baseline.totals.span_seconds_recall()
    q_inner = quote.totals.span_seconds_recall()
    b_not = baseline.totals.not_ad_seconds_cut
    q_not = quote.totals.not_ad_seconds_cut
    not_ad_pool = max(baseline.totals.not_ad_seconds, 1.0)
    catch_ok = q_inner + 1e-12 >= b_inner - 0.01
    not_ad_ok = q_not <= b_not + 0.01 * not_ad_pool + 1e-12
    return {
        "catch_ok": catch_ok,
        "not_ad_ok": not_ad_ok,
        "pass": catch_ok and not_ad_ok,
        "baseline_inner": b_inner,
        "quote_inner": q_inner,
        "inner_delta_pp": (q_inner - b_inner) * 100.0,
        "baseline_not_ad": b_not,
        "quote_not_ad": q_not,
        "not_ad_pool": not_ad_pool,
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
    shows = load_holdout_shows(HOLDOUT_SHOWS)
    labels_path = OUT / "labels.json"
    downloads_path = OUT / "downloads.json"

    if labels_path.is_file() and downloads_path.is_file():
        logging.info("reusing holdout labels %s", labels_path)
        labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
        windows = list(labels_payload.get("windows") or [])
        downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
        judge_spend = float(labels_payload.get("judge_spend_usd") or 0.0)
    else:
        cfg = AutoGoldConfig(
            output_dir=OUT,
            download=True,
            enable_dsp=True,
            include_dai_probes=True,
            max_judge_usd=JUDGE_CAP,
        )
        whisper = ChunkWhisper(mode="local", model_name="base")
        judge = GoldJudge(mode="gemini", max_usd=JUDGE_CAP)
        results = []
        downloads = {}
        for show in shows:
            logging.info("holdout fetch/label %s", show.show_id)
            result = _run_show(show, cfg, whisper, judge)
            results.append(result)
            audio = result.audio_path
            if audio is not None and result.episode and result.episode.duration_seconds is None:
                probed = probe_duration_seconds(audio)
                if probed is not None:
                    result.episode.duration_seconds = probed
            downloads[show.show_id] = {
                "resolved_title": result.episode.episode_title if result.episode else show.title,
                "audio_path": str(audio) if audio else None,
                "error": result.episode.error if result.episode else "no episode",
                "duration_s": result.episode.duration_seconds if result.episode else None,
                "ad_mechanism": show.ad_mechanism,
                "genre": show.genre,
            }
            (OUT / "metrics_partial.json").write_text(
                json.dumps({"n": len(results), "judge_usd": judge.spent_usd}, indent=2)
                + "\n",
                encoding="utf-8",
            )
        windows = windows_from_labels(results)
        judge_spend = float(judge.spent_usd)
        labels_payload = {
            "gold_family": GOLD_FAMILY,
            "judge_spend_usd": judge_spend,
            "judge_calls": judge.n_calls,
            "n_is_ad": sum(1 for row in windows if row.get("is_ad")),
            "windows": windows,
            "untouched_constants": True,
        }
        labels_path.write_text(json.dumps(labels_payload, indent=2) + "\n", encoding="utf-8")
        downloads_path.write_text(json.dumps(downloads, indent=2) + "\n", encoding="utf-8")

    have_audio = [sid for sid, row in downloads.items() if row.get("audio_path")]
    if not have_audio:
        logging.error("no holdout audio downloaded")
        return 2

    force_local_writer()
    tracker = SpendTracker()
    tracker.install()
    try:
        bl_config = production_config(enable_quote_anchor_boundaries=False)
        qa_config = production_config(enable_quote_anchor_boundaries=True)
        bl_app = make_flask(OUT / "baseline.sqlite")
        qa_app = make_flask(OUT / "quote.sqlite")
        transcriber = CachedLocalWhisper(logging.getLogger("holdout"), "base")
        baseline_prod: dict[str, dict] = {}
        quote_prod: dict[str, dict] = {}
        walls_bl: dict[str, float] = {}
        walls_qa: dict[str, float] = {}
        for show_id in have_audio:
            if tracker.usd > PIPELINE_CAP:
                logging.warning("pipeline cap reached before %s", show_id)
                break
            audio = Path(str(downloads[show_id]["audio_path"]))
            if not audio.is_absolute():
                audio = REPO / audio
            title = str(downloads[show_id].get("resolved_title") or show_id)
            t0 = time.time()
            baseline_prod[show_id] = run_production_cut(
                show_id=show_id,
                title=title,
                audio_path=audio,
                output_dir=OUT / "baseline",
                config=bl_config,
                app=bl_app,
                transcriber=transcriber,
                tracker=tracker,
            )
            walls_bl[show_id] = time.time() - t0
            t1 = time.time()
            quote_prod[show_id] = run_quote_anchor_production_cut(
                show_id=show_id,
                title=title,
                audio_path=audio,
                output_dir=OUT / "quote-anchor",
                config=qa_config,
                app=qa_app,
                transcriber=transcriber,
                tracker=tracker,
                words_cache=WORDS / f"{show_id}.json",
                llm_cache_dir=LLM_CACHE,
                skip_audio_clip=True,
                require_llm_cache=False,
            )
            walls_qa[show_id] = time.time() - t1
            logging.info(
                "%s baseline=%.1fs qa=%.1fs spend=$%.3f",
                show_id,
                walls_bl[show_id],
                walls_qa[show_id],
                tracker.usd,
            )
    finally:
        tracker.restore()

    scored = set(baseline_prod) & set(quote_prod)
    labeled = [row for row in windows if not row.get("skipped") and row.get("show_id") in scored]
    bl_cuts = {
        sid: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        for sid, row in baseline_prod.items()
        if sid in scored
    }
    qa_cuts = {
        sid: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        for sid, row in quote_prod.items()
        if sid in scored
    }
    baseline = score_windows(labeled, bl_cuts)
    quote = score_windows(labeled, qa_cuts)
    gate = holdout_ok(baseline, quote)
    summary = {
        "gold_family": GOLD_FAMILY,
        "claim": CLAIM_LOCKED,
        "scout": False,
        "untouched_constants": True,
        "n_shows": len(scored),
        "shows": sorted(scored),
        "judge_spend_usd": judge_spend,
        "pipeline_spend_usd": tracker.usd,
        "baseline": {
            "inner": baseline.totals.span_seconds_recall(),
            "opening": f"{baseline.totals.n_opening_span_caught}/{baseline.totals.n_opening_windows}",
            "not_ad": baseline.totals.not_ad_seconds_cut,
            "edge": baseline.totals.overcut_seconds + baseline.totals.undercut_seconds,
            "wall_s_per_episode": (
                sum(walls_bl.values()) / max(1, len(walls_bl))
            ),
        },
        "quote_anchor": {
            "inner": quote.totals.span_seconds_recall(),
            "opening": f"{quote.totals.n_opening_span_caught}/{quote.totals.n_opening_windows}",
            "not_ad": quote.totals.not_ad_seconds_cut,
            "edge": quote.totals.overcut_seconds + quote.totals.undercut_seconds,
            "wall_s_per_episode": (
                sum(walls_qa.values()) / max(1, len(walls_qa))
            ),
            "n_quotes": sum(int(r.get("n_quotes") or 0) for r in quote_prod.values()),
            "n_word": sum(int(r.get("n_word") or 0) for r in quote_prod.values()),
            "n_dropped": sum(int(r.get("n_dropped") or 0) for r in quote_prod.values()),
        },
        "gate": gate,
        "walls_baseline": walls_bl,
        "walls_quote_anchor": walls_qa,
    }
    (OUT / "score.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
