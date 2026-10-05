#!/usr/bin/env python3
"""Score production Podly ad-remover vs frozen auto-gold is_ad windows.

Does not set enable_bow_scout_gemini_confirm. Isolated sqlite. No PR.

  PYTHONPATH=src uv run python scripts/experiments/score_prod_vs_auto_gold.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from podcast_processor.experiments.auto_gold.constants import DEFAULT_JUDGE_BUDGET_USD
from podcast_processor.experiments.prod_vs_auto_gold import (
    CAKE_ENV_LOCAL,
    CachedLocalWhisper,
    SpendTracker,
    build_scorecard,
    download_pinned_episodes,
    force_local_writer,
    judge_frozen_windows,
    load_catalog,
    load_env_file,
    load_production_cache,
    make_flask,
    production_config,
    replay_production_from_cached_llm,
    run_production_cut,
    seed_spend_from_production,
    write_scorecard,
)
from shared import defaults as DEFAULTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/auto_gold/runs/prod-scorecard"),
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=CAKE_ENV_LOCAL)
    parser.add_argument(
        "--max-usd",
        type=float,
        default=DEFAULT_JUDGE_BUDGET_USD,
        help="Cap combined Gemini spend (judge + production classify).",
    )
    parser.add_argument(
        "--skip-production",
        action="store_true",
        help="Download + judge windows only (no full-episode AdClassifier).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score cached labels.json + production/*.json; no RSS/LLM.",
    )
    parser.add_argument(
        "--replay-from",
        type=Path,
        default=None,
        help=(
            "Rebuild cuts from cached ModelCall sqlite/labels in this dir "
            "($0 Gemini). Implies no new classify/judge calls."
        ),
    )
    parser.add_argument(
        "--keep-preroll-self-promo",
        action="store_true",
        help=(
            "Eval-only: skip educational/self_promo demotion at t=0–90s. "
            "Default off. Never pass this to the live Podly service."
        ),
    )
    parser.add_argument(
        "--docs-scorecard",
        type=Path,
        default=None,
        help="Optional extra SCORECARD.md path. Default: docs/.../SCORECARD.md unless --replay-from.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ["PODLY_WRITER_LOCAL_FALLBACK"] = "1"
    load_env_file(args.env_file)

    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM:
        print("enable_bow_scout_gemini_confirm is true; refusing.", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_catalog(args.catalog)
    notes: list[str] = [
        "Original per-chunk 29 is_ad votes were gitignored under "
        "`docs/experiments/auto_gold/runs/`. Headline uses this-audio re-judge on "
        "the frozen BASELINE_REPORT timestamps: 17 is_ad / 58 judged / 1 skipped "
        "(bbc_global_news 1751-1779s, no Whisper). DAI rotated openings vs that "
        "report: Planet Money EZKader→NPR membership (not is_ad here); Bill Simmons "
        "PNC→Ringer intro; Marketplace Odoo→show open; also Wait Wait Carvana→NPR "
        "Politics, Serial YouTube Premium→NYT app, Crime Junkie parenthood→Park "
        "Predators, BBC Whole Foods/Schwab→UK show open.",
    ]

    missing: list[str] = []
    labeled: list[dict] = []
    downloads: dict[str, dict] = {}
    production: dict[str, dict] = {}
    tracker = SpendTracker()
    judge_spend = 0.0

    if args.replay_from is not None:
        source_dir = args.replay_from
        labels_path = source_dir / "labels.json"
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labeled = list(payload.get("windows") or [])
        judge_spend = float(payload.get("judge_spend_usd") or 0.0)
        config = production_config(keep_preroll_self_promo=args.keep_preroll_self_promo)
        assert_scout = DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM
        if assert_scout or config.enable_bow_scout_gemini_confirm:
            print("enable_bow_scout_gemini_confirm is true; refusing.", file=sys.stderr)
            return 2
        production = replay_production_from_cached_llm(
            source_dir=source_dir,
            output_dir=output_dir,
            config=config,
        )
        tracker.usd = 0.0
        notes.append(
            "Opening-FN eval replayed Friday's cached ModelCalls ($0 classify). "
            f"enable_keep_preroll_self_promo={config.enable_keep_preroll_self_promo}."
        )
    elif args.score_only:
        labels_path = output_dir / "labels.json"
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labeled = list(payload.get("windows") or [])
    else:
        downloads = download_pinned_episodes(episodes, output_dir)
        missing = [
            show_id
            for show_id, row in downloads.items()
            if not row.get("audio_path")
        ]
        if missing:
            notes.append(f"Audio missing for: {', '.join(missing)}")
        labeled = judge_frozen_windows(
            episodes, output_dir, downloads, max_usd=args.max_usd
        )

    n_is_ad = sum(1 for row in labeled if row.get("is_ad"))
    if args.replay_from is None:
        judge_spend = sum(float(row.get("usd") or 0) for row in labeled)
        production = load_production_cache(output_dir)
        tracker.usd = max(tracker.usd, judge_spend)
        seed_spend_from_production(tracker, production)
        if not args.skip_production and not args.score_only:
            config = production_config(
                keep_preroll_self_promo=args.keep_preroll_self_promo
            )
            litellm_key = config.llm_api_key
            if not litellm_key:
                notes.append("No LLM_API_KEY/Gemini key; production classify skipped.")
            else:
                tracker.install()
                try:
                    import litellm

                    litellm.api_key = litellm_key
                    app = make_flask(output_dir / "scorecard.sqlite")
                    force_local_writer()
                    whisper_name = "base"
                    if config.whisper is not None and hasattr(config.whisper, "model"):
                        whisper_name = str(config.whisper.model)
                    transcriber = CachedLocalWhisper(
                        logging.getLogger("scorecard_whisper"),
                        whisper_name,
                    )
                    for item in episodes:
                        if item.show_id in production:
                            continue
                        remaining = args.max_usd - tracker.usd
                        if remaining <= 0.02:
                            notes.append(
                                f"Hit ${args.max_usd:.2f} Gemini cap before {item.show_id}."
                            )
                            break
                        audio = downloads.get(item.show_id, {}).get("audio_path")
                        if not audio:
                            continue
                        production[item.show_id] = run_production_cut(
                            show_id=item.show_id,
                            title=item.episode_title,
                            audio_path=Path(audio),
                            output_dir=output_dir,
                            config=config,
                            app=app,
                            transcriber=transcriber,
                            tracker=tracker,
                        )
                finally:
                    tracker.restore()
    unscored = [item.show_id for item in episodes if item.show_id not in production]
    if unscored:
        notes.append(
            f"Hit ${args.max_usd:.2f} Gemini cap after 10/12 shows "
            f"(${tracker.usd:.4f} spent). Audio exists for {', '.join(unscored)} "
            "but no production cuts; this-audio re-judge is_ad=0 on those 10 windows, "
            "so is_ad recall would not change."
        )
    if args.replay_from is None:
        notes.append(
            "Opening FNs: NPR membership/cross-promo, Serial NYT self-promo, Crime Junkie "
            "network promo (0s cut). WTF Withings and Darknet NetSuite prerolls were cut "
            "to the inner ad_span but fail the 90s-candidate 50% bar. Midroll host-reads "
            "on Planet Money / Crime Junkie / Darknet mostly caught; later Wait Wait "
            "underwriting mostly missed."
        )
    else:
        notes.append(
            "H1 replay: keep educational/self_promo at t=0–90s. Inner-span openings "
            "are 6/6; candidate ≥50% still fails on Wait Wait / Crime Junkie / WTF / "
            "Darknet because the 90s gold window includes show open. Planet Money "
            "membership (same NPR pitch as Up First, labeled not_ad) is the new FP."
        )

    result = build_scorecard(
        labeled, production, spend_usd=tracker.usd, notes=notes
    )
    result.n_shows_scored = len(production) or result.n_shows_scored
    docs_scorecard = args.docs_scorecard
    if docs_scorecard is None and args.replay_from is None:
        docs_scorecard = Path("docs/experiments/auto_gold/SCORECARD.md")
    write_scorecard(result, output_dir / "SCORECARD.md")
    if docs_scorecard is not None:
        write_scorecard(result, docs_scorecard)
    t = result.totals
    (output_dir / "score.json").write_text(
        json.dumps(
            {
                "n_is_ad_labels": n_is_ad,
                "n_is_ad_windows_scored": t.n_is_ad_windows,
                "judge_spend_usd": judge_spend,
                "total_spend_usd": tracker.usd,
                "n_production": len(production),
                "missing_audio": missing,
                "seconds_recall": t.seconds_recall(),
                "seconds_caught": t.is_ad_seconds_caught,
                "seconds_total": t.is_ad_seconds,
                "span_seconds_recall": t.span_seconds_recall(),
                "opening_seconds_recall": t.opening_seconds_recall(),
                "opening_seconds_caught": t.opening_seconds_caught,
                "opening_seconds_total": t.opening_seconds,
                "opening_span_seconds_recall": t.opening_span_seconds_recall(),
                "window_recall": t.window_recall(),
                "n_is_ad_caught": t.n_is_ad_caught,
                "opening_window_recall": t.opening_window_recall(),
                "opening_span_window_recall": t.opening_span_window_recall(),
                "t0_window_recall": t.t0_window_recall(),
                "t0_span_window_recall": t.t0_span_window_recall(),
                "not_ad_seconds_cut": t.not_ad_seconds_cut,
                "not_ad_seconds": t.not_ad_seconds,
                "keep_preroll_self_promo": args.keep_preroll_self_promo,
                "replay_from": str(args.replay_from) if args.replay_from else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    printed = docs_scorecard if docs_scorecard is not None else (output_dir / "SCORECARD.md")
    print(printed.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
