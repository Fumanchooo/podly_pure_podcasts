#!/usr/bin/env python3
"""Quote-anchor eval vs frozen auto-gold labels. Scout off. Local CPU Whisper.

  PYTHONPATH=src python scripts/experiments/run_quote_anchor.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from podcast_processor.experiments.auto_gold.constants import DEFAULT_GEMINI_MODEL
from podcast_processor.experiments.prod_vs_auto_gold import (
    CAKE_ENV_LOCAL,
    load_env_file,
)
from podcast_processor.experiments.quote_anchor_eval import (
    DEFAULT_SILENCE_CACHE,
    PRIOR_UTC_SPEND_USD,
    run_quote_anchor_eval,
)
from shared import defaults as DEFAULTS
from shared.env import gemini_api_key

REPO_ROOT = Path(__file__).resolve().parents[2]
CURSOR_ENV = Path("/home/jdr/cursor/.env")

BASELINE = {
    "inner_span_catch": 0.965,
    "opening_inner": "7/7",
    "not_ad_cut": 2.7,
    "overcut": 9.8,
    "undercut": 16.6,
    "edge_error": 26.4,
    "start_error_median": 0.00,
    "start_error_p90": 1.09,
    "end_error_median": 0.07,
    "end_error_p90": 3.60,
}


def _load_keys(env_file: Path) -> None:
    load_env_file(env_file)
    if CURSOR_ENV.is_file() and CURSOR_ENV.resolve() != env_file.resolve():
        load_env_file(CURSOR_ENV)
    key = gemini_api_key() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("LLM_API_KEY", key)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/auto_gold/runs/quote-anchor"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("docs/experiments/auto_gold/runs/prod-scorecard"),
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path(
            "docs/experiments/auto_gold/runs/prod-scorecard/scorecard.sqlite"
        ),
    )
    parser.add_argument("--env-file", type=Path, default=CAKE_ENV_LOCAL)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("WHISPER_LOCAL_MODEL") or "base",
    )
    parser.add_argument("--prior-usd", type=float, default=PRIOR_UTC_SPEND_USD)
    parser.add_argument(
        "--silence-cache",
        type=Path,
        default=DEFAULT_SILENCE_CACHE,
    )
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--cut-mode", choices=("pure", "hybrid"), default="pure")
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Reuse cached word timestamps; never call Whisper.",
    )
    parser.add_argument("--pad-seconds", type=float, default=0.0)
    parser.add_argument("--llm-cache-dir", type=Path, default=None)
    parser.add_argument("--budget-ledger", type=Path, default=None)
    parser.add_argument("--words-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM:
        logging.error("enable_bow_scout_gemini_confirm must stay false")
        return 2
    _load_keys(args.env_file)
    if not gemini_api_key() and not os.environ.get("LLM_API_KEY"):
        logging.error("GEMINI_API_KEY / LLM_API_KEY missing")
        return 2

    result = run_quote_anchor_eval(
        repo_root=REPO_ROOT,
        labels_path=args.labels,
        transcripts_dir=args.audio_dir / "transcripts",
        audio_dir=args.audio_dir,
        sqlite_path=args.sqlite,
        words_dir=args.words_dir or (args.output_dir / "words"),
        output_dir=args.output_dir,
        model=args.model,
        whisper_model=args.whisper_model,
        prior_usd=args.prior_usd,
        silence_cache_dir=args.silence_cache,
        prompt_version=args.prompt_version,
        cut_mode=args.cut_mode,
        skip_transcribe=args.skip_transcribe,
        pad_seconds=args.pad_seconds,
        llm_cache_dir=args.llm_cache_dir,
        budget_ledger=args.budget_ledger,
    )
    sc = result.scorecard.totals
    pin, pout = result.tokens.per_episode(max(1, result.scorecard.n_shows_scored))
    bpin, bpout = result.baseline_tokens.per_episode()
    opening = f"{sc.n_opening_span_caught}/{sc.n_opening_windows}"
    inner = sc.span_seconds_recall()
    edge = sc.overcut_seconds + sc.undercut_seconds
    gates = {
        "inner_span_catch": inner >= 0.965 - 1e-9,
        "opening_inner": opening == "7/7",
        "not_ad_cut": sc.not_ad_seconds_cut <= 2.7 + 0.05,
        "edge_error": edge < 26.4,
        "no_silent_drop": result.n_dropped == 0,
        "input_tokens": pin <= bpin + 1.0,
        "n_shows": result.scorecard.n_shows_scored == 10,
    }
    folded = all(gates.values())
    table = {
        "metric": [
            "inner_span_catch",
            "opening_inner",
            "not_ad_cut_s",
            "overcut_s",
            "undercut_s",
            "total_edge_error",
            "start_median",
            "start_p90",
            "end_median",
            "end_p90",
            "quote_match_rate",
            "input_tokens_per_episode",
            "output_tokens_per_episode",
            "spend_usd",
        ],
        "baseline": [
            BASELINE["inner_span_catch"],
            BASELINE["opening_inner"],
            BASELINE["not_ad_cut"],
            BASELINE["overcut"],
            BASELINE["undercut"],
            BASELINE["edge_error"],
            BASELINE["start_error_median"],
            BASELINE["start_error_p90"],
            BASELINE["end_error_median"],
            BASELINE["end_error_p90"],
            None,
            bpin,
            bpout,
            0.0,
        ],
        "quote_anchor": [
            inner,
            opening,
            sc.not_ad_seconds_cut,
            sc.overcut_seconds,
            sc.undercut_seconds,
            edge,
            sc.start_error_median,
            sc.start_error_p90,
            sc.end_error_median,
            sc.end_error_p90,
            result.match_rate,
            pin,
            pout,
            result.spend_usd,
        ],
        "gates": gates,
        "folded": folded,
        "n_quotes": result.n_quotes,
        "n_word": result.n_word,
        "n_segment_fallback": result.n_segment_fallback,
        "n_chunk_fallback": result.n_chunk_fallback,
        "n_dropped": result.n_dropped,
        "utc_total_usd": result.utc_spend_usd,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "| metric | baseline | quote-anchor |",
        "| --- | ---: | ---: |",
        f"| inner-span catch | 96.5% | {inner:.1%} |",
        f"| opening inner-span | 7/7 | {opening} |",
        f"| not_ad cut s | 2.7 | {_fmt(sc.not_ad_seconds_cut, 2)} |",
        f"| overcut s | 9.8 | {_fmt(sc.overcut_seconds, 2)} |",
        f"| undercut s | 16.6 | {_fmt(sc.undercut_seconds, 2)} |",
        f"| total edge error | 26.4 | {_fmt(edge, 2)} |",
        f"| start median/p90 | 0.00 / 1.09 | {_fmt(sc.start_error_median, 2)} / {_fmt(sc.start_error_p90, 2)} |",
        f"| end median/p90 | 0.07 / 3.60 | {_fmt(sc.end_error_median, 2)} / {_fmt(sc.end_error_p90, 2)} |",
        f"| quote match rate | — | {result.match_rate:.1%} "
        f"({result.n_word}/{result.n_quotes}; fallback "
        f"{result.n_segment_fallback} seg + {result.n_chunk_fallback} chunk; "
        f"dropped {result.n_dropped}) |",
        f"| input tokens/episode | {bpin:.0f} | {pin:.0f} |",
        f"| output tokens/episode | {bpout:.0f} | {pout:.0f} |",
        f"| Gemini $ (session / UTC day) | $0 / $0.0027 | "
        f"${result.spend_usd:.4f} / ${result.utc_spend_usd:.4f} |",
        "",
        f"gates: {json.dumps(gates)}",
        f"folded: {folded}",
    ]
    (args.output_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if result.n_dropped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
