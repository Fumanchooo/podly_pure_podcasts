#!/usr/bin/env python3
"""Score baseline vs pure-quote vs prompt-v2 vs hybrid. $0 except prompt-v2 LLM.

Reuses cached word timestamps. Does not re-transcribe. Scout off.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from podcast_processor.boundary_snap import refine_cuts
from podcast_processor.experiments.auto_gold.scorecard import score_windows
from podcast_processor.experiments.prod_vs_auto_gold import (
    CAKE_ENV_LOCAL,
    load_env_file,
)
from podcast_processor.experiments.quote_anchor_eval import (
    DEFAULT_SILENCE_CACHE,
    load_json,
    load_silence_gaps,
    run_quote_anchor_eval,
)
from shared import defaults as DEFAULTS
from shared.env import gemini_api_key

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
POLICY_PROD = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/production"
AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
SQLITE = AUDIO / "scorecard.sqlite"
RUN = REPO / "docs/experiments/auto_gold/runs/quote-anchor"
WORDS = RUN / "words"
LEDGER = RUN / "gemini_budget.json"
SILENCE = REPO / DEFAULT_SILENCE_CACHE

GATES_INNER = 0.965
GATES_NOT_AD = 2.75
GATES_EDGE = 26.4


def _load_keys() -> None:
    load_env_file(CAKE_ENV_LOCAL)
    load_env_file(Path("/home/jdr/cursor/.env"))
    key = gemini_api_key() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("LLM_API_KEY", key)


def _cuts_from_prod(path: Path) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for item in path.glob("*.json"):
        row = json.loads(item.read_text(encoding="utf-8"))
        sid = str(row.get("show_id") or item.stem)
        out[sid] = [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
    return out


def snap_baseline() -> dict[str, list[tuple[float, float]]]:
    downloads = load_json(AUDIO / "downloads.json")
    snapped: dict[str, list[tuple[float, float]]] = {}
    for sid, cuts in _cuts_from_prod(POLICY_PROD).items():
        if sid in {"bill_simmons", "marketplace"}:
            continue
        audio = Path(str(downloads[sid]["audio_path"]))
        if not audio.is_absolute():
            audio = REPO / audio
        gaps = load_silence_gaps(sid, audio, SILENCE)
        snapped[sid] = refine_cuts(
            cuts, audio_path=audio, gaps=gaps, radius=2.0
        )
    return snapped


def _metrics(result, *, tokens_ok_vs: float | None = None) -> dict:
    sc = result.scorecard.totals
    n = max(1, result.scorecard.n_shows_scored)
    pin, pout = result.tokens.per_episode(n)
    opening = f"{sc.n_opening_span_caught}/{sc.n_opening_windows}"
    inner = sc.span_seconds_recall()
    edge = sc.overcut_seconds + sc.undercut_seconds
    gates = {
        "inner_span_catch": inner >= GATES_INNER - 1e-9,
        "opening_inner": opening == "7/7",
        "not_ad_cut": sc.not_ad_seconds_cut <= GATES_NOT_AD,
        "edge_error": edge < GATES_EDGE,
        "no_silent_drop": result.n_dropped == 0,
        "n_shows": result.scorecard.n_shows_scored == 10,
    }
    if tokens_ok_vs is not None:
        gates["input_tokens"] = pin <= tokens_ok_vs + 1.0
    return {
        "inner": inner,
        "opening": opening,
        "not_ad": sc.not_ad_seconds_cut,
        "overcut": sc.overcut_seconds,
        "undercut": sc.undercut_seconds,
        "edge": edge,
        "start_med": sc.start_error_median,
        "start_p90": sc.start_error_p90,
        "end_med": sc.end_error_median,
        "end_p90": sc.end_error_p90,
        "match_rate": result.match_rate,
        "n_quotes": result.n_quotes,
        "n_word": result.n_word,
        "n_seg": result.n_segment_fallback,
        "n_chunk": result.n_chunk_fallback,
        "n_dropped": result.n_dropped,
        "in_tok": pin,
        "out_tok": pout,
        "spend": result.spend_usd,
        "utc_spend": result.utc_spend_usd,
        "n_shows": result.scorecard.n_shows_scored,
        "gates": gates,
        "folded": all(gates.values()),
    }


def _score_cuts(cuts: dict[str, list[tuple[float, float]]]) -> dict:
    labels = load_json(LABELS)
    rows = [
        row
        for row in labels.get("windows") or []
        if not row.get("skipped") and row.get("show_id") in cuts
    ]
    result = score_windows(rows, cuts)
    opening = f"{result.totals.n_opening_span_caught}/{result.totals.n_opening_windows}"
    inner = result.totals.span_seconds_recall()
    edge = result.totals.overcut_seconds + result.totals.undercut_seconds
    gates = {
        "inner_span_catch": inner >= GATES_INNER - 1e-9,
        "opening_inner": opening == "7/7",
        "not_ad_cut": result.totals.not_ad_seconds_cut <= GATES_NOT_AD,
        "edge_error": edge < GATES_EDGE,
        "no_silent_drop": True,
        "n_shows": len(cuts) == 10,
        "input_tokens": True,
    }
    return {
        "inner": inner,
        "opening": opening,
        "not_ad": result.totals.not_ad_seconds_cut,
        "overcut": result.totals.overcut_seconds,
        "undercut": result.totals.undercut_seconds,
        "edge": edge,
        "start_med": result.totals.start_error_median,
        "start_p90": result.totals.start_error_p90,
        "end_med": result.totals.end_error_median,
        "end_p90": result.totals.end_error_p90,
        "match_rate": None,
        "n_quotes": None,
        "n_word": None,
        "n_seg": None,
        "n_chunk": None,
        "n_dropped": 0,
        "in_tok": None,
        "out_tok": None,
        "spend": 0.0,
        "utc_spend": None,
        "n_shows": len(cuts),
        "gates": gates,
        "folded": all(gates.values()),
    }


def _run_variant(*, name: str, prompt_version: str, cut_mode: str, pad: float, cache: Path, baseline: dict | None) -> dict:
    logging.info("variant %s prompt=%s cut=%s", name, prompt_version, cut_mode)
    out = RUN / name
    result = run_quote_anchor_eval(
        repo_root=REPO,
        labels_path=LABELS,
        transcripts_dir=AUDIO / "transcripts",
        audio_dir=AUDIO,
        sqlite_path=SQLITE,
        words_dir=WORDS,
        output_dir=out,
        skip_transcribe=True,
        prompt_version=prompt_version,
        cut_mode=cut_mode,
        pad_seconds=pad,
        llm_cache_dir=cache,
        budget_ledger=LEDGER,
        prior_usd=0.0,
        silence_cache_dir=SILENCE,
        baseline_cuts=baseline,
    )
    bpin, _ = result.baseline_tokens.per_episode()
    return _metrics(result, tokens_ok_vs=bpin)


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_tok(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM:
        logging.error("scout must stay false")
        return 2
    _load_keys()
    if not gemini_api_key() and not os.environ.get("LLM_API_KEY"):
        logging.error("GEMINI_API_KEY missing")
        return 2

    baseline_cuts = snap_baseline()
    rows = {
        "baseline": _score_cuts(baseline_cuts),
        "pure_quote": _run_variant(
            name="pure-v1",
            prompt_version="v1",
            cut_mode="pure",
            pad=0.0,
            cache=RUN / "llm_cache",
            baseline=baseline_cuts,
        ),
        "hybrid": _run_variant(
            name="hybrid-v1",
            prompt_version="v1",
            cut_mode="hybrid",
            pad=0.0,
            cache=RUN / "llm_cache",
            baseline=baseline_cuts,
        ),
        "prompt_v2": _run_variant(
            name="pure-v2",
            prompt_version="v2",
            cut_mode="pure",
            pad=0.0,
            cache=RUN / "llm_cache_v2",
            baseline=baseline_cuts,
        ),
    }
    # $0 extra: prompt-v2 quotes with hybrid expand.
    rows["hybrid_v2"] = _run_variant(
        name="hybrid-v2",
        prompt_version="v2",
        cut_mode="hybrid",
        pad=0.0,
        cache=RUN / "llm_cache_v2",
        baseline=baseline_cuts,
    )
    rows["lexical_v2"] = _run_variant(
        name="lexical-v2",
        prompt_version="v2",
        cut_mode="lexical",
        pad=0.0,
        cache=RUN / "llm_cache_v2",
        baseline=baseline_cuts,
    )

    order = [
        "baseline",
        "pure_quote",
        "prompt_v2",
        "hybrid_v2",
        "lexical_v2",
    ]
    headers = ["metric", *order]
    metrics = [
        ("inner-span catch", lambda r: _fmt_pct(r["inner"])),
        ("opening inner-span", lambda r: r["opening"]),
        ("not_ad cut s", lambda r: _fmt(r["not_ad"])),
        ("overcut s", lambda r: _fmt(r["overcut"])),
        ("undercut s", lambda r: _fmt(r["undercut"])),
        ("total edge error", lambda r: _fmt(r["edge"])),
        ("start median/p90", lambda r: f"{_fmt(r['start_med'])} / {_fmt(r['start_p90'])}"),
        ("end median/p90", lambda r: f"{_fmt(r['end_med'])} / {_fmt(r['end_p90'])}"),
        (
            "quote match rate",
            lambda r: "—"
            if r["match_rate"] is None
            else f"{r['match_rate']:.1%} ({r['n_word']}/{r['n_quotes']}; drop {r['n_dropped']})",
        ),
        ("input tokens/episode", lambda r: _fmt_tok(r["in_tok"])),
        ("output tokens/episode", lambda r: _fmt_tok(r["out_tok"])),
        ("Gemini $ session", lambda r: _fmt(r["spend"], 4)),
        ("n_shows", lambda r: str(r["n_shows"])),
        ("gates pass", lambda r: "yes" if r["folded"] else "no"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + [":---:" for _ in order]) + " |",
    ]
    for label, fn in metrics:
        lines.append("| " + " | ".join([label] + [fn(rows[k]) for k in order]) + " |")
    folded = [k for k in order if k != "baseline" and rows[k]["folded"]]
    lines.append("")
    lines.append(
        "folded variant: " + (folded[0] if folded else "none (live podly.service unchanged)")
    )
    for key in order:
        lines.append(f"{key} gates: {json.dumps(rows[key]['gates'])}")

    from podcast_processor.quote_lexicon import TUNE_NOTES

    lines.append("")
    lines.append("Tune notes (rule → motivating show; not used as show IDs in the matcher):")
    for note in TUNE_NOTES:
        lines.append(f"- {note.rule}: {note.motivated_by} [{note.class_hint}]")

    lex_cuts = _cuts_from_prod(RUN / "lexical-v2" / "production")
    labels = load_json(LABELS)
    label_rows = [
        row
        for row in labels.get("windows") or []
        if not row.get("skipped") and row.get("show_id") in lex_cuts
    ]
    lines.append("")
    lines.append("Leave-one-show-out (lexical_v2):")
    for leave in sorted(lex_cuts):
        sub = {k: v for k, v in lex_cuts.items() if k != leave}
        result = score_windows(
            [row for row in label_rows if row.get("show_id") != leave],
            sub,
        )
        tot = result.totals
        opening = f"{tot.n_opening_span_caught}/{tot.n_opening_windows}"
        edge = tot.overcut_seconds + tot.undercut_seconds
        lines.append(
            f"- leave {leave}: inner={tot.span_seconds_recall():.1%} "
            f"edge={edge:.2f}s not_ad={tot.not_ad_seconds_cut:.2f}s open={opening}"
        )

    text = "\n".join(lines) + "\n"
    (RUN / "VARIANT_COMPARISON.md").write_text(text, encoding="utf-8")
    (RUN / "VARIANT_COMPARISON.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
