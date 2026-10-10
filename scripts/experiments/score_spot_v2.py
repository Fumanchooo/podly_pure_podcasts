#!/usr/bin/env python3
"""Score spot-v2 flags vs live 1b7728d spot-then-confirm.

--spotter-only / --signals / --leakage are $0. Confirm spends Gemini under
the UTC $0.50 ledger. Scout off. Does not touch live podly.service.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from podcast_processor.ad_spotter import (
    SpotHit,
    gold_misses,
    intra_episode_ngrams,
    names_in_text,
    ngram_hits,
    repeated_ngrams,
    signal_prf,
    spot_candidates,
    spot_hits,
)
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
    GeminiBudget,
    TokenTotals,
    load_json,
    load_silence_gaps,
)
from podcast_processor.experiments.word_transcripts import (
    load_word_payload,
    payload_is_aligned,
    words_from_payload,
)
from podcast_processor.quote_anchor import WordTiming
from podcast_processor.sponsor_lexicon import SPONSOR_NAMES, TUNE_NOTES
from podcast_processor.spotter_audio import audio_spot_hits
from podcast_processor.transcribe import LocalWhisperTranscriber
from shared import defaults as DEFAULTS
from shared.env import gemini_api_key

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
RUN = REPO / "docs/experiments/auto_gold/runs/quote-anchor"
# Live 1b7728d production cuts from the 2026-10-09 fold.
LIVE_ORIG = RUN / "spot-confirm" / "original10"
HOLDOUT = RUN / "holdout"
HOLDOUT_LABELS = HOLDOUT / "labels.json"
HOLDOUT_LIVE = RUN / "spot-confirm" / "holdout"
SILENCE = REPO / DEFAULT_SILENCE_CACHE
LEDGER = RUN / "gemini_budget.json"
UTC_CAP = 0.50
SKIP_SHOWS = {"bill_simmons", "marketplace"}
CALL_SHOWS = ("atp", "npr_wait_wait")

ABLATIONS = {
    "none": {},
    "sentence": {"enable_quote_full_sentence_quotes": True},
    "merge": {"enable_spot_merge_calls": True},
    "richer": {"enable_richer_spotter": True},
    "bound": {"enable_quote_fallback_bound": True},
    "all": {
        "enable_quote_full_sentence_quotes": True,
        "enable_spot_merge_calls": True,
        "enable_richer_spotter": True,
        "enable_quote_fallback_bound": True,
    },
    "unique": {"enable_quote_unique_anchors": True},
    "unique_merge": {
        "enable_quote_unique_anchors": True,
        "enable_spot_merge_calls": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="spot-v2")
    parser.add_argument("--spotter-only", action="store_true")
    parser.add_argument("--signals", action="store_true")
    parser.add_argument("--leakage", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--require-llm-cache", action="store_true")
    parser.add_argument("--ablation", default="all", choices=sorted(ABLATIONS))
    parser.add_argument("--extra-name", action="append", default=[])
    parser.add_argument("--extra-phrase", action="append", default=[])
    return parser.parse_args()


def _load_keys() -> None:
    load_env_file(CAKE_ENV_LOCAL)
    load_env_file(Path("/home/jdr/cursor/.env"))
    key = gemini_api_key() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("LLM_API_KEY", key)


def _words(path: Path) -> list[WordTiming]:
    payload = load_word_payload(path)
    if payload and payload_is_aligned(payload):
        return words_from_payload(payload)
    return []


def _span_text(words: list[WordTiming], start: float, end: float) -> str:
    return " ".join(w.word for w in words if w.end > start and w.start < end)


def gold_spans_for(
    windows: list[dict],
    show_ids: set[str],
    words_by_show: dict[str, list[WordTiming]],
) -> list[tuple[float, float, str, str]]:
    spans: list[tuple[float, float, str, str]] = []
    for row in windows:
        if row.get("skipped") or not row.get("is_ad"):
            continue
        sid = str(row.get("show_id") or "")
        if sid not in show_ids:
            continue
        inner = list(row.get("ad_spans") or [])
        if not inner:
            inner = [{"start": row["start"], "end": row["end"]}]
        words = words_by_show.get(sid) or []
        for sp in inner:
            a, b = float(sp["start"]), float(sp["end"])
            spans.append((a, b, sid, _span_text(words, a, b)))
    return spans


def _metrics(totals, *, n_shows: int, opening_need: str | None = None) -> dict:
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
        "opening_ok": opening_need is None or opening == opening_need,
    }


def live_cuts(prod_dir: Path) -> dict[str, list[tuple[float, float]]]:
    cuts: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(prod_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        sid = str(row.get("show_id") or path.stem)
        cuts[sid] = [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
    return cuts


def live_tokens(prod_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in sorted(prod_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        sid = str(row.get("show_id") or path.stem)
        out[sid] = {
            "input_tokens": float(row.get("input_tokens") or 0),
            "output_tokens": float(row.get("output_tokens") or 0),
            "calls": float(row.get("n_llm_calls") or 0),
            "candidates": float(row.get("n_candidates") or 0),
        }
    return out


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values) or 1)


def leaked_names(gold: list[tuple[float, float, str, str]]) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for _a, _b, _sid, text in gold:
        for name in names_in_text(text, SPONSOR_NAMES):
            key = name.lower()
            if key not in seen:
                seen.add(key)
                found.append(name)
    return tuple(found)


def score_signals(
    scored: list[str],
    words_by_show: dict[str, list[WordTiming]],
    gold: list[tuple[float, float, str, str]],
    downloads: dict,
    ngrams: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    by_kind: dict[str, list] = {
        "cue": [],
        "sponsor": [],
        "url": [],
        "ngram": [],
        "intra_ngram": [],
        "loudness": [],
        "speaker": [],
        "combined": [],
    }
    for sid in scored:
        words = words_by_show[sid]
        hits = spot_hits(words)
        intra = intra_episode_ngrams(words)
        intra_h = ngram_hits(words, intra)
        cross_h = ngram_hits(words, ngrams)
        audio = downloads.get(sid, {}).get("audio_path")
        if audio:
            ap = Path(str(audio))
            if not ap.is_absolute():
                ap = REPO / ap
            loud, speak = audio_spot_hits(ap if ap.is_file() else None)
        else:
            loud, speak = [], []
        show_gold = [g for g in gold if g[2] == sid]
        by_kind["cue"].extend((h, show_gold) for h in hits if h.kind == "cue")
        by_kind["sponsor"].extend((h, show_gold) for h in hits if h.kind == "sponsor")
        by_kind["url"].extend(
            (h, show_gold) for h in hits if h.kind in {"url", "promo"}
        )
        by_kind["ngram"].extend((h, show_gold) for h in cross_h)
        by_kind["intra_ngram"].extend((h, show_gold) for h in intra_h)
        by_kind["loudness"].extend((h, show_gold) for h in loud)
        by_kind["speaker"].extend((h, show_gold) for h in speak)
        extra = list(cross_h) + list(intra_h) + list(loud) + list(speak)
        cands, _ = spot_candidates(
            words,
            ngram_phrases=tuple(dict.fromkeys((*ngrams, *intra))),
            extra_hits=extra or None,
        )
        by_kind["combined"].extend(
            (
                SpotHit(
                    start=c.start,
                    end=c.end,
                    kind="combined",
                    cue="cand",
                    weight=c.weight,
                ),
                show_gold,
            )
            for c in cands
        )
    report: dict[str, dict[str, float]] = {}
    for kind, rows in by_kind.items():
        if kind == "combined":
            hits = [h for h, _g in rows if h is not None]
            gold_all = gold
            report[kind] = signal_prf(hits, gold_all)
            continue
        hits = [h for h, _g in rows]
        report[kind] = signal_prf(hits, gold)
    return report


def main() -> int:
    args = parse_args()
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM:
        print("scout must stay false", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["PODLY_WRITER_LOCAL_FALLBACK"] = "1"
    _load_keys()

    holdout = bool(args.holdout)
    labels_path = HOLDOUT_LABELS if holdout else LABELS
    downloads_path = (HOLDOUT / "downloads.json") if holdout else (AUDIO / "downloads.json")
    live_dir = (HOLDOUT_LIVE / "production") if holdout else (LIVE_ORIG / "production")
    words_dir = RUN / "words-aligned"
    transcripts_dir = (HOLDOUT / "baseline" / "transcripts") if holdout else (AUDIO / "transcripts")
    tag = args.ablation
    if args.leakage:
        tag = f"{tag}-leak"
    out = RUN / args.out / tag / ("holdout" if holdout else "original10")
    out.mkdir(parents=True, exist_ok=True)

    labels = load_json(labels_path)
    windows = list(labels.get("windows") or [])
    downloads = load_json(downloads_path)
    scored = [
        sid
        for sid, row in downloads.items()
        if row.get("audio_path") and (holdout or sid not in SKIP_SHOWS)
    ]
    if holdout:
        scored = [
            sid
            for sid in scored
            if (words_dir / f"{sid}.json").is_file()
            or (HOLDOUT / "words" / f"{sid}.json").is_file()
        ]
    else:
        scored = [sid for sid in scored if (words_dir / f"{sid}.json").is_file()]

    extra_phrases = tuple(args.extra_phrase)
    extra_names = tuple(args.extra_name)
    words_by_show: dict[str, list[WordTiming]] = {}
    catalog: dict[str, list[WordTiming]] = {}
    for sid in scored:
        path = words_dir / f"{sid}.json"
        if not path.is_file():
            path = HOLDOUT / "words" / f"{sid}.json"
        words_by_show[sid] = _words(path)
        catalog[sid] = words_by_show[sid]
    ngrams = repeated_ngrams(catalog, n=6, min_episodes=2)

    gold = gold_spans_for(windows, set(scored), words_by_show)
    leaked = leaked_names(gold)
    exclude = leaked if args.leakage else ()
    flags = dict(ABLATIONS[args.ablation])
    drop_weak = bool(flags.get("enable_spot_merge_calls"))
    merge_calls = drop_weak
    richer = bool(flags.get("enable_richer_spotter"))

    missed_all: list[dict] = []
    spotter_rows: dict[str, dict] = {}
    for sid in scored:
        extra_hits = []
        corroborating = []
        ngram_phrases = ngrams
        if richer:
            intra = intra_episode_ngrams(words_by_show[sid])
            ngram_phrases = tuple(dict.fromkeys((*ngrams, *intra)))
            audio = downloads.get(sid, {}).get("audio_path")
            if audio:
                ap = Path(str(audio))
                if not ap.is_absolute():
                    ap = REPO / ap
                if ap.is_file():
                    loud, speak = audio_spot_hits(ap)
                    corroborating.extend(loud)
                    corroborating.extend(speak)
        cands, stats = spot_candidates(
            words_by_show[sid],
            extra_phrases=extra_phrases,
            extra_names=extra_names,
            exclude_names=exclude,
            ngram_phrases=ngram_phrases,
            extra_hits=extra_hits or None,
            corroborating_hits=corroborating or None,
            drop_weak=drop_weak,
            merge_calls=merge_calls,
        )
        show_gold = [g for g in gold if g[2] == sid]
        missed = gold_misses(show_gold, cands)
        missed_all.extend(missed)
        spotter_rows[sid] = {
            "n_candidates": stats.n_candidates,
            "n_hits": stats.n_hits,
            "n_gold": len(show_gold),
            "n_missed": len(missed),
        }
        logging.info(
            "spotter %s candidates=%s hits=%s gold=%s missed=%s",
            sid,
            stats.n_candidates,
            stats.n_hits,
            len(show_gold),
            len(missed),
        )

    n_gold = len(gold)
    n_miss = len(missed_all)
    recall = 1.0 if n_gold == 0 else (n_gold - n_miss) / float(n_gold)
    signals = {}
    if args.signals:
        signals = score_signals(scored, words_by_show, gold, downloads, ngrams)
    tuned = [n.item for n in TUNE_NOTES if n.tuned_on_original10]
    spotter_summary = {
        "n_gold_ads": n_gold,
        "n_missed": n_miss,
        "recall": recall,
        "leakage": bool(args.leakage),
        "leaked_names": list(leaked),
        "exclude_names": list(exclude),
        "tuned_on_seen": tuned,
        "ngrams": list(ngrams[:40]),
        "per_show": spotter_rows,
        "calls_focus": {sid: spotter_rows.get(sid, {}) for sid in CALL_SHOWS},
        "signals": signals,
        "missed": [
            {
                "show_id": m["show_id"],
                "start": m["start"],
                "end": m["end"],
                "text": str(m["text"])[:500],
            }
            for m in missed_all
        ],
        "holdout": holdout,
        "ablation": args.ablation,
    }
    (out / "spotter.json").write_text(
        json.dumps(spotter_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: spotter_summary[k]
                for k in (
                    "n_gold_ads",
                    "n_missed",
                    "recall",
                    "leaked_names",
                    "calls_focus",
                    "missed",
                )
            },
            indent=2,
        )
    )
    if args.spotter_only or args.leakage or args.signals:
        (out / "signals.json").write_text(
            json.dumps(signals, indent=2) + "\n", encoding="utf-8"
        )
        if args.spotter_only or args.leakage:
            return 0 if args.leakage or n_miss == 0 else 1

    live = live_cuts(live_dir)
    live_tok = live_tokens(live_dir)
    live_scored = score_windows(
        [row for row in windows if not row.get("skipped") and row.get("show_id") in live],
        live,
    )
    opening_need = None if holdout else "7/7"
    config = production_config(
        enable_quote_anchor_boundaries=True,
        enable_spot_then_confirm=True,
        **flags,
    )
    app = make_flask(out / "scorecard.sqlite")
    force_local_writer()
    tracker = SpendTracker()
    budget = GeminiBudget(cap_usd=UTC_CAP, prior_usd=0.0, ledger=LEDGER)
    logging.info(
        "gemini ledger date=%s spent=$%.4f remaining=$%.4f ablation=%s",
        budget.daily.date,
        budget.daily.spent_usd,
        budget.remaining,
        args.ablation,
    )
    transcriber = LocalWhisperTranscriber(logging.getLogger("pipeline"), "base")
    production: dict[str, dict] = {}
    tokens = TokenTotals()
    tracker.install()
    try:
        with app.app_context():
            for show_id in scored:
                audio = Path(str(downloads[show_id]["audio_path"]))
                if not audio.is_absolute():
                    audio = REPO / audio
                if not audio.is_file():
                    logging.warning("missing audio %s", show_id)
                    continue
                remaining = budget.remaining
                if remaining < 0.02 and not args.require_llm_cache:
                    logging.warning("UTC cap before %s remaining $%.3f", show_id, remaining)
                    break
                wpath = words_dir / f"{show_id}.json"
                if not wpath.is_file():
                    wpath = HOLDOUT / "words" / f"{show_id}.json"
                usd_before = tracker.usd
                ngram_phrases = ngrams
                if richer:
                    ngram_phrases = tuple(
                        dict.fromkeys(
                            (*ngrams, *intra_episode_ngrams(words_by_show[show_id]))
                        )
                    )
                payload = run_quote_anchor_production_cut(
                    show_id=show_id,
                    title=str(downloads[show_id].get("resolved_title") or show_id),
                    audio_path=audio,
                    output_dir=out,
                    config=config,
                    app=app,
                    transcriber=transcriber,
                    tracker=tracker,
                    words_cache=wpath,
                    llm_cache_dir=out / "llm_cache",
                    skip_audio_clip=True,
                    transcripts_dir=transcripts_dir,
                    require_llm_cache=args.require_llm_cache,
                    extra_phrases=extra_phrases,
                    extra_names=extra_names,
                    ngram_phrases=ngram_phrases,
                    silence_gaps=[
                        (g.start, g.end)
                        for g in load_silence_gaps(show_id, audio, SILENCE)
                    ],
                )
                production[show_id] = payload
                tokens.input_tokens += int(payload.get("input_tokens") or 0)
                tokens.output_tokens += int(payload.get("output_tokens") or 0)
                tokens.calls += int(payload.get("n_llm_calls") or 0)
                tokens.n_episodes += 1
                delta = tracker.usd - usd_before
                if delta > 0:
                    budget.charge(
                        delta,
                        input_tokens=int(payload.get("input_tokens") or 0),
                        output_tokens=int(payload.get("output_tokens") or 0),
                    )
                logging.info(
                    "%s calls=%s cands=%s quotes=%s cuts=%s in=%s out=%s spend=$%.4f",
                    show_id,
                    payload.get("n_llm_calls"),
                    payload.get("n_candidates"),
                    payload.get("n_quotes"),
                    len(payload.get("cut_segments_s") or []),
                    payload.get("input_tokens"),
                    payload.get("output_tokens"),
                    tracker.usd,
                )
    finally:
        tracker.restore()

    cand_cuts = {
        sid: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        for sid, row in production.items()
    }
    cand_scored = score_windows(
        [row for row in windows if not row.get("skipped") and row.get("show_id") in cand_cuts],
        cand_cuts,
    )
    n = max(1, len(production))
    live_n = max(1, len(live_tok))
    per_show_calls = {
        sid: int(row.get("n_llm_calls") or 0) for sid, row in production.items()
    }
    summary = {
        "gold_family": "general_podcast_ads",
        "claim": CLAIM_LOCKED,
        "scout": False,
        "spot_then_confirm": True,
        "ablation": args.ablation,
        "flags": flags,
        "holdout": holdout,
        "live_commit": "1b7728d",
        "gemini_usd": tracker.usd,
        "spotter_recall": recall,
        "n_spotter_missed": n_miss,
        "leaked_names": list(leaked),
        "signals": signals,
        "live": _metrics(live_scored.totals, n_shows=len(live), opening_need=opening_need),
        "candidate": _metrics(
            cand_scored.totals, n_shows=len(production), opening_need=opening_need
        ),
        "calls_per_episode": {
            "live": mean([v["calls"] for v in live_tok.values()]),
            "candidate": tokens.calls / n,
        },
        "calls_focus": {
            sid: {
                "live": live_tok.get(sid, {}).get("calls"),
                "candidate": per_show_calls.get(sid),
            }
            for sid in CALL_SHOWS
        },
        "input_tokens_per_episode": {
            "live": mean([v["input_tokens"] for v in live_tok.values()]),
            "candidate": tokens.input_tokens / n,
        },
        "output_tokens_per_episode": {
            "live": mean([v["output_tokens"] for v in live_tok.values()]),
            "candidate": tokens.output_tokens / n,
        },
        "n_episodes": n,
        "live_n": live_n,
        "reasks": sum(int(row.get("n_reasks") or 0) for row in production.values()),
        "unique_ok": sum(
            int(row.get("n_unique_ok") or 0) for row in production.values()
        ),
        "unique_fallbacks": sum(
            int(row.get("n_unique_fallbacks") or 0) for row in production.values()
        ),
        "reask_rate": (
            sum(int(row.get("n_reasks") or 0) for row in production.values())
            / float(tokens.calls or 1)
        ),
        "spotter_missed": spotter_summary["missed"],
    }
    live_m = summary["live"]
    cand_m = summary["candidate"]
    summary["gates"] = {
        "catch": cand_m["inner"] + 1e-12 >= live_m["inner"],
        "not_ad": cand_m["not_ad"] <= 1e-12,
        "openings": cand_m["opening_ok"]
        and (
            holdout
            or cand_m["opening"] == live_m["opening"]
            or int(cand_m["opening"].split("/")[0])
            >= int(live_m["opening"].split("/")[0])
        ),
        "budget": tracker.usd <= UTC_CAP + 1e-12,
    }
    summary["fold"] = all(summary["gates"].values())
    (out / "score.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "SCORECARD.md").write_text(render_scorecard_md(cand_scored), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["fold"] else 1


if __name__ == "__main__":
    sys.exit(main())
