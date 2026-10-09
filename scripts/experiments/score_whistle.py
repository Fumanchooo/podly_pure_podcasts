#!/usr/bin/env python3
"""Score Cactus Whistle STT vs local Whisper on frozen original 10 + holdout.

$0 ASR/spotter by default. --e2e runs spot-then-confirm (Gemini, UTC $0.50
ledger). Does not touch live podly.service. Flag stays default off.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from podcast_processor.ad_spotter import gold_misses, repeated_ngrams, spot_candidates
from podcast_processor.experiments.auto_gold.audio_ops import extract_chunk_wav
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
from podcast_processor.quote_anchor import WordTiming, normalize_tokens, token_edit_distance
from podcast_processor.transcribe import LocalWhisperTranscriber
from podcast_processor.whistle_transcribe import (
    aligned_token_mae,
    payload_is_whistle,
    sponsor_keywords,
    transcribe_word_timestamps_whistle,
    word_holes,
)
from shared import defaults as DEFAULTS
from shared.env import gemini_api_key

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "docs/experiments/auto_gold/runs/policy-fold-2026-10-05/labels.json"
AUDIO = REPO / "docs/experiments/auto_gold/runs/prod-scorecard"
RUN = REPO / "docs/experiments/auto_gold/runs/quote-anchor"
LIVE_ORIG = RUN / "spot-confirm" / "original10" / "production"
HOLDOUT = RUN / "holdout"
HOLDOUT_LABELS = HOLDOUT / "labels.json"
HOLDOUT_LIVE = RUN / "spot-confirm" / "holdout" / "production"
SILENCE = REPO / DEFAULT_SILENCE_CACHE
LEDGER = RUN / "gemini_budget.json"
UTC_CAP = 0.50
SKIP_SHOWS = {"bill_simmons", "marketplace"}
OUT_ROOT = RUN / "whistle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--keywords", action="store_true", help="biased ASR (default: both)")
    parser.add_argument("--plain", action="store_true", help="unbiased ASR only")
    parser.add_argument("--show", action="append", default=[])
    parser.add_argument("--require-llm-cache", action="store_true")
    return parser.parse_args()


def _load_keys() -> None:
    load_env_file(CAKE_ENV_LOCAL)
    load_env_file(Path("/home/jdr/cursor/.env"))
    key = gemini_api_key() or os.environ.get("LLM_API_KEY", "").strip()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)
        os.environ.setdefault("LLM_API_KEY", key)


def _as_words(path: Path) -> list[WordTiming]:
    payload = load_word_payload(path)
    if not payload:
        return []
    if payload_is_aligned(payload) or payload_is_whistle(payload):
        return words_from_payload(payload)
    return []


def _tokens(words: list[WordTiming]) -> list[str]:
    out: list[str] = []
    for word in words:
        out.extend(normalize_tokens(word.word))
    return out


def _wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return token_edit_distance(ref, hyp) / float(len(ref))


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
            spans.append(
                (float(sp["start"]), float(sp["end"]), sid, _span_text(words, float(sp["start"]), float(sp["end"])))
            )
    return spans


def _metrics(totals, *, n_shows: int, opening_need: str | None = None) -> dict:
    opening = f"{totals.n_opening_span_caught}/{totals.n_opening_windows}"
    return {
        "inner": totals.span_seconds_recall(),
        "opening": opening,
        "not_ad": totals.not_ad_seconds_cut,
        "overcut": totals.overcut_seconds,
        "undercut": totals.undercut_seconds,
        "edge": totals.overcut_seconds + totals.undercut_seconds,
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


def scored_shows(holdout: bool, wanted: list[str]) -> tuple[list[str], dict, Path, Path, list]:
    labels_path = HOLDOUT_LABELS if holdout else LABELS
    downloads_path = (HOLDOUT / "downloads.json") if holdout else (AUDIO / "downloads.json")
    labels = load_json(labels_path)
    downloads = load_json(downloads_path)
    scored = [
        sid
        for sid, row in downloads.items()
        if row.get("audio_path") and (holdout or sid not in SKIP_SHOWS)
    ]
    words_dir = RUN / "words-aligned"
    scored = [sid for sid in scored if (words_dir / f"{sid}.json").is_file()]
    if wanted:
        scored = [sid for sid in scored if sid in set(wanted)]
    return scored, downloads, words_dir, Path(labels_path), list(labels.get("windows") or [])


def audio_for(show_id: str, downloads: dict) -> Path:
    audio = Path(str(downloads[show_id]["audio_path"]))
    if not audio.is_absolute():
        audio = REPO / audio
    return audio


def ensure_whistle(
    show_id: str,
    audio: Path,
    cache: Path,
    *,
    keywords: list[str] | None,
) -> list[WordTiming]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        words = _as_words(cache)
        if words:
            return words
    logging.info("whistle transcribe %s keywords=%s", show_id, bool(keywords))
    return transcribe_word_timestamps_whistle(audio, cache, keywords=keywords)


def disagreement_windows(
    whisper: list[WordTiming],
    whistle: list[WordTiming],
    *,
    win_s: float = 20.0,
    hop_s: float = 15.0,
    top_n: int = 2,
) -> list[dict]:
    end = 0.0
    if whisper:
        end = max(end, whisper[-1].end)
    if whistle:
        end = max(end, whistle[-1].end)
    rows: list[dict] = []
    t = 0.0
    while t < end:
        a, b = t, min(end, t + win_s)
        ref = _tokens([w for w in whisper if w.end > a and w.start < b])
        hyp = _tokens([w for w in whistle if w.end > a and w.start < b])
        if len(ref) >= 8:
            rows.append(
                {
                    "start": a,
                    "end": b,
                    "wer": _wer(ref, hyp),
                    "n_whisper": len(ref),
                    "n_whistle": len(hyp),
                    "whisper": " ".join(ref[:80]),
                    "whistle": " ".join(hyp[:80]),
                }
            )
        t += hop_s
    rows.sort(key=lambda r: -float(r["wer"]))
    return rows[:top_n]


def spotter_for(
    scored: list[str],
    words_by_show: dict[str, list[WordTiming]],
    windows: list[dict],
) -> dict:
    catalog = {sid: words_by_show[sid] for sid in scored}
    ngrams = repeated_ngrams(catalog, n=6, min_episodes=2)
    gold = gold_spans_for(windows, set(scored), words_by_show)
    missed_all: list[dict] = []
    per_show: dict[str, dict] = {}
    for sid in scored:
        cands, stats = spot_candidates(words_by_show[sid], ngram_phrases=ngrams)
        show_gold = [g for g in gold if g[2] == sid]
        missed = gold_misses(show_gold, cands)
        missed_all.extend(missed)
        per_show[sid] = {
            "n_candidates": stats.n_candidates,
            "n_hits": stats.n_hits,
            "n_gold": len(show_gold),
            "n_missed": len(missed),
        }
    n_gold = len(gold)
    n_miss = len(missed_all)
    return {
        "n_gold_ads": n_gold,
        "n_missed": n_miss,
        "recall": 1.0 if n_gold == 0 else (n_gold - n_miss) / float(n_gold),
        "per_show": per_show,
        "missed": [
            {
                "show_id": m["show_id"],
                "start": m["start"],
                "end": m["end"],
                "text": str(m["text"])[:400],
            }
            for m in missed_all
        ],
    }


def asr_row(
    show_id: str,
    whisper: list[WordTiming],
    whistle: list[WordTiming],
    cache: Path,
) -> dict:
    ref = _tokens(whisper)
    hyp = _tokens(whistle)
    payload = load_word_payload(cache) or {}
    holes_w = word_holes(whisper)
    holes_h = word_holes(whistle)
    mae = aligned_token_mae(whisper, whistle)
    return {
        "show_id": show_id,
        "n_whisper": len(ref),
        "n_whistle": len(hyp),
        "word_ratio": (len(hyp) / float(len(ref))) if ref else 0.0,
        "wer": _wer(ref, hyp),
        "n_whisper_holes": len(holes_w),
        "whisper_hole_s": round(sum(b - a for a, b in holes_w), 2),
        "n_whistle_holes": len(holes_h),
        "whistle_hole_s": round(sum(b - a for a, b in holes_h), 2),
        "timestamp_mae_s": mae["mae_s"],
        "timestamp_median_s": mae["median_abs_s"],
        "timestamp_p90_s": mae["p90_abs_s"],
        "n_time_matched": mae["n_matched"],
        "wall_s": payload.get("wall_s"),
        "cpu_s": payload.get("cpu_s"),
        "audio_s": payload.get("audio_s"),
        "n_truncated": payload.get("n_truncated"),
        "n_seam_gaps": len(payload.get("seam_gaps") or []),
        "seam_gaps": payload.get("seam_gaps") or [],
        "disagreements": disagreement_windows(whisper, whistle),
    }


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values) or 1)


def run_e2e(
    *,
    scored: list[str],
    downloads: dict,
    windows: list[dict],
    words_dir: Path,
    holdout: bool,
    variant: str,
    require_llm_cache: bool,
) -> dict:
    live_dir = HOLDOUT_LIVE if holdout else LIVE_ORIG
    transcripts_dir = (HOLDOUT / "baseline" / "transcripts") if holdout else (AUDIO / "transcripts")
    out = OUT_ROOT / variant / ("holdout" if holdout else "original10")
    out.mkdir(parents=True, exist_ok=True)
    opening_need = None if holdout else "7/7"
    config = production_config(
        enable_quote_anchor_boundaries=True,
        enable_spot_then_confirm=True,
        enable_whistle_stt=True,
    )
    app = make_flask(out / "scorecard.sqlite")
    force_local_writer()
    tracker = SpendTracker()
    budget = GeminiBudget(cap_usd=UTC_CAP, prior_usd=0.0, ledger=LEDGER)
    transcriber = LocalWhisperTranscriber(logging.getLogger("pipeline"), "base")
    production: dict[str, dict] = {}
    tokens = TokenTotals()
    words_by_show = {sid: _as_words(words_dir / f"{sid}.json") for sid in scored}
    ngrams = repeated_ngrams(words_by_show, n=6, min_episodes=2)
    tracker.install()
    try:
        with app.app_context():
            for show_id in scored:
                audio = audio_for(show_id, downloads)
                remaining = budget.remaining - tracker.usd
                if remaining < 0.02 and not require_llm_cache:
                    logging.warning("UTC cap before %s remaining $%.3f", show_id, remaining)
                    break
                usd_before = tracker.usd
                payload = run_quote_anchor_production_cut(
                    show_id=show_id,
                    title=str(downloads[show_id].get("resolved_title") or show_id),
                    audio_path=audio,
                    output_dir=out,
                    config=config,
                    app=app,
                    transcriber=transcriber,
                    tracker=tracker,
                    words_cache=words_dir / f"{show_id}.json",
                    llm_cache_dir=out / "llm_cache",
                    skip_audio_clip=True,
                    transcripts_dir=transcripts_dir,
                    require_llm_cache=require_llm_cache,
                    ngram_phrases=ngrams,
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
                    "e2e %s calls=%s cuts=%s spend=$%.4f",
                    show_id,
                    payload.get("n_llm_calls"),
                    len(payload.get("cut_segments_s") or []),
                    tracker.usd,
                )
    finally:
        tracker.restore()
    live = live_cuts(live_dir)
    cand_cuts = {
        sid: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        for sid, row in production.items()
    }
    live_scored = score_windows(
        [row for row in windows if not row.get("skipped") and row.get("show_id") in live],
        live,
    )
    cand_scored = score_windows(
        [row for row in windows if not row.get("skipped") and row.get("show_id") in cand_cuts],
        cand_cuts,
    )
    n = max(1, len(production))
    summary = {
        "gold_family": "general_podcast_ads",
        "claim": CLAIM_LOCKED,
        "scout": False,
        "spot_then_confirm": True,
        "whistle": True,
        "variant": variant,
        "holdout": holdout,
        "gemini_usd": tracker.usd,
        "live": _metrics(live_scored.totals, n_shows=len(live), opening_need=opening_need),
        "candidate": _metrics(
            cand_scored.totals, n_shows=len(production), opening_need=opening_need
        ),
        "calls_per_episode": tokens.calls / n,
        "input_tokens_per_episode": tokens.input_tokens / n,
        "output_tokens_per_episode": tokens.output_tokens / n,
        "n_episodes": n,
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
    return summary


def main() -> int:
    args = parse_args()
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM or DEFAULTS.ENABLE_WHISTLE_STT:
        print("scout/whistle code defaults must stay false", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ["PODLY_WRITER_LOCAL_FALLBACK"] = "1"
    _load_keys()

    scored, downloads, whisper_dir, _labels_path, windows = scored_shows(
        bool(args.holdout), list(args.show)
    )
    variants: list[str] = []
    if args.plain and not args.keywords:
        variants = ["plain"]
    elif args.keywords and not args.plain:
        variants = ["keywords"]
    else:
        variants = ["plain", "keywords"]

    keys = sponsor_keywords()
    asr_out: dict[str, dict] = {}
    spotter_out: dict[str, dict] = {}
    for variant in variants:
        vdir = OUT_ROOT / variant / ("holdout" if args.holdout else "original10")
        wdir = vdir / "words"
        wdir.mkdir(parents=True, exist_ok=True)
        words_by_show: dict[str, list[WordTiming]] = {}
        rows: list[dict] = []
        for sid in scored:
            audio = audio_for(sid, downloads)
            cache = wdir / f"{sid}.json"
            kws = keys if variant == "keywords" else None
            started = time.time()
            whistle = ensure_whistle(sid, audio, cache, keywords=kws)
            logging.info("ready %s/%s %s in %.1fs", variant, sid, len(whistle), time.time() - started)
            whisper = _as_words(whisper_dir / f"{sid}.json")
            words_by_show[sid] = whistle
            row = asr_row(sid, whisper, whistle, cache)
            for disc in row["disagreements"][:1]:
                dest = vdir / "listen" / f"{sid}_{int(disc['start'])}-{int(disc['end'])}.wav"
                if not dest.is_file() and audio.is_file():
                    try:
                        extract_chunk_wav(audio, dest, float(disc["start"]), float(disc["end"]))
                        disc["clip"] = str(dest)
                    except Exception:  # noqa: BLE001
                        logging.exception("clip extract failed %s", sid)
            rows.append(row)
        spotter_out[variant] = spotter_for(scored, words_by_show, windows)
        audio_hours = sum(float(r.get("audio_s") or 0) for r in rows) / 3600.0
        wall = sum(float(r.get("wall_s") or 0) for r in rows)
        cpu = sum(float(r.get("cpu_s") or 0) for r in rows)
        asr_out[variant] = {
            "n_shows": len(rows),
            "wer_mean": mean([float(r["wer"]) for r in rows]),
            "word_ratio_mean": mean([float(r["word_ratio"]) for r in rows]),
            "timestamp_mae_mean": mean([float(r["timestamp_mae_s"]) for r in rows]),
            "timestamp_median_mean": mean([float(r["timestamp_median_s"]) for r in rows]),
            "n_truncated": sum(int(r.get("n_truncated") or 0) for r in rows),
            "n_seam_gaps": sum(int(r.get("n_seam_gaps") or 0) for r in rows),
            "whisper_hole_s": sum(float(r["whisper_hole_s"]) for r in rows),
            "whistle_hole_s": sum(float(r["whistle_hole_s"]) for r in rows),
            "wall_s": wall,
            "cpu_s": cpu,
            "audio_s": audio_hours * 3600.0,
            "wall_per_episode_hour": (wall / audio_hours) if audio_hours else 0.0,
            "cpu_per_episode_hour": (cpu / audio_hours) if audio_hours else 0.0,
            "spotter": spotter_out[variant],
            "shows": rows,
        }
        (vdir / "asr.json").write_text(
            json.dumps(asr_out[variant], indent=2) + "\n", encoding="utf-8"
        )
        logging.info(
            "%s wer=%.3f ratio=%.3f mae=%.3f trunc=%s seams=%s spotter=%s/%s wall/eh=%.1fs",
            variant,
            asr_out[variant]["wer_mean"],
            asr_out[variant]["word_ratio_mean"],
            asr_out[variant]["timestamp_mae_mean"],
            asr_out[variant]["n_truncated"],
            asr_out[variant]["n_seam_gaps"],
            spotter_out[variant]["n_gold_ads"] - spotter_out[variant]["n_missed"],
            spotter_out[variant]["n_gold_ads"],
            asr_out[variant]["wall_per_episode_hour"],
        )

    e2e = None
    if args.e2e:
        pick = "keywords"
        if spotter_out.get("keywords") and spotter_out.get("plain"):
            if spotter_out["plain"]["recall"] > spotter_out["keywords"]["recall"] + 1e-9:
                pick = "plain"
        elif "plain" in asr_out and "keywords" not in asr_out:
            pick = "plain"
        wdir = OUT_ROOT / pick / ("holdout" if args.holdout else "original10") / "words"
        e2e = run_e2e(
            scored=scored,
            downloads=downloads,
            windows=windows,
            words_dir=wdir,
            holdout=bool(args.holdout),
            variant=pick,
            require_llm_cache=bool(args.require_llm_cache),
        )

    summary = {
        "holdout": bool(args.holdout),
        "shows": scored,
        "asr": {k: {kk: vv for kk, vv in v.items() if kk != "shows"} for k, v in asr_out.items()},
        "spotter": spotter_out,
        "e2e": e2e,
    }
    dest = OUT_ROOT / ("holdout" if args.holdout else "original10")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
