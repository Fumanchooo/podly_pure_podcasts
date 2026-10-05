"""Eval quote-anchored cuts vs frozen auto-gold labels. Scout off. Local Whisper."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm

from podcast_processor.boundary_snap import (
    detect_silence_gaps,
    parse_silencedetect,
    refine_cuts,
)
from podcast_processor.experiments.auto_gold.constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_JUDGE_BUDGET_USD,
    GOLD_FAMILY,
    JUDGE_GEMINI_INPUT_USD_PER_M,
    JUDGE_GEMINI_OUTPUT_USD_PER_M,
)
from podcast_processor.experiments.auto_gold.scorecard import (
    CLAIM_LOCKED,
    ScorecardResult,
    render_scorecard_md,
    score_windows,
)
from podcast_processor.experiments.budget import (
    BudgetExceeded,
    DailyBudget,
    load_budget,
    write_budget,
)
from podcast_processor.experiments.cost_model import ModelPrices, usd_for_tokens
from podcast_processor.experiments.gemini_confirm import estimate_tokens_from_text
from podcast_processor.experiments.word_transcripts import (
    load_word_payload,
    segments_from_payload,
    transcribe_word_timestamps,
    words_from_payload,
)
from podcast_processor.quote_anchor import (
    QuoteAd,
    QuoteMatch,
    SegmentSpan,
    WordTiming,
    cuts_from_matches,
    hybrid_cuts_from_matches,
    match_rate,
    resolve_ads,
    word_index_range,
)
from podcast_processor.quote_anchor_prompt import (
    QUOTE_SYSTEM_PROMPT,
    quote_user_prompt,
    system_prompt_for,
)
from shared import defaults as DEFAULTS
from shared.defaults import PROCESSING_NUM_SEGMENTS_TO_INPUT_TO_PROMPT

logger = logging.getLogger(__name__)

UTC_DAY_CAP_USD = DEFAULT_JUDGE_BUDGET_USD
PRIOR_UTC_SPEND_USD = 0.0027  # policy re-judge earlier today
CHUNK_SEGMENTS = PROCESSING_NUM_SEGMENTS_TO_INPUT_TO_PROMPT
MIN_CONFIDENCE = DEFAULTS.OUTPUT_MIN_CONFIDENCE
QUOTE_PRICES = ModelPrices(
    name=DEFAULT_GEMINI_MODEL,
    input_usd_per_million=JUDGE_GEMINI_INPUT_USD_PER_M,
    output_usd_per_million=JUDGE_GEMINI_OUTPUT_USD_PER_M,
)
DEFAULT_SILENCE_CACHE = Path(
    "docs/experiments/auto_gold/runs/boundary-snap-eval/silence_cache"
)


@dataclass
class TokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    n_episodes: int = 0

    def per_episode(self, n: int | None = None) -> tuple[float, float]:
        denom = float(n if n is not None else self.n_episodes or 1)
        return self.input_tokens / denom, self.output_tokens / denom


@dataclass
class QuoteEvalResult:
    scorecard: ScorecardResult
    match_rate: float
    n_quotes: int
    n_word: int
    n_segment_fallback: int
    n_chunk_fallback: int
    n_dropped: int
    tokens: TokenTotals
    baseline_tokens: TokenTotals
    spend_usd: float
    utc_spend_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_tokens_from_sqlite(sqlite_path: Path) -> TokenTotals:
    totals = TokenTotals()
    if not sqlite_path.is_file():
        return totals
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    posts = {row[0]: str(row[1]).split(":")[-1] for row in con.execute("select id, guid from post")}
    per_show: dict[str, TokenTotals] = {}
    for post_id, prompt, response in con.execute(
        "select post_id, prompt, response from model_call where status='success'"
    ):
        head = (response or "")[:240]
        if "refined_start" in head or '"ad_segments"' not in (response or "") and "'ad_segments'" not in (response or ""):
            if "refined_start" in head:
                continue
            if "ad_segments" not in (response or ""):
                continue
        show = posts.get(int(post_id), str(post_id))
        bucket = per_show.setdefault(show, TokenTotals())
        pin = estimate_tokens_from_text(prompt or "")
        pout = estimate_tokens_from_text(response or "")
        bucket.input_tokens += pin
        bucket.output_tokens += pout
        bucket.calls += 1
        totals.input_tokens += pin
        totals.output_tokens += pout
        totals.calls += 1
    totals.n_episodes = len(per_show)
    con.close()
    return totals


def parse_quote_ads(raw: str) -> list[QuoteAd]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return []
    blob = text[start : end + 1]
    payload: Any
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        try:
            payload = json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return []
    rows = payload.get("ads") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    ads: list[QuoteAd] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start_q = str(row.get("start_quote") or "").strip()
        end_q = str(row.get("end_quote") or start_q).strip()
        if not start_q:
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        ads.append(
            QuoteAd(
                start_quote=start_q,
                end_quote=end_q or start_q,
                content_type=str(row.get("content_type") or "promotional_external"),
                confidence=confidence,
            )
        )
    return ads


def keep_ad(ad: QuoteAd) -> bool:
    conf = ad.confidence
    if ad.content_type == "technical_discussion":
        conf = max(0.0, conf - 0.25)
    elif ad.content_type == "transition":
        conf = max(0.0, conf - 0.1)
    return conf >= MIN_CONFIDENCE


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha1(f"{system}\n---\n{user}".encode()).hexdigest()


class GeminiBudget:
    """UTC-day Gemini cap via DailyBudget. Cache hits are $0."""

    def __init__(self, *, cap_usd: float, prior_usd: float, ledger: Path) -> None:
        self.cap_usd = cap_usd
        self.prior_usd = prior_usd
        self.session_usd = 0.0
        self.ledger = ledger
        self.blocked = False
        today = datetime.now(UTC).date().isoformat()
        daily = load_budget(ledger, limit_usd=cap_usd)
        if daily.date != today:
            daily = DailyBudget(limit_usd=cap_usd, date=today)
        elif daily.spent_usd + 1e-12 < prior_usd:
            daily.spent_usd = prior_usd
        self.daily = daily
        write_budget(self.daily, self.ledger)

    @property
    def remaining(self) -> float:
        return self.daily.remaining_usd

    def charge(self, usd: float, *, input_tokens: int, output_tokens: int) -> None:
        self.session_usd += usd
        self.daily.record(
            usd,
            source="quote-anchor",
            cached=False,
            provider="gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        write_budget(self.daily, self.ledger)

    def record_cache(self, *, input_tokens: int, output_tokens: int) -> None:
        self.daily.record(
            0.0,
            source="quote-anchor",
            cached=True,
            provider="gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        write_budget(self.daily, self.ledger)

    def allow(self, estimated_usd: float) -> bool:
        if self.daily.can_afford(estimated_usd):
            return True
        self.blocked = True
        return False


def complete_json(
    *,
    user: str,
    cache_dir: Path,
    budget: GeminiBudget,
    tokens: TokenTotals,
    model: str,
    system_prompt: str = QUOTE_SYSTEM_PROMPT,
) -> str:
    cache_path = cache_dir / f"{_prompt_hash(system_prompt, user)}.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        pin = int(payload.get("input_tokens") or 0)
        pout = int(payload.get("output_tokens") or 0)
        tokens.input_tokens += pin
        tokens.output_tokens += pout
        tokens.calls += 1
        budget.record_cache(input_tokens=pin, output_tokens=pout)
        return str(payload.get("content") or '{"ads":[]}')
    est_in = estimate_tokens_from_text(system_prompt + user)
    est_usd = usd_for_tokens(est_in, 400, QUOTE_PRICES)
    if not budget.allow(est_usd):
        raise BudgetExceeded(
            f"Gemini UTC-day cap would be exceeded (remaining ${budget.remaining:.4f})"
        )
    last_error: Exception | None = None
    response = None
    for attempt in range(6):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                timeout=300,
                response_format={"type": "json_object"},
                max_tokens=512,
                reasoning_effort="low",
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            text = str(exc).lower()
            retryable = (
                "503" in text
                or "unavailable" in text
                or "429" in text
                or "rate limit" in text
                or "internal" in text
            )
            if not retryable or attempt == 5:
                raise
            delay = min(32.0, 4.0 * (2 ** attempt))
            logger.warning("Gemini retry %s after %.0fs: %s", attempt + 1, delay, type(exc).__name__)
            time.sleep(delay)
    if response is None:
        raise last_error or RuntimeError("Gemini call failed")
    content = response.choices[0].message.content or '{"ads":[]}'
    usage = getattr(response, "usage", None)
    pin = int(getattr(usage, "prompt_tokens", 0) or est_in)
    pout = int(getattr(usage, "completion_tokens", 0) or estimate_tokens_from_text(content))
    usd = usd_for_tokens(pin, pout, QUOTE_PRICES)
    budget.charge(usd, input_tokens=pin, output_tokens=pout)
    tokens.input_tokens += pin
    tokens.output_tokens += pout
    tokens.usd += usd
    tokens.calls += 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"content": content, "input_tokens": pin, "output_tokens": pout, "usd": usd},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return content


def _segments_from_transcript(path: Path) -> list[SegmentSpan]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        SegmentSpan(start=float(row["start"]), end=float(row["end"]), text=str(row.get("text") or ""))
        for row in rows
    ]


def load_silence_gaps(show_id: str, audio_path: Path, cache_dir: Path | None) -> list:
    if cache_dir is not None:
        log_path = cache_dir / f"{show_id}.silencedetect.log"
        if log_path.is_file():
            return parse_silencedetect(log_path.read_text(encoding="utf-8", errors="replace"))
    if audio_path.is_file():
        return detect_silence_gaps(audio_path)
    return []


def _segments_for_show(
    *,
    word_cache: Path,
    transcripts_dir: Path,
    show_id: str,
) -> list[SegmentSpan]:
    payload = load_word_payload(word_cache)
    if payload:
        rows = segments_from_payload(payload)
        if rows:
            return [
                SegmentSpan(start=a, end=b, text=text) for a, b, text in rows if b > a
            ]
    return _segments_from_transcript(transcripts_dir / f"{show_id}.json")


def quote_cuts_for_show(
    *,
    show_id: str,
    title: str,
    segments: list[SegmentSpan],
    words: list[WordTiming],
    audio_path: Path | None,
    cache_dir: Path,
    budget: GeminiBudget,
    tokens: TokenTotals,
    model: str,
    silence_cache_dir: Path | None = None,
    prompt_version: str = "v1",
    cut_mode: str = "pure",
    baseline_cuts: list[tuple[float, float]] | None = None,
    pad_seconds: float = 0.0,
) -> tuple[list[tuple[float, float]], list[QuoteMatch]]:
    system_prompt = system_prompt_for(prompt_version)
    matches: list[QuoteMatch] = []
    idx = 0
    while idx < len(segments):
        chunk = segments[idx : idx + CHUNK_SEGMENTS]
        chunk_start = chunk[0].start
        chunk_end = chunk[-1].end
        user = quote_user_prompt(
            title=title,
            start_seconds=chunk_start,
            texts=[seg.text for seg in chunk],
            prompt_version=prompt_version,
        )
        raw = complete_json(
            user=user,
            cache_dir=cache_dir,
            budget=budget,
            tokens=tokens,
            model=model,
            system_prompt=system_prompt,
        )
        ads = [ad for ad in parse_quote_ads(raw) if keep_ad(ad)]
        w0, w1 = word_index_range(words, max(0.0, chunk_start - 1.0), chunk_end + 1.0)
        matches.extend(
            resolve_ads(
                ads,
                words,
                segments,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                word_from=w0,
                word_to=w1,
            )
        )
        idx += len(chunk)
    gaps = []
    if audio_path is not None and audio_path.is_file():
        gaps = [
            (g.start, g.end)
            for g in load_silence_gaps(show_id, audio_path, silence_cache_dir)
        ]
    if cut_mode == "hybrid":
        cuts = hybrid_cuts_from_matches(
            matches,
            gaps,
            baseline=baseline_cuts,
            pad_seconds=pad_seconds,
        )
    else:
        cuts = cuts_from_matches(matches)
        if audio_path is not None and audio_path.is_file():
            from podcast_processor.boundary_snap import SilenceGap

            typed_gaps = [SilenceGap(start=a, end=b) for a, b in gaps]
            cuts = refine_cuts(
                cuts, audio_path=audio_path, gaps=typed_gaps, radius=2.0
            )
    tokens.n_episodes += 1
    return cuts, matches


def run_quote_anchor_eval(
    *,
    repo_root: Path,
    labels_path: Path,
    transcripts_dir: Path,
    audio_dir: Path,
    sqlite_path: Path,
    words_dir: Path,
    output_dir: Path,
    model: str | None = None,
    whisper_model: str = "base",
    prior_usd: float = PRIOR_UTC_SPEND_USD,
    silence_cache_dir: Path | None = None,
    prompt_version: str = "v1",
    cut_mode: str = "pure",
    skip_transcribe: bool = False,
    baseline_cuts: dict[str, list[tuple[float, float]]] | None = None,
    pad_seconds: float = 0.0,
    llm_cache_dir: Path | None = None,
    budget_ledger: Path | None = None,
) -> QuoteEvalResult:
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM:
        raise RuntimeError("enable_bow_scout_gemini_confirm must stay false")
    labels_payload = load_json(labels_path)
    labeled = list(labels_payload.get("windows") or [])
    downloads = load_json(audio_dir / "downloads.json")
    scored = [
        sid
        for sid, row in downloads.items()
        if row.get("audio_path") and sid not in {"bill_simmons", "marketplace"}
        and (transcripts_dir / f"{sid}.json").is_file()
    ]

    def _audio_size(show_id: str) -> int:
        audio = Path(str(downloads[show_id]["audio_path"]))
        if not audio.is_absolute():
            audio = repo_root / audio
        return audio.stat().st_size if audio.is_file() else 1 << 62

    scored.sort(key=_audio_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    if silence_cache_dir is None:
        silence_cache_dir = repo_root / DEFAULT_SILENCE_CACHE
    budget = GeminiBudget(
        cap_usd=UTC_DAY_CAP_USD,
        prior_usd=prior_usd,
        ledger=budget_ledger or (output_dir / "gemini_budget.json"),
    )
    llm_cache = llm_cache_dir or (output_dir / "llm_cache")
    tokens = TokenTotals()
    model_name = model or DEFAULT_GEMINI_MODEL
    production: dict[str, dict[str, Any]] = {}
    all_matches: list[QuoteMatch] = []
    asr_model: Any = None
    notes: list[str] = [
        f"Quote-anchor eval prompt={prompt_version} cut_mode={cut_mode}. "
        "Scout confirm off. Self-promo counts as ads. general_podcast_ads LOCKED."
    ]
    for show_id in scored:
        audio_raw = downloads[show_id]["audio_path"]
        audio_path = Path(str(audio_raw))
        if not audio_path.is_absolute():
            audio_path = repo_root / audio_path
        word_cache = words_dir / f"{show_id}.json"
        payload = load_word_payload(word_cache)
        words: list[WordTiming] = []
        if payload and payload.get("word_timestamps") and payload.get("words"):
            words = words_from_payload(payload)
        if not words:
            if skip_transcribe:
                notes.append(f"skip {show_id}: no word-timestamp cache")
                continue
            logger.info("transcribing words for %s", show_id)
            if asr_model is None:
                import whisper

                asr_model = whisper.load_model(name=whisper_model)
            words = transcribe_word_timestamps(
                audio_path,
                word_cache,
                model_name=whisper_model,
                model=asr_model,
            )
        segments = _segments_for_show(
            word_cache=word_cache,
            transcripts_dir=transcripts_dir,
            show_id=show_id,
        )
        title = str(downloads[show_id].get("resolved_title") or show_id)
        try:
            cuts, matches = quote_cuts_for_show(
                show_id=show_id,
                title=title,
                segments=segments,
                words=words,
                audio_path=audio_path,
                cache_dir=llm_cache,
                budget=budget,
                tokens=tokens,
                model=model_name,
                silence_cache_dir=silence_cache_dir,
                prompt_version=prompt_version,
                cut_mode=cut_mode,
                baseline_cuts=(baseline_cuts or {}).get(show_id),
                pad_seconds=pad_seconds,
            )
        except BudgetExceeded as exc:
            budget.blocked = True
            notes.append(f"Hit UTC-day Gemini cap before finishing {show_id}: {exc}")
            logger.warning("%s", notes[-1])
            break
        dropped_here = sum(1 for item in matches if item.dropped)
        all_matches.extend(matches)
        production[show_id] = {
            "show_id": show_id,
            "cut_segments_s": [[a, b] for a, b in cuts],
            "n_quotes": len(matches),
            "n_word": sum(1 for item in matches if item.method == "word"),
            "n_segment_fallback": sum(
                1 for item in matches if item.method == "segment_fallback"
            ),
            "n_chunk_fallback": sum(
                1 for item in matches if item.method == "chunk_fallback"
            ),
            "n_dropped": dropped_here,
            "match_rate": match_rate(matches),
            "prompt_version": prompt_version,
            "cut_mode": cut_mode,
            "quotes": [
                {
                    "start": item.start,
                    "end": item.end,
                    "method": item.method,
                    "score": item.score,
                    "start_quote": item.ad.start_quote,
                    "end_quote": item.ad.end_quote,
                    "bound_start": item.bound_start,
                    "bound_end": item.bound_end,
                }
                for item in matches
            ],
        }
        (output_dir / "production").mkdir(parents=True, exist_ok=True)
        (output_dir / "production" / f"{show_id}.json").write_text(
            json.dumps(production[show_id], indent=2) + "\n", encoding="utf-8"
        )
    result = score_windows(
        [row for row in labeled if not row.get("skipped") and row.get("show_id") in production],
        {
            sid: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
            for sid, row in production.items()
        },
    )
    result.spend_usd = budget.session_usd
    result.spend_cap_usd = UTC_DAY_CAP_USD
    result.n_shows_catalog = 12
    result.n_shows_scored = len(production)
    result.notes = notes
    n_word = sum(1 for m in all_matches if m.method == "word")
    n_seg = sum(1 for m in all_matches if m.method == "segment_fallback")
    n_chunk = sum(1 for m in all_matches if m.method == "chunk_fallback")
    dropped = sum(1 for m in all_matches if m.dropped)
    eval_result = QuoteEvalResult(
        scorecard=result,
        match_rate=match_rate(all_matches),
        n_quotes=len(all_matches),
        n_word=n_word,
        n_segment_fallback=n_seg,
        n_chunk_fallback=n_chunk,
        n_dropped=dropped,
        tokens=tokens,
        baseline_tokens=baseline_tokens_from_sqlite(sqlite_path),
        spend_usd=budget.session_usd,
        utc_spend_usd=budget.daily.spent_usd,
        notes=notes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SCORECARD.md").write_text(render_scorecard_md(result), encoding="utf-8")
    pin, pout = tokens.per_episode(max(1, len(production)))
    bpin, bpout = eval_result.baseline_tokens.per_episode()
    summary = {
        "gold_family": GOLD_FAMILY,
        "n_shows": len(production),
        "match_rate": eval_result.match_rate,
        "n_quotes": eval_result.n_quotes,
        "n_word": n_word,
        "n_segment_fallback": n_seg,
        "n_chunk_fallback": n_chunk,
        "n_dropped": dropped,
        "inner_span_catch": result.totals.span_seconds_recall(),
        "opening_inner": f"{result.totals.n_opening_span_caught}/{result.totals.n_opening_windows}",
        "not_ad_cut": result.totals.not_ad_seconds_cut,
        "overcut": result.totals.overcut_seconds,
        "undercut": result.totals.undercut_seconds,
        "edge_error": result.totals.overcut_seconds + result.totals.undercut_seconds,
        "start_error_median": result.totals.start_error_median,
        "start_error_p90": result.totals.start_error_p90,
        "end_error_median": result.totals.end_error_median,
        "end_error_p90": result.totals.end_error_p90,
        "input_tokens_per_episode": pin,
        "output_tokens_per_episode": pout,
        "baseline_input_tokens_per_episode": bpin,
        "baseline_output_tokens_per_episode": bpout,
        "spend_usd": budget.session_usd,
        "utc_spend_usd": budget.daily.spent_usd,
        "prior_usd": prior_usd,
        "claim": CLAIM_LOCKED,
        "scout": False,
    }
    (output_dir / "score.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return eval_result
