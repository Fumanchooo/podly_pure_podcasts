"""Production quote-anchor classifier (prompt_v2 + lexical_v2).

Word-level Whisper (local CPU) → timestamp-free quote request → in-order
fuzzy match → lexical_v2 → Identification rows + stored cuts. AudioProcessor
then applies the existing ±2s silence snap. Fail-closed: never drop a
reported ad (segment-edge, then chunk span). Scout confirm stays off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm

from app.models import ModelCall, Post, TranscriptSegment
from app.writer.client import writer_client
from podcast_processor.boundary_snap import detect_silence_gaps, interpolate_word_times
from podcast_processor.experiments.word_transcripts import (
    transcribe_word_timestamps,
    words_from_payload,
)
from podcast_processor.quote_anchor import (
    QUOTE_ANCHOR_SOURCE,
    QuoteMatch,
    SegmentSpan,
    WordTiming,
    keep_ad,
    parse_quote_ads,
    resolve_ads,
    word_index_range,
)
from podcast_processor.quote_anchor_prompt import quote_user_prompt, system_prompt_for
from podcast_processor.quote_lexicon import lexical_cuts_from_matches
from shared import defaults as DEFAULTS
from shared.config import Config, TestWhisperConfig

logger = logging.getLogger("global_logger")

QUOTE_PROMPT_VERSION = "v2"
QUOTE_CHUNK_SEGMENTS = DEFAULTS.PROCESSING_NUM_SEGMENTS_TO_INPUT_TO_PROMPT
CompleteFn = Callable[[str, str], str]


def quote_anchor_boundaries_enabled(config: Config) -> bool:
    return bool(getattr(config, "enable_quote_anchor_boundaries", False))


def word_cache_path_for_audio(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    return path.with_name(path.name + ".words.json")


def interpolate_words_from_segments(
    segments: list[TranscriptSegment],
) -> list[WordTiming]:
    words: list[WordTiming] = []
    for segment in segments:
        for span in interpolate_word_times(
            float(segment.start_time or 0.0),
            float(segment.end_time or segment.start_time or 0.0),
            str(segment.text or ""),
        ):
            words.append(
                WordTiming(word=span.word, start=span.start, end=span.end)
            )
    return words


def segments_to_spans(segments: list[TranscriptSegment]) -> list[SegmentSpan]:
    return [
        SegmentSpan(
            start=float(segment.start_time or 0.0),
            end=float(segment.end_time or 0.0),
            text=str(segment.text or ""),
        )
        for segment in segments
    ]


class QuoteAnchorClassifier:
    """Identify ads by quotes, then cut with lexical_v2. Fail-closed."""

    def __init__(
        self,
        config: Config,
        logger: logging.Logger | None = None,
        db_session: Any | None = None,
        *,
        complete_fn: CompleteFn | None = None,
        word_cache_path: Path | None = None,
        llm_cache_dir: Path | None = None,
        whisper_model: str = "base",
        require_llm_cache: bool = False,
        pull_dai_bed: bool = False,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("global_logger")
        self.db_session = db_session
        self.complete_fn = complete_fn
        self.word_cache_path = word_cache_path
        self.llm_cache_dir = llm_cache_dir
        self.whisper_model = whisper_model
        self.require_llm_cache = require_llm_cache
        self.pull_dai_bed = pull_dai_bed
        self.last_matches: list[QuoteMatch] = []
        self.last_cuts: list[tuple[float, float]] = []
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_word_whisper_s = 0.0
        self.last_classify_s = 0.0
        self.silence_gaps_override: list[tuple[float, float]] | None = None

    def classify(
        self,
        *,
        transcript_segments: list[TranscriptSegment],
        post: Post,
        words: list[WordTiming] | None = None,
    ) -> list[tuple[float, float]]:
        started = time.time()
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        if not transcript_segments:
            self.last_matches = []
            self.last_cuts = []
            self._persist_cuts(post, [], [])
            self.last_classify_s = time.time() - started
            return []

        loaded = words if words is not None else self.load_words(post, transcript_segments)
        spans = segments_to_spans(transcript_segments)
        matches: list[QuoteMatch] = []
        idx = 0
        while idx < len(transcript_segments):
            chunk = transcript_segments[idx : idx + QUOTE_CHUNK_SEGMENTS]
            chunk_spans = spans[idx : idx + QUOTE_CHUNK_SEGMENTS]
            chunk_start = chunk_spans[0].start
            chunk_end = chunk_spans[-1].end
            user = quote_user_prompt(
                title=str(post.title or ""),
                start_seconds=chunk_start,
                texts=[seg.text or "" for seg in chunk],
                prompt_version=QUOTE_PROMPT_VERSION,
            )
            raw, model_call = self._complete_chunk(
                post=post,
                chunk=chunk,
                user=user,
            )
            ads = [
                ad
                for ad in parse_quote_ads(raw)
                if keep_ad(ad, float(self.config.output.min_confidence))
            ]
            w0, w1 = word_index_range(
                loaded, max(0.0, chunk_start - 1.0), chunk_end + 1.0
            )
            chunk_matches = resolve_ads(
                ads,
                loaded,
                spans,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                word_from=w0,
                word_to=w1,
            )
            if any(item.dropped for item in chunk_matches):
                raise RuntimeError("quote-anchor dropped an ad; fail-closed violated")
            if model_call is not None:
                self._write_identifications(chunk, chunk_matches, model_call)
            matches.extend(chunk_matches)
            idx += len(chunk)

        gaps = self._silence_gaps(post)
        cuts = lexical_cuts_from_matches(
            matches, loaded, gaps, pull_dai_bed=self.pull_dai_bed
        )
        # Fail-closed: a match that lexical collapsed still keeps its raw span.
        if len(cuts) < len(matches):
            used = {(round(a, 3), round(b, 3)) for a, b in cuts}
            for item in matches:
                key = (round(item.start, 3), round(item.end, 3))
                if item.end > item.start and key not in used:
                    cuts.append((item.start, item.end))
                    self.logger.warning(
                        "lexical dropped a quote span; restoring %.1f-%.1f",
                        item.start,
                        item.end,
                    )
        self.last_matches = matches
        self.last_cuts = cuts
        self._persist_cuts(post, cuts, matches)
        self.last_classify_s = time.time() - started
        self.logger.info(
            "quote-anchor classified post %s quotes=%s word=%s seg_fb=%s chunk_fb=%s cuts=%s",
            post.id,
            len(matches),
            sum(1 for item in matches if item.method == "word"),
            sum(1 for item in matches if item.method == "segment_fallback"),
            sum(1 for item in matches if item.method == "chunk_fallback"),
            len(cuts),
        )
        return cuts

    def load_words(
        self,
        post: Post,
        transcript_segments: list[TranscriptSegment],
    ) -> list[WordTiming]:
        if isinstance(self.config.whisper, TestWhisperConfig):
            return interpolate_words_from_segments(transcript_segments)
        audio = post.unprocessed_audio_path
        cache = self.word_cache_path
        if cache is None and audio:
            cache = word_cache_path_for_audio(audio)
        if cache is not None and cache.is_file():
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                words = words_from_payload(payload)
                if words:
                    return words
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                self.logger.warning("unreadable word cache %s", cache, exc_info=True)
        if audio and Path(str(audio)).is_file() and cache is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            started = time.time()
            try:
                words = transcribe_word_timestamps(
                    Path(str(audio)),
                    cache,
                    model_name=self.whisper_model,
                )
                self.last_word_whisper_s = time.time() - started
                if words:
                    return words
            except Exception:  # noqa: BLE001
                self.logger.exception(
                    "word-timestamp whisper failed for post %s; interpolating",
                    post.id,
                )
                self.last_word_whisper_s = time.time() - started
        self.logger.warning(
            "interpolating word times from segments for post %s", post.id
        )
        return interpolate_words_from_segments(transcript_segments)

    def _silence_gaps(self, post: Post) -> list[tuple[float, float]]:
        if self.silence_gaps_override is not None:
            return list(self.silence_gaps_override)
        audio = post.unprocessed_audio_path
        if not audio or not Path(str(audio)).is_file():
            return []
        try:
            return [(gap.start, gap.end) for gap in detect_silence_gaps(Path(str(audio)))]
        except Exception:  # noqa: BLE001
            self.logger.warning("silencedetect failed for post %s", post.id, exc_info=True)
            return []

    def _complete_chunk(
        self,
        *,
        post: Post,
        chunk: list[TranscriptSegment],
        user: str,
    ) -> tuple[str, ModelCall | None]:
        system_prompt = system_prompt_for(QUOTE_PROMPT_VERSION)
        model_call = self._upsert_model_call(
            post=post,
            first_seq_num=int(chunk[0].sequence_num),
            last_seq_num=int(chunk[-1].sequence_num),
            prompt=user,
        )
        if self.complete_fn is not None:
            content = self.complete_fn(system_prompt, user)
            self._mark_model_call(model_call, content, status="success")
            return content, model_call
        cached = self._read_llm_cache(system_prompt, user)
        if cached is not None:
            self.last_input_tokens += int(cached.get("input_tokens") or 0)
            self.last_output_tokens += int(cached.get("output_tokens") or 0)
            content = str(cached.get("content") or '{"ads":[]}')
            self._mark_model_call(model_call, content, status="success")
            return content, model_call
        if self.require_llm_cache:
            raise RuntimeError(
                "quote-anchor LLM cache miss; refusing live Gemini call"
            )
        if isinstance(self.config.whisper, TestWhisperConfig):
            content = '{"ads":[]}'
            self._mark_model_call(model_call, content, status="success")
            return content, model_call
        content, pin, pout = self._call_llm(system_prompt, user)
        self.last_input_tokens += pin
        self.last_output_tokens += pout
        self._write_llm_cache(system_prompt, user, content, pin, pout)
        self._mark_model_call(model_call, content, status="success")
        return content, model_call

    def _call_llm(self, system_prompt: str, user: str) -> tuple[str, int, int]:
        last_error: Exception | None = None
        retries = int(getattr(self.config, "llm_max_retry_attempts", 5) or 5)
        for attempt in range(retries):
            try:
                response = litellm.completion(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                    ],
                    timeout=int(getattr(self.config, "openai_timeout", 300) or 300),
                    response_format={"type": "json_object"},
                    max_tokens=512,
                    reasoning_effort="low",
                    api_key=self.config.llm_api_key,
                )
                content = response.choices[0].message.content or '{"ads":[]}'
                usage = getattr(response, "usage", None)
                pin = int(getattr(usage, "prompt_tokens", 0) or 0)
                pout = int(getattr(usage, "completion_tokens", 0) or 0)
                return content, pin, pout
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                text = str(exc).lower()
                retryable = (
                    "503" in text
                    or "unavailable" in text
                    or "429" in text
                    or "rate limit" in text
                    or "internal" in text
                    or "name resolution" in text
                    or "apiconnectionerror" in text
                    or "connecterror" in text
                    or "connect" in text
                )
                if not retryable or attempt >= retries - 1:
                    raise
                delay = min(32.0, 4.0 * (2**attempt))
                self.logger.warning(
                    "quote-anchor Gemini retry %s after %.0fs: %s",
                    attempt + 1,
                    delay,
                    type(exc).__name__,
                )
                time.sleep(delay)
        raise last_error or RuntimeError("quote-anchor LLM call failed")

    def _cache_key(self, system_prompt: str, user: str) -> str:
        return hashlib.sha1(f"{system_prompt}\n---\n{user}".encode()).hexdigest()

    def _read_llm_cache(self, system_prompt: str, user: str) -> dict[str, Any] | None:
        if self.llm_cache_dir is None:
            return None
        path = self.llm_cache_dir / f"{self._cache_key(system_prompt, user)}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_llm_cache(
        self,
        system_prompt: str,
        user: str,
        content: str,
        pin: int,
        pout: int,
    ) -> None:
        if self.llm_cache_dir is None:
            return
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.llm_cache_dir / f"{self._cache_key(system_prompt, user)}.json"
        path.write_text(
            json.dumps(
                {
                    "content": content,
                    "input_tokens": pin,
                    "output_tokens": pout,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _upsert_model_call(
        self,
        *,
        post: Post,
        first_seq_num: int,
        last_seq_num: int,
        prompt: str,
    ) -> ModelCall | None:
        if post.id is None:
            return None
        result = writer_client.action(
            "upsert_model_call",
            {
                "post_id": post.id,
                "model_name": self.config.llm_model,
                "first_segment_sequence_num": first_seq_num,
                "last_segment_sequence_num": last_seq_num,
                "prompt": prompt,
            },
            wait=True,
        )
        if not result or not result.success:
            self.logger.error(
                "quote-anchor failed to upsert ModelCall: %s",
                getattr(result, "error", None),
            )
            return None
        model_call_id = (result.data or {}).get("model_call_id")
        session = self.db_session
        if session is None or model_call_id is None:
            return None
        return session.get(ModelCall, int(model_call_id))

    def _mark_model_call(
        self,
        model_call: ModelCall | None,
        content: str,
        *,
        status: str,
    ) -> None:
        if model_call is None or model_call.id is None:
            return
        writer_client.update(
            "ModelCall",
            model_call.id,
            {
                "response": content,
                "status": status,
                "error_message": None,
                "retry_attempts": 1,
            },
            wait=True,
        )
        model_call.response = content
        model_call.status = status

    def _write_identifications(
        self,
        chunk: list[TranscriptSegment],
        matches: list[QuoteMatch],
        model_call: ModelCall,
    ) -> None:
        if not matches or model_call.id is None:
            return
        to_insert: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in matches:
            confidence = max(float(item.ad.confidence), float(item.score))
            for segment in chunk:
                if segment.id is None or segment.id in seen:
                    continue
                if min(item.end, float(segment.end_time)) <= max(
                    item.start, float(segment.start_time)
                ):
                    continue
                seen.add(segment.id)
                to_insert.append(
                    {
                        "transcript_segment_id": segment.id,
                        "model_call_id": model_call.id,
                        "label": "ad",
                        "confidence": confidence,
                    }
                )
        if not to_insert:
            return
        res = writer_client.action(
            "insert_identifications",
            {"identifications": to_insert},
            wait=True,
        )
        if not res or not res.success:
            raise RuntimeError(
                getattr(res, "error", "Failed to insert quote-anchor identifications")
            )

    def _persist_cuts(
        self,
        post: Post,
        cuts: list[tuple[float, float]],
        matches: list[QuoteMatch],
    ) -> None:
        if post.id is None:
            return
        payload: list[dict[str, Any]] = [
            {
                "source": QUOTE_ANCHOR_SOURCE,
                "empty": not cuts,
                "n_quotes": len(matches),
                "n_word": sum(1 for item in matches if item.method == "word"),
                "n_segment_fallback": sum(
                    1 for item in matches if item.method == "segment_fallback"
                ),
                "n_chunk_fallback": sum(
                    1 for item in matches if item.method == "chunk_fallback"
                ),
                "orig_start": 0.0,
                "orig_end": 0.0,
                "refined_start": 0.0,
                "refined_end": 0.0,
            }
        ]
        for cut, item in _zip_cuts_to_matches(cuts, matches):
            payload.append(
                {
                    "source": QUOTE_ANCHOR_SOURCE,
                    "orig_start": float(item.start if item is not None else cut[0]),
                    "orig_end": float(item.end if item is not None else cut[1]),
                    "refined_start": float(cut[0]),
                    "refined_end": float(cut[1]),
                    "method": item.method if item is not None else "lexical",
                    "score": float(item.score) if item is not None else 0.0,
                    "start_quote": item.ad.start_quote if item is not None else "",
                    "end_quote": item.ad.end_quote if item is not None else "",
                    "empty": False,
                }
            )
        res = writer_client.update(
            "Post",
            post.id,
            {
                "refined_ad_boundaries": payload,
                "refined_ad_boundaries_updated_at": datetime.now(UTC).replace(
                    tzinfo=None
                ),
            },
            wait=True,
        )
        if not res or not res.success:
            raise RuntimeError(
                getattr(res, "error", "Failed to persist quote-anchor cuts")
            )
        post.refined_ad_boundaries = payload


def _zip_cuts_to_matches(
    cuts: list[tuple[float, float]],
    matches: list[QuoteMatch],
) -> list[tuple[tuple[float, float], QuoteMatch | None]]:
    remaining = list(matches)
    paired: list[tuple[tuple[float, float], QuoteMatch | None]] = []
    for cut in cuts:
        best: QuoteMatch | None = None
        best_overlap = 0.0
        for item in remaining:
            overlap = max(0.0, min(cut[1], item.end) - max(cut[0], item.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = item
        if best is not None and best_overlap > 0.0:
            remaining.remove(best)
            paired.append((cut, best))
        else:
            paired.append((cut, None))
    return paired
