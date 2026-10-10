"""Run production AdClassifier+cut against frozen auto-gold windows.

Scout confirm stays off. Isolated sqlite (does not touch Cake's live DB).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
from flask import Flask
from jinja2 import Template

from app.extensions import db
from app.models import Feed, Identification, ModelCall, Post, TranscriptSegment
from app.writer.client import writer_client
from podcast_processor.ad_classifier import AdClassifier
from podcast_processor.audio import clip_segments_with_fade, get_audio_duration_ms
from podcast_processor.audio_processor import AudioProcessor
from podcast_processor.experiments.auto_gold.audio_ops import extract_chunk_wav
from podcast_processor.experiments.auto_gold.constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_JUDGE_BUDGET_USD,
    DEFAULT_WHISPER_MODEL,
    GOLD_FAMILY,
    JUDGE_GEMINI_INPUT_USD_PER_M,
    JUDGE_GEMINI_OUTPUT_USD_PER_M,
)
from podcast_processor.experiments.auto_gold.downloader import fetch_show_episode
from podcast_processor.experiments.auto_gold.judge import GoldJudge
from podcast_processor.experiments.auto_gold.scorecard import (
    PREROLL_END_SECONDS,
    ScorecardResult,
    fp_notes,
    opening_fn_notes,
    render_scorecard_md,
    score_windows,
)
from podcast_processor.experiments.auto_gold.shows import load_shows
from podcast_processor.experiments.auto_gold.types import CandidateChunk
from podcast_processor.experiments.auto_gold.whisper_chunks import ChunkWhisper
from podcast_processor.experiments.cost_model import ModelPrices, usd_for_tokens
from podcast_processor.prompt import generate_system_prompt
from podcast_processor.transcribe import LocalWhisperTranscriber, Segment
from shared import defaults as DEFAULTS
from shared.config import Config, LocalWhisperConfig, OutputConfig, ProcessingConfig
from shared.env import gemini_api_key

logger = logging.getLogger(__name__)


class CachedLocalWhisper(LocalWhisperTranscriber):
    """Load the Whisper weights once (production transcriber reloads every call)."""

    def __init__(self, logger: logging.Logger, whisper_model: str):
        super().__init__(logger, whisper_model)
        self._model: Any = None

    def transcribe(self, audio_file_path: str, language: str) -> list[Segment]:
        import whisper

        self.logger.info("Using cached local whisper model=%s", self.whisper_model)
        if self._model is None:
            self._model = whisper.load_model(name=self.whisper_model)
        started = time.time()
        result = self._model.transcribe(audio_file_path, fp16=False, language=language)
        self.logger.info("Transcription completed in %s", time.time() - started)
        typed_segments = self.convert_to_pydantic(list(result["segments"]))
        return self.local_seg_to_seg(typed_segments)


CAKE_ENV_LOCAL = Path("/home/jdr/cursor/repos/podly_pure_podcasts/.env.local")
GEMINI_PRICES = ModelPrices(
    name=DEFAULT_GEMINI_MODEL,
    input_usd_per_million=JUDGE_GEMINI_INPUT_USD_PER_M,
    output_usd_per_million=JUDGE_GEMINI_OUTPUT_USD_PER_M,
)


@dataclass
class CatalogEpisode:
    show_id: str
    genre: str
    ad_mechanism: str
    episode_title: str
    windows: list[dict[str, Any]]


class SpendTracker:
    def __init__(self) -> None:
        self.usd = 0.0
        self.calls = 0
        self._orig = litellm.completion

    def install(self) -> None:
        tracker = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            response = tracker._orig(*args, **kwargs)
            usage = getattr(response, "usage", None)
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            tracker.usd += usd_for_tokens(prompt, completion, GEMINI_PRICES)
            tracker.calls += 1
            return response

        litellm.completion = wrapped  # type: ignore[method-assign]

    def restore(self) -> None:
        litellm.completion = self._orig  # type: ignore[method-assign]


def load_env_file(path: Path) -> None:
    if not path.is_file():
        logger.warning("env file missing: %s", path)
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def assert_scout_off(config: Config) -> None:
    if DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM or config.enable_bow_scout_gemini_confirm:
        raise RuntimeError(
            "enable_bow_scout_gemini_confirm must stay false for this scorecard"
        )


def production_config(
    *,
    enable_quote_anchor_boundaries: bool = False,
    enable_spot_then_confirm: bool = False,
    enable_quote_full_sentence_quotes: bool = False,
    enable_spot_merge_calls: bool = False,
    enable_richer_spotter: bool = False,
    enable_quote_fallback_bound: bool = False,
    enable_quote_unique_anchors: bool = False,
) -> Config:
    llm_key = os.environ.get("LLM_API_KEY") or gemini_api_key()
    whisper_model = os.environ.get("WHISPER_LOCAL_MODEL") or "base"
    config = Config(
        llm_api_key=llm_key,
        llm_model=os.environ.get("LLM_MODEL") or DEFAULT_GEMINI_MODEL,
        enable_bow_scout_gemini_confirm=False,
        enable_quote_anchor_boundaries=enable_quote_anchor_boundaries,
        enable_spot_then_confirm=enable_spot_then_confirm,
        enable_quote_full_sentence_quotes=enable_quote_full_sentence_quotes,
        enable_spot_merge_calls=enable_spot_merge_calls,
        enable_richer_spotter=enable_richer_spotter,
        enable_quote_fallback_bound=enable_quote_fallback_bound,
        enable_quote_unique_anchors=enable_quote_unique_anchors,
        enable_boundary_refinement=DEFAULTS.ENABLE_BOUNDARY_REFINEMENT,
        enable_dsp_boundary_snap=DEFAULTS.ENABLE_DSP_BOUNDARY_SNAP,
        whisper=LocalWhisperConfig(model=whisper_model, whisper_type="local"),
        output=OutputConfig(
            fade_ms=DEFAULTS.OUTPUT_FADE_MS,
            min_ad_segement_separation_seconds=(
                DEFAULTS.OUTPUT_MIN_AD_SEGMENT_SEPARATION_SECONDS
            ),
            min_ad_segment_length_seconds=DEFAULTS.OUTPUT_MIN_AD_SEGMENT_LENGTH_SECONDS,
            min_confidence=DEFAULTS.OUTPUT_MIN_CONFIDENCE,
        ),
        processing=ProcessingConfig(
            num_segments_to_input_to_prompt=(
                DEFAULTS.PROCESSING_NUM_SEGMENTS_TO_INPUT_TO_PROMPT
            ),
            max_overlap_segments=DEFAULTS.PROCESSING_MAX_OVERLAP_SEGMENTS,
        ),
    )
    assert_scout_off(config)
    return config


def default_catalog_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "experiments"
        / "auto_gold"
        / "scorecard_catalog.json"
    )


def load_catalog(path: Path | None = None) -> list[CatalogEpisode]:
    payload = json.loads((path or default_catalog_path()).read_text(encoding="utf-8"))
    if payload.get("gold_family") != GOLD_FAMILY:
        raise ValueError("catalog gold_family lock broken")
    episodes: list[CatalogEpisode] = []
    for row in payload["episodes"]:
        episodes.append(
            CatalogEpisode(
                show_id=str(row["show_id"]),
                genre=str(row["genre"]),
                ad_mechanism=str(row.get("ad_mechanism") or ""),
                episode_title=str(row["episode_title"]),
                windows=list(row["windows"]),
            )
        )
    return episodes


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def force_local_writer() -> None:
    """Never talk to Cake's live writer (127.0.0.1:50001)."""

    def _submit(cmd: Any, wait: bool = False, timeout: int = 10) -> Any:
        del timeout
        result = writer_client._local_execute(cmd)
        return result if wait else None

    writer_client.submit = _submit  # type: ignore[method-assign]


def make_flask(db_path: Path) -> Flask:
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    app = Flask("auto_gold_scorecard")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(db_path)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["TESTING"] = True
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def download_pinned_episodes(
    episodes: list[CatalogEpisode],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    shows = {show.show_id: show for show in load_shows()}
    summary: dict[str, dict[str, Any]] = {}
    for item in episodes:
        show = shows[item.show_id]
        episode, audio_path = fetch_show_episode(
            show,
            output_dir,
            download=True,
            wanted_title=item.episode_title,
        )
        summary[item.show_id] = {
            "wanted_title": item.episode_title,
            "resolved_title": episode.episode_title,
            "audio_path": str(audio_path) if audio_path else None,
            "audio_url": episode.audio_url,
            "error": episode.error,
            "rss_ok": episode.rss_ok,
            "pinned_ok": bool(
                audio_path
                and episode.episode_title
                and item.episode_title.casefold()[:20]
                in (episode.episode_title or "").casefold()
            )
            or bool(audio_path and not episode.error),
        }
        logger.info(
            "download %s -> %s err=%s",
            item.show_id,
            audio_path,
            episode.error,
        )
    _write_json(output_dir / "downloads.json", summary)
    return summary


def judge_frozen_windows(
    episodes: list[CatalogEpisode],
    output_dir: Path,
    downloads: dict[str, dict[str, Any]],
    *,
    max_usd: float,
) -> list[dict[str, Any]]:
    labels_path = output_dir / "labels.json"
    existing_windows: list[dict[str, Any]] = []
    if labels_path.exists():
        existing = _read_json(labels_path)
        loaded = (
            existing.get("windows") if isinstance(existing, dict) else existing
        )
        if isinstance(loaded, list):
            existing_windows = loaded
    have_audio = {
        show_id
        for show_id, row in downloads.items()
        if row.get("audio_path")
    }
    skipped_with_audio = [
        row
        for row in existing_windows
        if row.get("skipped") and row.get("show_id") in have_audio
    ]
    complete = existing_windows and not skipped_with_audio
    if complete and len(existing_windows) >= sum(len(e.windows) for e in episodes):
        logger.info("reusing %s (%s windows)", labels_path, len(existing_windows))
        return existing_windows
    prior: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in existing_windows:
        prior[(str(row["show_id"]), float(row["start"]), float(row["end"]))] = row
    whisper = ChunkWhisper(mode="local", model_name=DEFAULT_WHISPER_MODEL)
    judge = GoldJudge(mode="gemini", max_usd=max_usd)
    labeled: list[dict[str, Any]] = []
    for item in episodes:
        audio = downloads.get(item.show_id, {}).get("audio_path")
        for window in item.windows:
            key = (item.show_id, float(window["start"]), float(window["end"]))
            cached = prior.get(key)
            if cached and not cached.get("skipped"):
                labeled.append(cached)
                continue
            if not audio:
                labeled.append(
                    {
                        "show_id": item.show_id,
                        "start": window["start"],
                        "end": window["end"],
                        "sources": window.get("sources") or [],
                        "is_ad": False,
                        "skipped": True,
                        "skip_reason": "no audio",
                    }
                )
                continue
            src = Path(audio)
            chunk = CandidateChunk(
                start=float(window["start"]),
                end=float(window["end"]),
                sources=list(window.get("sources") or []),
            )
            wav = (
                output_dir
                / "chunks"
                / item.show_id
                / f"{chunk.start:.0f}-{chunk.end:.0f}.wav"
            )
            if not wav.exists():
                extract_chunk_wav(src, wav, chunk.start, chunk.end)
            transcript = whisper.transcribe_chunk(chunk, wav)
            label = judge.judge(transcript)
            labeled.append(
                {
                    "show_id": item.show_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "sources": chunk.sources,
                    "is_ad": bool(label.is_ad) and not label.skipped,
                    "skipped": label.skipped,
                    "skip_reason": label.skip_reason,
                    "content_type": label.content_type,
                    "confidence": label.confidence,
                    "ad_spans": label.ad_spans,
                    "usd": label.usd,
                    "excerpt": (transcript.text or "")[:240],
                }
            )
            logger.info(
                "judge %s %.0f-%.0f is_ad=%s skipped=%s",
                item.show_id,
                chunk.start,
                chunk.end,
                label.is_ad,
                label.skipped,
            )
    prior_spend = 0.0
    if labels_path.exists():
        existing_payload = _read_json(labels_path)
        if isinstance(existing_payload, dict):
            prior_spend = float(existing_payload.get("judge_spend_usd") or 0.0)
    payload = {
        "judge_spend_usd": prior_spend + judge.spent_usd,
        "judge_calls": judge.n_calls,
        "n_is_ad": sum(1 for row in labeled if row.get("is_ad")),
        "windows": labeled,
    }
    _write_json(labels_path, payload)
    return labeled


def _load_or_transcribe(
    audio_path: Path,
    cache_path: Path,
    transcriber: LocalWhisperTranscriber,
) -> list[Segment]:
    if cache_path.exists():
        rows = _read_json(cache_path)
        return [Segment(start=r["start"], end=r["end"], text=r["text"]) for r in rows]
    logger.info("full-episode whisper %s", audio_path)
    segments = transcriber.transcribe(str(audio_path), language="en")
    _write_json(
        cache_path,
        [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    )
    return segments


def run_production_cut(
    *,
    show_id: str,
    title: str,
    audio_path: Path,
    output_dir: Path,
    config: Config,
    app: Flask,
    transcriber: LocalWhisperTranscriber,
    tracker: SpendTracker,
) -> dict[str, Any]:
    cache_path = output_dir / "transcripts" / f"{show_id}.json"
    prod_path = output_dir / "production" / f"{show_id}.json"
    if prod_path.exists():
        return _read_json(prod_path)
    started = time.time()
    segments = _load_or_transcribe(audio_path, cache_path, transcriber)
    cut_path = output_dir / "cut" / f"{show_id}.mp3"
    cut_path.parent.mkdir(parents=True, exist_ok=True)
    user_prompt = Template(
        (Path("src/user_prompt.jinja")).read_text(encoding="utf-8")
    )
    try:
        system_prompt = Path("src/system_prompt.txt").read_text(encoding="utf-8")
    except OSError:
        system_prompt = generate_system_prompt()

    with app.app_context():
        guid = f"scorecard:{show_id}"
        post = db.session.query(Post).filter_by(guid=guid).one_or_none()
        if post is None:
            feed = Feed(
                title=show_id,
                rss_url=f"https://scorecard.local/{show_id}.xml",
                ad_detection_strategy="llm",
            )
            post = Post(
                feed=feed,
                guid=guid,
                download_url=f"https://scorecard.local/{show_id}.mp3",
                title=title,
                description=show_id,
                unprocessed_audio_path=str(audio_path),
                whitelisted=True,
            )
            db.session.add_all([feed, post])
            db.session.commit()
        post.unprocessed_audio_path = str(audio_path)
        db.session.commit()
        db_segments = (
            db.session.query(TranscriptSegment)
            .filter_by(post_id=post.id)
            .order_by(TranscriptSegment.sequence_num)
            .all()
        )
        if not db_segments:
            db_segments = []
            for idx, seg in enumerate(segments):
                db_segments.append(
                    TranscriptSegment(
                        post_id=post.id,
                        sequence_num=idx,
                        start_time=seg.start,
                        end_time=seg.end,
                        text=seg.text,
                    )
                )
            db.session.add_all(db_segments)
            db.session.commit()

        litellm.api_key = config.llm_api_key
        classifier = AdClassifier(config=config, db_session=db.session)
        classifier.classify(
            transcript_segments=db_segments,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            post=post,
        )
        processor = AudioProcessor(config=config, db_session=db.session)
        ad_seconds = processor.get_ad_segments(post)
        duration_ms = get_audio_duration_ms(str(audio_path)) or 0
        merged_ms = processor.merge_ad_segments(
            duration_ms=duration_ms,
            ad_segments=ad_seconds,
            min_ad_segment_length_seconds=float(
                config.output.min_ad_segment_length_seconds
            ),
            min_ad_segment_separation_seconds=float(
                config.output.min_ad_segment_separation_seconds
            ),
        )
        merged_ms = processor.snap_merged_segments(merged_ms, str(audio_path))
        if merged_ms:
            clip_segments_with_fade(
                ad_segments_ms=merged_ms,
                fade_ms=config.output.fade_ms,
                in_path=str(audio_path),
                out_path=str(cut_path),
                use_vbr=True,
            )
        elif not cut_path.exists():
            cut_path.write_bytes(audio_path.read_bytes())

    payload = {
        "show_id": show_id,
        "n_transcript_segments": len(segments),
        "ad_segments_s": [[a, b] for a, b in ad_seconds],
        "cut_segments_s": [[a / 1000.0, b / 1000.0] for a, b in merged_ms],
        "cut_path": str(cut_path),
        "duration_s": duration_ms / 1000.0,
        "spend_usd_after": tracker.usd,
        "scout_confirm": False,
        "wall_s": time.time() - started,
    }
    _write_json(prod_path, payload)
    return payload


def run_quote_anchor_production_cut(
    *,
    show_id: str,
    title: str,
    audio_path: Path,
    output_dir: Path,
    config: Config,
    app: Flask,
    transcriber: LocalWhisperTranscriber,
    tracker: SpendTracker,
    words_cache: Path | None = None,
    llm_cache_dir: Path | None = None,
    complete_fn: Any | None = None,
    skip_audio_clip: bool = True,
    silence_gaps: list[tuple[float, float]] | None = None,
    transcripts_dir: Path | None = None,
    require_llm_cache: bool = False,
    prefer_transcript_segments: bool = True,
    pull_dai_bed: bool = False,
    extra_phrases: tuple[str, ...] = (),
    extra_names: tuple[str, ...] = (),
    ngram_phrases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Integrated quote-anchor classify + AudioProcessor merge/snap.

    Does not call AdClassifier / neighbor expand / boundary refiner.
    """
    from podcast_processor.quote_anchor_classifier import QuoteAnchorClassifier
    from podcast_processor.experiments.word_transcripts import (
        load_word_payload,
        segments_from_payload,
    )
    from podcast_processor.transcribe import Segment as WhisperSegment

    cache_path = (transcripts_dir or (output_dir / "transcripts")) / f"{show_id}.json"
    prod_path = output_dir / "production" / f"{show_id}.json"
    if prod_path.exists():
        return _read_json(prod_path)
    segments: list[WhisperSegment]
    word_payload = load_word_payload(words_cache) if words_cache else None
    word_segs = segments_from_payload(word_payload) if word_payload else []
    if word_segs and not prefer_transcript_segments:
        segments = [
            WhisperSegment(start=a, end=b, text=text) for a, b, text in word_segs if b > a
        ]
    else:
        segments = _load_or_transcribe(audio_path, cache_path, transcriber)
    cut_path = output_dir / "cut" / f"{show_id}.mp3"
    cut_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with app.app_context():
        guid = f"scorecard-qa:{show_id}"
        post = db.session.query(Post).filter_by(guid=guid).one_or_none()
        if post is None:
            feed = Feed(
                title=show_id,
                rss_url=f"https://scorecard.local/qa/{show_id}.xml",
                ad_detection_strategy="llm",
            )
            post = Post(
                feed=feed,
                guid=guid,
                download_url=f"https://scorecard.local/qa/{show_id}.mp3",
                title=title,
                description=show_id,
                unprocessed_audio_path=str(audio_path),
                whitelisted=True,
            )
            db.session.add_all([feed, post])
            db.session.commit()
        post.unprocessed_audio_path = str(audio_path)
        post.refined_ad_boundaries = None
        db.session.commit()
        existing_seg_ids = [
            row[0]
            for row in db.session.query(TranscriptSegment.id)
            .filter_by(post_id=post.id)
            .all()
        ]
        if existing_seg_ids:
            db.session.query(Identification).filter(
                Identification.transcript_segment_id.in_(existing_seg_ids)
            ).delete(synchronize_session=False)
        db.session.query(ModelCall).filter(ModelCall.post_id == post.id).delete()
        db.session.query(TranscriptSegment).filter(
            TranscriptSegment.post_id == post.id
        ).delete()
        db.session.commit()
        db_segments = []
        for idx, seg in enumerate(segments):
            db_segments.append(
                TranscriptSegment(
                    post_id=post.id,
                    sequence_num=idx,
                    start_time=seg.start,
                    end_time=seg.end,
                    text=seg.text,
                )
            )
        db.session.add_all(db_segments)
        db.session.commit()

        litellm.api_key = config.llm_api_key
        classifier = QuoteAnchorClassifier(
            config=config,
            logger=logger,
            db_session=db.session,
            complete_fn=complete_fn,
            word_cache_path=words_cache,
            llm_cache_dir=llm_cache_dir,
            require_llm_cache=require_llm_cache,
            pull_dai_bed=pull_dai_bed,
            extra_phrases=extra_phrases,
            extra_names=extra_names,
            ngram_phrases=ngram_phrases,
        )
        classifier.silence_gaps_override = silence_gaps
        classifier.classify(transcript_segments=db_segments, post=post)
        processor = AudioProcessor(config=config, db_session=db.session)
        ad_seconds = processor.get_ad_segments(post)
        duration_ms = get_audio_duration_ms(str(audio_path)) or 0
        merged_ms = processor.merge_ad_segments(
            duration_ms=duration_ms,
            ad_segments=ad_seconds,
            min_ad_segment_length_seconds=float(
                config.output.min_ad_segment_length_seconds
            ),
            min_ad_segment_separation_seconds=float(
                config.output.min_ad_segment_separation_seconds
            ),
        )
        if silence_gaps is not None:
            from podcast_processor.boundary_snap import SilenceGap, refine_cuts

            typed_gaps = [SilenceGap(start=a, end=b) for a, b in silence_gaps]
            snapped = refine_cuts(
                [(a / 1000.0, b / 1000.0) for a, b in merged_ms],
                gaps=typed_gaps,
                radius=2.0,
            )
            merged_ms = [(int(a * 1000), int(b * 1000)) for a, b in snapped]
        else:
            merged_ms = processor.snap_merged_segments(merged_ms, str(audio_path))
        if not skip_audio_clip and merged_ms:
            clip_segments_with_fade(
                ad_segments_ms=merged_ms,
                fade_ms=config.output.fade_ms,
                in_path=str(audio_path),
                out_path=str(cut_path),
                use_vbr=True,
            )

    wall_s = time.time() - started
    payload = {
        "show_id": show_id,
        "n_transcript_segments": len(segments),
        "ad_segments_s": [[a, b] for a, b in ad_seconds],
        "cut_segments_s": [[a / 1000.0, b / 1000.0] for a, b in merged_ms],
        "cut_path": str(cut_path),
        "duration_s": duration_ms / 1000.0,
        "spend_usd_after": tracker.usd,
        "scout_confirm": False,
        "quote_anchor": True,
        "n_quotes": len(classifier.last_matches),
        "n_word": sum(1 for item in classifier.last_matches if item.method == "word"),
        "n_segment_fallback": sum(
            1 for item in classifier.last_matches if item.method == "segment_fallback"
        ),
        "n_chunk_fallback": sum(
            1 for item in classifier.last_matches if item.method == "chunk_fallback"
        ),
        "n_dropped": sum(1 for item in classifier.last_matches if item.dropped),
        "input_tokens": classifier.last_input_tokens,
        "output_tokens": classifier.last_output_tokens,
        "n_llm_calls": classifier.last_llm_calls,
        "n_candidates": len(classifier.last_candidates),
        "n_extends": classifier.last_extends,
        "n_reasks": classifier.last_reasks,
        "n_unique_ok": classifier.last_unique_ok,
        "n_unique_fallbacks": classifier.last_unique_fallbacks,
        "spot_then_confirm": bool(
            getattr(config, "enable_spot_then_confirm", False)
        ),
        "wall_s": wall_s,
        "word_whisper_s": classifier.last_word_whisper_s,
        "classify_s": classifier.last_classify_s,
    }
    _write_json(prod_path, payload)
    return payload


def load_production_cache(output_dir: Path) -> dict[str, dict[str, Any]]:
    prod_dir = output_dir / "production"
    cached: dict[str, dict[str, Any]] = {}
    if not prod_dir.is_dir():
        return cached
    for path in sorted(prod_dir.glob("*.json")):
        row = _read_json(path)
        cached[str(row.get("show_id") or path.stem)] = row
    return cached


def apply_boundary_snap_to_production(
    production: dict[str, dict[str, Any]],
    *,
    repo_root: Path,
    audio_dir: Path,
    radius: float,
    prefer: str = "nearest",
) -> dict[str, dict[str, Any]]:
    """Eval-only: snap cached cuts using transcripts + ffmpeg silencedetect ($0)."""
    from podcast_processor.boundary_snap import detect_silence_gaps, refine_cuts

    downloads_path = audio_dir / "downloads.json"
    downloads: dict[str, Any] = {}
    if downloads_path.is_file():
        downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
    transcripts_dir = audio_dir / "transcripts"
    refined: dict[str, dict[str, Any]] = {}
    for show_id, row in production.items():
        payload = dict(row)
        cuts = [(float(a), float(b)) for a, b in (row.get("cut_segments_s") or [])]
        audio_raw = (downloads.get(show_id) or {}).get("audio_path")
        audio_path: Path | None = Path(str(audio_raw)) if audio_raw else None
        if audio_path is not None and not audio_path.is_absolute():
            audio_path = repo_root / audio_path
        segments: list[tuple[float, float, str]] = []
        transcript_path = transcripts_dir / f"{show_id}.json"
        if transcript_path.is_file():
            for seg in json.loads(transcript_path.read_text(encoding="utf-8")):
                segments.append(
                    (
                        float(seg["start"]),
                        float(seg["end"]),
                        str(seg.get("text") or ""),
                    )
                )
        snapped = refine_cuts(
            cuts,
            audio_path=audio_path,
            segments=segments or None,
            radius=radius,
            prefer=prefer,  # type: ignore[arg-type]
            gaps=(
                detect_silence_gaps(audio_path)
                if audio_path is not None and audio_path.is_file()
                else None
            ),
        )
        payload["cut_segments_s_unsnapped"] = [[a, b] for a, b in cuts]
        payload["cut_segments_s"] = [[a, b] for a, b in snapped]
        payload["boundary_snap"] = True
        payload["boundary_snap_radius"] = radius
        payload["boundary_snap_prefer"] = prefer
        refined[show_id] = payload
    return refined


def seed_spend_from_production(
    tracker: SpendTracker, production: dict[str, dict[str, Any]]
) -> None:
    if not production:
        return
    tracker.usd = max(
        float(tracker.usd),
        max(float(row.get("spend_usd_after") or 0.0) for row in production.values()),
    )


def build_scorecard(
    labeled: list[dict[str, Any]],
    production: dict[str, dict[str, Any]],
    *,
    spend_usd: float,
    notes: list[str],
) -> ScorecardResult:
    scored_ids = set(production)
    cuts = {
        show_id: [(float(a), float(b)) for a, b in row.get("cut_segments_s") or []]
        for show_id, row in production.items()
    }
    result = score_windows(
        [
            row
            for row in labeled
            if not row.get("skipped") and row.get("show_id") in scored_ids
        ],
        cuts,
        preroll_end=PREROLL_END_SECONDS,
    )
    result.spend_usd = spend_usd
    result.spend_cap_usd = DEFAULT_JUDGE_BUDGET_USD
    result.n_shows_catalog = 12
    result.n_shows_scored = len(production)
    result.notes = notes
    result.outliers = opening_fn_notes(result) + fp_notes(result)
    return result


def write_scorecard(result: ScorecardResult, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_scorecard_md(result), encoding="utf-8")


def _is_classification_response(response: str | None) -> bool:
    if not response:
        return False
    head = response.lstrip()[:240]
    if "refined_start" in head:
        return False
    return '"ad_segments"' in response or "'ad_segments'" in response


def replay_production_from_cached_llm(
    *,
    source_dir: Path,
    output_dir: Path,
    config: Config,
) -> dict[str, dict[str, Any]]:
    """Rebuild cuts from cached ModelCall responses. No new LLM calls.

    Copies Friday's scorecard sqlite, drops identifications, re-applies
    _create_identifications with the current confidence policy, expands
    neighbors, and reuses existing refined_ad_boundaries (no refine LLM).
    """
    src_sqlite = source_dir / "scorecard.sqlite"
    if not src_sqlite.is_file():
        raise FileNotFoundError(f"missing cached sqlite: {src_sqlite}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_sqlite = output_dir / "scorecard.sqlite"
    shutil.copy2(src_sqlite, dest_sqlite)

    source_prod = load_production_cache(source_dir)
    app = make_flask(dest_sqlite)
    force_local_writer()
    production: dict[str, dict[str, Any]] = {}

    with app.app_context():
        db.session.query(Identification).delete()
        db.session.commit()

        classifier = AdClassifier(config=config, db_session=db.session)
        processor = AudioProcessor(config=config, db_session=db.session)
        posts = db.session.query(Post).order_by(Post.id).all()
        n_replayed = 0
        for post in posts:
            guid = str(post.guid or "")
            show_id = guid.split(":", 1)[-1] if guid.startswith("scorecard:") else guid
            segments = (
                db.session.query(TranscriptSegment)
                .filter_by(post_id=post.id)
                .order_by(TranscriptSegment.sequence_num)
                .all()
            )
            by_seq = {int(seg.sequence_num): seg for seg in segments}
            model_calls = (
                db.session.query(ModelCall)
                .filter_by(post_id=post.id, status="success")
                .order_by(ModelCall.id)
                .all()
            )
            for model_call in model_calls:
                if not _is_classification_response(model_call.response):
                    continue
                chunk = [
                    by_seq[seq]
                    for seq in range(
                        int(model_call.first_segment_sequence_num),
                        int(model_call.last_segment_sequence_num) + 1,
                    )
                    if seq in by_seq
                ]
                if not chunk:
                    continue
                classifier._process_successful_response(
                    model_call=model_call,
                    current_chunk_db_segments=chunk,
                )
                n_replayed += 1

            ad_identifications = (
                db.session.query(Identification)
                .join(TranscriptSegment)
                .filter(
                    TranscriptSegment.post_id == post.id,
                    Identification.label == "ad",
                )
                .all()
            )
            if ad_identifications:
                model_call = ad_identifications[0].model_call
                if model_call is not None:
                    classifier.expand_neighbors_bulk(
                        ad_identifications=ad_identifications,
                        model_call=model_call,
                        post_id=post.id,
                        window=5,
                    )

            ad_seconds = processor.get_ad_segments(post)
            prior = source_prod.get(show_id) or {}
            duration_s = float(prior.get("duration_s") or 0.0)
            duration_ms = int(round(duration_s * 1000.0))
            merged_ms = processor.merge_ad_segments(
                duration_ms=duration_ms,
                ad_segments=ad_seconds,
                min_ad_segment_length_seconds=float(
                    config.output.min_ad_segment_length_seconds
                ),
                min_ad_segment_separation_seconds=float(
                    config.output.min_ad_segment_separation_seconds
                ),
            )
            snap_audio = post.unprocessed_audio_path
            if not snap_audio or not Path(str(snap_audio)).is_file():
                downloads_path = source_dir / "downloads.json"
                if downloads_path.is_file():
                    raw = (
                        json.loads(downloads_path.read_text(encoding="utf-8"))
                        .get(show_id, {})
                        .get("audio_path")
                    )
                    if raw:
                        candidate = Path(str(raw))
                        if not candidate.is_absolute():
                            candidate = Path(__file__).resolve().parents[3] / candidate
                        if candidate.is_file():
                            snap_audio = str(candidate)
            merged_ms = processor.snap_merged_segments(merged_ms, snap_audio)
            payload = {
                "show_id": show_id,
                "n_transcript_segments": len(segments),
                "ad_segments_s": [[a, b] for a, b in ad_seconds],
                "cut_segments_s": [[a / 1000.0, b / 1000.0] for a, b in merged_ms],
                "cut_path": prior.get("cut_path"),
                "duration_s": duration_s,
                "spend_usd_after": 0.0,
                "scout_confirm": False,
                "replay_cached_llm": True,
            }
            prod_path = output_dir / "production" / f"{show_id}.json"
            _write_json(prod_path, payload)
            production[show_id] = payload
            logger.info(
                "replayed %s ads=%s cuts=%s",
                show_id,
                payload["ad_segments_s"][:3],
                payload["cut_segments_s"][:3],
            )
        logger.info("replayed %s classification model_calls, $0 Gemini", n_replayed)
    return production


def _is_self_promo_not_ad(row: dict[str, Any]) -> bool:
    if row.get("skipped") or row.get("is_ad"):
        return False
    return str(row.get("content_type") or "") == "educational/self_promo"


def rejudge_self_promo_not_ad_windows(
    *,
    source_dir: Path,
    output_dir: Path,
    max_usd: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Re-judge not_ad educational/self_promo windows only. Other labels unchanged."""
    labels_path = source_dir / "labels.json"
    payload = _read_json(labels_path)
    windows = list(payload.get("windows") or [])
    downloads = {}
    downloads_path = source_dir / "downloads.json"
    if downloads_path.is_file():
        downloads = _read_json(downloads_path)
    targets = [
        (idx, row) for idx, row in enumerate(windows) if _is_self_promo_not_ad(row)
    ]
    if not targets:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "label_flips.json", {"flips": [], "spend_usd": 0.0})
        _write_json(output_dir / "labels.json", payload)
        return windows, [], 0.0

    whisper = ChunkWhisper(mode="local", model_name=DEFAULT_WHISPER_MODEL)
    judge = GoldJudge(mode="gemini", max_usd=max_usd)
    flips: list[dict[str, Any]] = []
    for idx, row in targets:
        show_id = str(row["show_id"])
        start = float(row["start"])
        end = float(row["end"])
        audio = downloads.get(show_id, {}).get("audio_path")
        wav = source_dir / "chunks" / show_id / f"{start:.0f}-{end:.0f}.wav"
        if not wav.is_file():
            if not audio:
                logger.warning("skip rejudge %s %.0f-%.0f: no audio/wav", show_id, start, end)
                continue
            wav.parent.mkdir(parents=True, exist_ok=True)
            extract_chunk_wav(Path(audio), wav, start, end)
        chunk = CandidateChunk(
            start=start,
            end=end,
            sources=list(row.get("sources") or []),
        )
        transcript = whisper.transcribe_chunk(chunk, wav)
        label = judge.judge(transcript)
        new_row = {
            **row,
            "is_ad": bool(label.is_ad) and not label.skipped,
            "skipped": label.skipped,
            "skip_reason": label.skip_reason,
            "content_type": label.content_type,
            "confidence": label.confidence,
            "ad_spans": label.ad_spans,
            "usd": (row.get("usd") or 0.0) + (label.usd or 0.0),
            "excerpt": (transcript.text or row.get("excerpt") or "")[:240],
            "rejudged_policy_2026_10_05": True,
        }
        windows[idx] = new_row
        flip = {
            "show_id": show_id,
            "start": start,
            "end": end,
            "old_is_ad": bool(row.get("is_ad")),
            "new_is_ad": bool(new_row["is_ad"]),
            "old_content_type": row.get("content_type"),
            "new_content_type": new_row.get("content_type"),
            "old_ad_spans": row.get("ad_spans") or [],
            "new_ad_spans": new_row.get("ad_spans") or [],
            "usd": label.usd,
            "skipped": label.skipped,
            "skip_reason": label.skip_reason,
        }
        flips.append(flip)
        logger.info(
            "rejudge %s %.0f-%.0f is_ad %s -> %s type %s -> %s $%.4f",
            show_id,
            start,
            end,
            flip["old_is_ad"],
            flip["new_is_ad"],
            flip["old_content_type"],
            flip["new_content_type"],
            float(label.usd or 0.0),
        )

    out_payload = {
        "judge_spend_usd": float(payload.get("judge_spend_usd") or 0.0) + judge.spent_usd,
        "judge_calls": int(payload.get("judge_calls") or 0) + judge.n_calls,
        "n_is_ad": sum(1 for row in windows if row.get("is_ad")),
        "windows": windows,
        "policy_rejudge_spend_usd": judge.spent_usd,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "labels.json", out_payload)
    _write_json(
        output_dir / "label_flips.json",
        {"flips": flips, "spend_usd": judge.spent_usd, "n_targets": len(targets)},
    )
    return windows, flips, judge.spent_usd
