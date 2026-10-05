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


def production_config(*, keep_preroll_self_promo: bool = False) -> Config:
    llm_key = os.environ.get("LLM_API_KEY") or gemini_api_key()
    whisper_model = os.environ.get("WHISPER_LOCAL_MODEL") or "base"
    config = Config(
        llm_api_key=llm_key,
        llm_model=os.environ.get("LLM_MODEL") or DEFAULT_GEMINI_MODEL,
        enable_bow_scout_gemini_confirm=False,
        enable_keep_preroll_self_promo=keep_preroll_self_promo,
        enable_boundary_refinement=DEFAULTS.ENABLE_BOUNDARY_REFINEMENT,
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
                "keep_preroll_self_promo": bool(config.enable_keep_preroll_self_promo),
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
