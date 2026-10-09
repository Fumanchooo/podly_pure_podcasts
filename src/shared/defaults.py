from __future__ import annotations

# Centralized default values for application configuration.
# Single source of truth for defaults across runtime, DB models, and Pydantic config.

# LLM defaults
LLM_DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
OPENAI_DEFAULT_MAX_TOKENS = 4096
OPENAI_DEFAULT_TIMEOUT_SEC = 300
LLM_DEFAULT_MAX_CONCURRENT_CALLS = 3
LLM_DEFAULT_MAX_RETRY_ATTEMPTS = 5
LLM_ENABLE_TOKEN_RATE_LIMITING = False
LLM_MAX_INPUT_TOKENS_PER_CALL: int | None = None
LLM_MAX_INPUT_TOKENS_PER_MINUTE: int | None = None
ENABLE_BOUNDARY_REFINEMENT = True
ENABLE_WORD_LEVEL_BOUNDARY_REFINDER = False
ENABLE_LLM_CHAPTER_FALLBACK_TAGGING = False
# $0 ffmpeg silencedetect snap of cut edges. Folded 2026-10-05 (±2.0s nearest).
ENABLE_DSP_BOUNDARY_SNAP = True
# Experimental CueDetector-scout + Gemini-confirm classifier. Default off:
# production stays on the full AdClassifier LLM walk until this is explicitly
# enabled *and* the experimental path is wired. See docs/experiments/AGENT_EVAL.md.
ENABLE_BOW_SCOUT_GEMINI_CONFIRM = False
# Quote-anchor cutter (prompt_v2 + lexical_v2). Default off until the
# original-10 + holdout gates pass; live deploy flips this True.
ENABLE_QUOTE_ANCHOR_BOUNDARIES = False
# Spot-then-confirm: $0 lexicon spotter + one LLM call per candidate
# (padded region), not a full-episode chunk walk. Default off.
ENABLE_SPOT_THEN_CONFIRM = False

# Whisper defaults
WHISPER_DEFAULT_TYPE = "groq"
WHISPER_LOCAL_MODEL = "base.en"
WHISPER_REMOTE_BASE_URL = "https://api.openai.com/v1"
WHISPER_REMOTE_MODEL = "whisper-1"
WHISPER_REMOTE_LANGUAGE = "en"
WHISPER_REMOTE_TIMEOUT_SEC = 600
WHISPER_REMOTE_CHUNKSIZE_MB = 24

WHISPER_GROQ_MODEL = "whisper-large-v3-turbo"
WHISPER_GROQ_LANGUAGE = "en"
WHISPER_GROQ_MAX_RETRIES = 0

# Processing defaults
PROCESSING_NUM_SEGMENTS_TO_INPUT_TO_PROMPT = 60
PROCESSING_MAX_OVERLAP_SEGMENTS = 30

# Output defaults
OUTPUT_FADE_MS = 3000
OUTPUT_MIN_AD_SEGMENT_SEPARATION_SECONDS = 60
OUTPUT_MIN_AD_SEGMENT_LENGTH_SECONDS = 14
OUTPUT_MIN_CONFIDENCE = 0.8
# Do not fill a hole of show content between two ads (Up First 0–24s + 73–99s
# membership CTAs were merged through ~50s of headlines at the 60s separation).
OUTPUT_MIN_AD_CONTENT_GAP_SECONDS = 12

# App defaults
APP_BACKGROUND_UPDATE_INTERVAL_MINUTE = 30
APP_AUTOMATICALLY_WHITELIST_NEW_EPISODES = True
APP_NUM_EPISODES_TO_WHITELIST_FROM_ARCHIVE_OF_NEW_FEED = 1
APP_POST_CLEANUP_RETENTION_DAYS = 5
APP_ENABLE_PUBLIC_LANDING_PAGE = False
APP_USER_LIMIT_TOTAL: int | None = None
APP_AUTOPROCESS_ON_DOWNLOAD = False
APP_COST_RATE_PER_HOUR = 0.04

# Credits defaults
MINUTES_PER_CREDIT = 60

# Chapter filter defaults
AD_DETECTION_DEFAULT_STRATEGY = "llm"
CHAPTER_FILTER_DEFAULT_STRINGS = (
    "sponsor,advertisement,ad break,promo,brought to you by"
)
