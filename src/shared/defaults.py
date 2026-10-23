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
# 2026-10-10 algo-theories (all default off; live must unset).
ENABLE_QUOTE_RESUME_TRIM = False
ENABLE_QUOTE_BACKFILL_LEADIN = False
ENABLE_QUOTE_CLOSER_TAIL = False
ENABLE_SPOT_WEAK_DROP = False
ENABLE_SNAP_EARLIER_START = False
# Keep classified DAI prerolls in (10s, 14s), e.g. Planet Money Pushkin
# 12.96s. Default off; live must set ENABLE_SHORT_MIN_AD_SEGMENT.
ENABLE_SHORT_MIN_AD_SEGMENT = False
OUTPUT_SHORT_MIN_AD_SEGMENT_LENGTH_SECONDS = 10
# Pad 60s on each side of every spotter candidate (full minute around
# the spot). Default 50s stays unless ENABLE_SPOT_MINUTE_PAD is set.
ENABLE_SPOT_MINUTE_PAD = False
# On confirm extend, union first+second ad spans instead of wiping the
# first good confirm. Default off; live must set ENABLE_SPOT_CONFIRM_UNION.
ENABLE_SPOT_CONFIRM_UNION = False
# Seconds of context on each side of a spotted cluster when the minute-pad
# flag is on. Off-path remains SPOT_PAD_S (50).
SPOT_MINUTE_PAD_SECONDS = 60.0
# Pad 70s each side of a spot (60s minute-pad failed holdout catch
# 99.01%→98.06% on TAL; 70s is the CTA-only lead-in that reaches
# Prof G "Support for the show comes from the AG one" from the URL).
ENABLE_SPOT_WIDE_PAD = False
SPOT_WIDE_PAD_SECONDS = 70.0
# Merge two spotter detections into one LLM confirm window when the
# gap between them is below this (default 2 minutes).
ENABLE_SPOT_BRIDGE_WINDOWS = False
SPOT_BRIDGE_GAP_SECONDS = 120.0
# Keep the first spotter cluster on the 50s pad and never bridge it
# into the next detection. Default off.
ENABLE_SPOT_OPENING_ISOLATE = False
# Confirm prompt v3: full-sentence quotes; sister teasers, newsletter /
# follow / YouTube plugs, self-promo and membership asks are ads.
ENABLE_QUOTE_PROMPT_FULL_BREAK = False

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
# Do not fill a hole of show content between two ads (Up First 0-24s + 73-99s
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
