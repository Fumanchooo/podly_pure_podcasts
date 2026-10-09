"""Production defaults stay on the current LLM AdClassifier path."""

from __future__ import annotations

import inspect

from podcast_processor.cue_detector import CORE_ANALYZE_KEYS, CueDetector
from podcast_processor.podcast_processor import (
    PodcastProcessor,
    bow_scout_gemini_confirm_enabled,
    quote_anchor_boundaries_enabled,
    spot_then_confirm_enabled,
    whistle_stt_enabled,
)
from shared import defaults as DEFAULTS
from shared.config import Config
from shared.test_utils import create_standard_test_config


def test_bow_scout_flag_defaults_off() -> None:
    assert DEFAULTS.ENABLE_BOW_SCOUT_GEMINI_CONFIRM is False
    config = create_standard_test_config()
    assert config.enable_bow_scout_gemini_confirm is False
    assert bow_scout_gemini_confirm_enabled(config) is False
    assert DEFAULTS.ENABLE_DSP_BOUNDARY_SNAP is True
    assert config.enable_dsp_boundary_snap is True
    assert DEFAULTS.ENABLE_QUOTE_ANCHOR_BOUNDARIES is False
    assert config.enable_quote_anchor_boundaries is False
    assert quote_anchor_boundaries_enabled(config) is False
    assert DEFAULTS.ENABLE_SPOT_THEN_CONFIRM is False
    assert config.enable_spot_then_confirm is False
    assert spot_then_confirm_enabled(config) is False
    assert DEFAULTS.ENABLE_WHISTLE_STT is False
    assert config.enable_whistle_stt is False
    assert whistle_stt_enabled(config) is False


def test_config_flag_can_be_enabled_without_swapping_defaults() -> None:
    config = create_standard_test_config()
    enabled = config.model_copy(update={"enable_bow_scout_gemini_confirm": True})
    assert bow_scout_gemini_confirm_enabled(enabled) is True
    assert create_standard_test_config().enable_bow_scout_gemini_confirm is False


def test_feed_ad_detection_default_remains_llm() -> None:
    assert DEFAULTS.AD_DETECTION_DEFAULT_STRATEGY == "llm"
    from app.models import Feed

    column = Feed.__table__.c.ad_detection_strategy
    assert column.default is not None
    assert column.default.arg == DEFAULTS.AD_DETECTION_DEFAULT_STRATEGY


def test_cue_detector_production_constructor_unchanged() -> None:
    detector = CueDetector()
    assert detector.include_scout_extras is False
    assert tuple(detector.analyze("hello").keys()) == CORE_ANALYZE_KEYS


def test_classify_falls_back_to_adclassifier_when_flag_on() -> None:
    source = inspect.getsource(PodcastProcessor._classify_ad_segments)
    assert "bow_scout_gemini_confirm_enabled" in source
    assert "self.ad_classifier.classify" in source
    assert "Falling back to the production AdClassifier" in source


def test_pydantic_config_field_defaults_false() -> None:
    field = Config.model_fields["enable_bow_scout_gemini_confirm"]
    assert field.default is False
    assert Config.model_fields["enable_spot_then_confirm"].default is False
    assert Config.model_fields["enable_whistle_stt"].default is False
