"""Duration accounting + residual CueDetector scan after hypothetical cuts.

Aligned with podcast_processor.audio.clip_segments_with_fade:

Fades apply to kept *content* edges only. Ad audio is not faded back in, so:

    output_ms ~ source_ms - sum(ad_duration_ms)

(the previous formula added `2 * fade_ms * n_cuts` of ad audio and leaked
preroll DAI). Simple-concat is the same duration. ffmpeg mux jitter of ~56ms
is documented in `src/tests/test_process_audio.py` and is **not** included
in the formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from podcast_processor.cue_detector import CueDetector
from podcast_processor.experiments.bow_scout import overlap_seconds
from podcast_processor.experiments.types import LabeledAd, ScoutSegment
from shared import defaults as DEFAULTS


@dataclass(frozen=True)
class DurationAccount:
    source_ms: int
    removed_ms: int
    fade_added_ms: int
    expected_output_ms: int
    n_cuts: int
    used_complex_filter: bool


@dataclass(frozen=True)
class ResidualCue:
    sequence_num: int
    start_time: int | float
    end_time: int | float
    text: str
    signals: dict[str, bool]


def expected_output_duration_ms(
    source_ms: int,
    ad_segments_ms: list[tuple[int, int]],
    fade_ms: int = DEFAULTS.OUTPUT_FADE_MS,
    complex_filter: bool = True,
) -> DurationAccount:
    if source_ms < 0:
        raise ValueError("source_ms must be non-negative")
    removed = 0
    for start_ms, end_ms in ad_segments_ms:
        if end_ms < start_ms:
            raise ValueError(f"ad segment end < start: {(start_ms, end_ms)}")
        removed += end_ms - start_ms
    n_cuts = len(ad_segments_ms)
    # fade_ms / complex_filter kept for API compatibility. Content-edge fades
    # do not restore ad audio, so they do not add duration.
    fade_added = 0
    expected = source_ms - removed + fade_added
    _ = fade_ms
    return DurationAccount(
        source_ms=source_ms,
        removed_ms=removed,
        fade_added_ms=fade_added,
        expected_output_ms=expected,
        n_cuts=n_cuts,
        used_complex_filter=complex_filter,
    )


def ads_to_ms(
    ads: list[LabeledAd] | list[tuple[float, float]],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for ad in ads:
        if isinstance(ad, LabeledAd):
            start_s, end_s = ad.start, ad.end
        else:
            start_s, end_s = ad
        result.append((round(start_s * 1000), round(end_s * 1000)))
    return result


def residual_cue_scan(
    segments: list[ScoutSegment],
    removed_ads: list[LabeledAd] | list[tuple[float, float]],
    detector: CueDetector | None = None,
    min_overlap: int | float = 0.25,
    strong_only: bool = True,
) -> list[ResidualCue]:
    """Flag CueDetector hits that remain after the proposed ad cuts.

    A leftover strong cue is a false-negative risk (ad leftover in output).
    """
    cue = detector or CueDetector(include_scout_extras=True)
    ads: list[tuple[float, float]]
    if removed_ads and isinstance(removed_ads[0], LabeledAd):
        ads = [(ad.start, ad.end) for ad in removed_ads]  # type: ignore[misc]
    else:
        ads = [(float(a[0]), float(a[1])) for a in removed_ads]  # type: ignore[index]

    leftovers: list[ResidualCue] = []
    for segment in segments:
        covered = sum(
            overlap_seconds(segment.start_time, segment.end_time, start, end)
            for start, end in ads
        )
        if covered >= min_overlap:
            continue
        signals = cue.analyze(segment.text)
        if strong_only:
            keep = cue.has_strong_cue(segment.text) or signals.get("sponsor", False)
        else:
            keep = any(signals.values())
        if keep:
            leftovers.append(
                ResidualCue(
                    sequence_num=segment.sequence_num,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=segment.text,
                    signals=signals,
                )
            )
    return leftovers
