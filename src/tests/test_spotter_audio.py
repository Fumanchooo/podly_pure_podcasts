"""Loudness / speaker-change $0 cues from synthetic frames."""

from __future__ import annotations

import numpy as np

from podcast_processor.spotter_audio import (
    loudness_hits_from_rms,
    speaker_hits_from_features,
)


def test_loudness_hits_step_change() -> None:
    rms = np.concatenate(
        [np.full(40, 0.05), np.full(20, 0.25), np.full(40, 0.05)]
    )
    hits = loudness_hits_from_rms(rms, hop_s=0.25, delta_db=4.0)
    assert hits
    assert all(h.kind == "loudness" for h in hits)
    assert any(h.end - h.start >= 1.5 for h in hits)


def test_speaker_hits_feature_outlier() -> None:
    n = 80
    rms = np.full(n, 0.08)
    zcr = np.full(n, 0.05)
    centroid = np.full(n, 800.0)
    centroid[30:50] = 3200.0
    zcr[30:50] = 0.35
    hits = speaker_hits_from_features(rms, zcr, centroid, hop_s=0.25, z_thresh=2.8)
    assert hits
    assert all(h.kind == "speaker" for h in hits)
