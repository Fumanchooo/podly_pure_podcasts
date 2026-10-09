# Cactus Whistle STT (2026-10-09)

Scout confirm stays **false**. `general_podcast_ads` remains **LOCKED**.
Code default `ENABLE_WHISTLE_STT=false`. Live stays on Whisper
segment-align + spot-then-confirm (`1b7728d`) unless this fold passes.

## Plan critique (before the run)

- 30 s / 320-token cap: transcribe in 28 s windows with 1.5 s overlap.
  Smoke on a 28 s Crime Junkie clip: 74 words, last word 27.28 s (cap not
  binding at ~2.6 wps). Fast-speech truncation is logged if a clip has
  ≥280 words or a ≥1 s tail after dense speech.
- Seam drops: stitch at the overlap midpoint, keep the first copy,
  record gaps ≥0.4 s.
- Keyword biasing can hallucinate lexicon names in content and inflate
  candidates / `not_ad`. Measure spotter recall with and without
  `SPONSOR_NAMES` before any Gemini e2e.
- WER is vs frozen Whisper **segment-align** words, not human gold.
  Decoder `word_timestamps=True` seek-drop is not the live baseline.
  Listen substitute: extract disagreement wavs and dump both texts
  (no human ear on this worker).
- E2E confirm prompts will miss the Whisper LLM cache. UTC leftover
  after spot-then-confirm is ~$0.35; one orig10+holdout pass is ~$0.15
  if similar. Spend only if spotter is close to 21/21.
- `cactus-needle` stays out of the live venv. Distinct
  `*.whistle.json` caches so Whisper `*.words.json` is never overwritten.
- Engine is not thread-safe; one process, a lock around `transcribe`.

## Smoke (28 s Crime Junkie 184–212)

| engine | wall | RTF | note |
| --- | ---: | ---: | --- |
| Whisper `base` CPU | 2.39 s | 0.085 | “45 caliber” |
| Whistle cold (weights load) | 14.6 s | 0.52 | includes first-load |
| Whistle warm | 3.88 s | 0.139 | “forty five taller” |

13.2 episode-hours × 0.14 RTF ≈ 1.8 h wall per ASR pass on Cake.

## Scorecard

_Pending ASR + spotter + optional e2e._

## Live

Do not enable `ENABLE_WHISTLE_STT` unless fold gates pass (catch ≥ live
spot-then-confirm, openings not worse, `not_ad` 0.00 s). Rollback remains
`git checkout --detach 1b7728d` with quote-anchor + spot-then-confirm on.
