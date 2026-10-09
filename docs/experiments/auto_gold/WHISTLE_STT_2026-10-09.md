# Cactus Whistle STT (2026-10-09)

Scout confirm stays **false**. `general_podcast_ads` remains **LOCKED**.
Code default `ENABLE_WHISTLE_STT=false`. **Not folded.** Live stays
Whisper segment-align + spot-then-confirm (`1b7728d`).

## Claim

Whistle is a working 16.9 MB CPU ASR on Cake (WER **13.3%** vs frozen
Whisper-base aligned words, RTF **0.20**, word-time median error
**0.18 s**) and the $0 spotter still hits **21/21** without keyword
bias, but spot-then-confirm catch on the original 10 drops
**97.93% → 89.71%** with undercut **9.9 s → 48.9 s**. Keyword biasing
**loses** one gold ad (20/21). Do not replace live Whisper.

## Plan critique (before the run)

- 30 s / 320-token cap: 28 s windows, 1.5 s overlap. Cap never bound
  (max 142 words / clip; 320-token warn is 280).
- Seam stitch at overlap midpoint; 270 orig10 + 180 holdout seam
  events logged. Some are DAI beds Whisper also holes; some are
  overlap-speech drops (Wait Wait 370–400: 19 vs 81 Whisper words).
- Keyword bias can drop content (Wait Wait 1888–1903 empty under
  keywords; spotter miss). Measure ±bias before e2e.
- WER vs Whisper segment-align, not human gold. Decoder
  `word_timestamps=True` seek-drop is not the live baseline.
- Listen substitute: disagreement windows dumped as text (worker
  cannot play audio). Worst: Wait Wait overlapping speech, BBC
  Ferguson accent (WER 0.36), ATP letter-spell URLs.
- E2E confirm prompts miss the Whisper LLM cache.

## ASR vs Whisper (unbiased)

| set | WER | word ratio | hole_s W / Whistle | wall s / ep-hour | CPU s / ep-hour | spotter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10 | **0.133** | 0.992 | 285 / 479 | **705** (RTF 0.196) | 5520 | **21/21** |
| holdout | **0.137** | 0.984 | 556 / 752 | 873* | 6437* | **7/7** |

\*Holdout wall/CPU inflated by running beside orig10 e2e. Use orig10
705 s/eh as the Cake number.

Smoke 28 s Crime Junkie 184–212: Whisper base **2.39 s / RTF 0.085**
(“45 caliber”); Whistle warm **3.88 s / RTF 0.139** (“forty five
taller”). Whisper path (segment decode + align) is still faster.

Per-show orig10 WER: planet_money 0.077, crime_junkie 0.082,
up_first 0.090, serial 0.097, bbc_global 0.079, atp 0.112,
darknet 0.117, wtf 0.145, wait_wait 0.169, **bbc_football 0.357**.

Word-timestamp MAE vs Whisper (two-pointer, |dt|≤2 s): mean **0.27 s**,
median **0.18 s**, p90 **0.62 s**. E2e edge blow-up is quote-token
mismatch, not 80 ms frame jitter.

320-token cap: **0/1800** clips with ≥280 words.

## Keyword biasing (`SPONSOR_NAMES`)

| | WER | hole_s | spotter |
| --- | ---: | ---: | ---: |
| plain | 0.133 | 479 | **21/21** |
| keywords | 0.136 | 570 | **20/21** |

Miss: `npr_wait_wait` 1888.5–1903 (Whistle words empty; Whisper/plain
have “Wait Wait Don’t Tell Me” / Austin Tice sister-show). Do not
enable keywords.

## E2e (spot-then-confirm + Whistle words)

| set | system | catch | openings | not_ad | edge | calls/ep | Gemini $ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10 | live Whisper `1b7728d` | 97.93% | 7/7 | 0.00s | 23.21s | 8.9 | $0 |
| original 10 | Whistle plain | **89.71%** | 7/7 | 0.00s | **64.47s** | 8.7 | $0.093 |
| holdout | live Whisper | 98.71% | 3/3 | 0.00s | 10.35s | 11.25 | $0 |
| holdout | Whistle plain | 98.71% | 3/3 | 0.00s | 10.35s | 11.0 | $0.046 |

Orig10 gates: catch **FAIL**, openings ok, not_ad ok. Holdout labeled
windows matched live cuts (not a reason to fold after orig10 miss).

UTC 2026-10-09 ledger after this work: **$0.285** of $0.50.

## Live

Unchanged: detached `1b7728d`,
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`,
`ENABLE_SPOT_THEN_CONFIRM=true`, `ENABLE_WHISTLE_STT` unset.
Rollback is still `git checkout --detach e902c89` and unset
spot-then-confirm.

## Next

1. Accuracy: keep Whisper segment-align; Whistle is not a fold.
2. If revisiting tiny ASR, try it only as a spotter (21/21) while
   matching quotes on Whisper tokens.
3. Overlap-speech / accent shows (Wait Wait, Ferguson) are the WER
   tail, not the 30 s cap.
