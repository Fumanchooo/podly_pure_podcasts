# Quote-anchor live deploy gate (2026-10-07)

`enable_quote_anchor_boundaries` stays **false**. Scout confirm stays **false**.
`general_podcast_ads` remains **LOCKED**. Live `podly.service` left at `20ed084`.

Structural path: LLM sees **segment** Whisper text (no timestamps). Words are
matcher-only. 8–12s DAI-bed pull is **off**. Holdout is **now-seen**.

Word timings are **segment-forced-align** (DTW on trusted segment text, lexicon
locked to those tokens). `word_timestamps=True` decode is not used.

## Scorecard

| set | system | inner catch | openings | not_ad | edge | input tok/ep |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original 10 | baseline | 96.52% | 7/7 | 2.72s | 26.43s | 23526 |
| original 10 | quote-anchor (structural, old words) | 95.09% | 7/7 | 0.00s | 32.34s | 22578 |
| original 10 | quote-anchor (structural, aligned words) | 96.22% | 7/7 | 28.00s | 60.11s | 22578 |
| holdout now-seen, corrected label | baseline | 98.93% | 3/3 | 0.00s | 22.38s | 29174 |
| holdout now-seen, corrected label | quote-anchor (structural, aligned words) | 98.71% | 3/3 | 0.00s | 10.35s | 30049 |

Original-10 gate (catch ≥96.5%, openings 7/7, not_ad ≤2.7s, edge ≤26.4s, tokens ≤ baseline): **fail**. Aligned words raise inner to 96.22% (openings 7/7, tokens OK) but blow not_ad (28s) and edge (60s).

Holdout (now-seen): **pass** (−0.22 pp catch). Indicator Wise is now a **word** match (6/6 quotes word; was 5 word + 1 segment_fallback). Inner unchanged because segment-edge already covered 422.

## Word-drop audit (14 shows)

Holes ≥5s where segment Whisper time-ranges have speech and the word stream has none:

| | holes | segment-speech in holes | labeled-ad overlaps | labeled-ad audio |
| --- | ---: | ---: | ---: | ---: |
| before (`word_timestamps=True` decode) | 98 | 646.0s | 6 | 34.43s |
| after (segment align + lexicon lock) | 40 | 200.5s | 4 | 25.39s |
| before original 10 | 46 | 313.3s | 3 | 27.19s |
| after original 10 | 17 | 89.7s | 3 | 24.75s |
| before holdout | 52 | 332.7s | 3 | 7.24s |
| after holdout | 23 | 110.8s | 1 | 0.64s |

Decode-drop (missing **text**) is fixed. Indicator 401–421: 0 → 52 words (`This message comes from Wise, …`). Remaining holes are **music beds inside sloppy segment timestamps** (Up First 865–887, Darknet 54–63): the next sentence’s words are timed to actual speech (~887 / ~62.7), while segment start includes the bed.

Align wall (Cake CPU, `base`, no second decode): **25–252s/episode** (mean **114s**). Decoder `word_timestamps=True` is a full extra transcribe and is slower; clips were ~2× segment decode.

## Root cause

openai-whisper `transcribe(..., word_timestamps=True)` **rewinds `seek` to the last DTW word** in each 30s window (`whisper/transcribe.py`). On long episodes that skips DAI/music after a pause. Isolated 55s clips often still hear the ad; full Indicator / Up First caches do not. Not VAD (`hallucination_silence_threshold` is None), not ffmpeg, not model size.

## Critique (not shipped)

- **faster-whisper**: would avoid the openai-whisper seek jump, but is a new CTranslate2 stack and a second full transcription unless it replaces production Whisper. `vad_filter` must stay off or DAI beds vanish.
- **larger model** (`small`/`medium`): same seek bug, 2–5× Cake-CPU cost.
- **Raw DTW labels**: recovered Wise but glued tokens (`Let'sjoin`) and original-10 inner fell to 89% / openings 6/7.

## Spend / flags

Gemini **$0** this step (cached LLM). Flag **false**. Live **20ed084**.

## Verdict

Do **not** enable `ENABLE_QUOTE_ANCHOR_BOUNDARIES` on live. Rollback identity unchanged: `20ed084`.
`general_podcast_ads` remains **LOCKED**.
