# Quote-anchor live deploy gate (2026-10-07)

Scout confirm stays **false**. `general_podcast_ads` remains **LOCKED**.

Original-10 gates **pass**. Live `podly.service` is switched to this experiment
commit with `ENABLE_QUOTE_ANCHOR_BOUNDARIES=true` in `.env.local` (code default
stays false). Rollback identity is still `20ed084` if the flag is unset and
the live checkout is detached there again.

Structural path: LLM sees **segment** Whisper text (no timestamps). Words are
matcher-only. 8–12s DAI-bed pull is **off**. Holdout is **now-seen**.

Word timings are **segment-forced-align** (DTW on trusted segment text, lexicon
locked to those tokens). `word_timestamps=True` decode is not used.

## Scorecard

| set | system | inner catch | openings | not_ad | edge | input tok/ep |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original 10 | baseline | 96.52% | 7/7 | 2.72s | 26.43s | 23526 |
| original 10 | quote-anchor (structural, old words) | 95.09% | 7/7 | 0.00s | 32.34s | 22578 |
| original 10 | quote-anchor (aligned words, pre-fix) | 96.22% | 7/7 | 28.00s | 60.11s | 22578 |
| original 10 | quote-anchor (aligned words, this commit) | **97.68%** | **7/7** | **0.00s** | **23.31s** | **22578** |
| holdout now-seen, corrected label | baseline | 98.93% | 3/3 | 0.00s | 22.38s | 29174 |
| holdout now-seen, corrected label | quote-anchor (this commit) | 98.71% | 3/3 | 0.00s | 10.35s | 30049 |

Original-10 gate (catch ≥96.5%, openings 7/7, not_ad ≤2.7s, edge ≤26.4s, tokens ≤ baseline): **pass**. 41/41 quotes word-matched; no segment/chunk fallback. Gemini **$0**.

Holdout (now-seen): **pass** (−0.22 pp catch). 21/21 quotes word-matched.

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

Remaining holes are music beds (Up First 865–887, Darknet 54–63). Cuts still
cover them via 12s hole-pull / CTA silence extend; we do not invent words in
the bed. Classify-time reanchor only clips real silences and packs a fully
stretched leading bed (Threat Locker 465→475).

Align wall (Cake CPU, `base`, no second decode): **25–252s/episode** (mean **114s**).

## Causes (aligned-word regression 0→28s not_ad / 32→60s edge)

| show / span | what happened | class |
| --- | --- | --- |
| crime_junkie 2026–2054 (28s not_ad) + 1964–1992 overcut | Shopify end quote `That's Shopify.com…` missed; chunk_fallback 1930–2147 ate show | (c) matcher used only first subtoken of `Shopify.com` |
| darknet Stay tuned | `S-I-N-C-H` / `SINCH.com` end miss → chunk 2465–2776 | (c) same |
| darknet NetSuite / Threat Locker | `NetSuite.com-darknet` / `D-O-P-P-E-L` end miss → segment_fallback | (c); Threat Locker start also (b) linear stretch 465–478 |
| npr_wait_wait 703–731, planet_money 1216 | forward-trim missed; interpol filled ≥0.8s pauses | (d) |
| serial 431 vs 458 | recovered PeakSaver speech; gold inner still 458+ | (b)/(c) more words |
| crime_junkie / wtf openings | small end stretch | (a) |

## Fixes ($0 replay)

1. Flatten every alphanumeric piece in a word so `Shopify.com` and `S-I-N-C-H` match the quote.
2. Re-anchor: clip words off ≥0.8s silence (restore Wait Wait / Planet Money pauses); pack only a fully stretched leading bed.
3. Start-only quote: if the end quote misses, keep the start hit and grow through adjacent segments (≤0.5s gap, ≤10s) so Planet Money 1922–1930 Sources & Methods is not a 4s stub.

## Spend / flags

Gemini **$0** this step (cached LLM). Code default **false**; live env **true**.
Scout **false**.

## Verdict

Enable `ENABLE_QUOTE_ANCHOR_BOUNDARIES` on live via `.env.local`. Detach
`podly.service` to this experiment commit. Rollback: unset the env flag and
`git checkout --detach 20ed084`.
`general_podcast_ads` remains **LOCKED**.
