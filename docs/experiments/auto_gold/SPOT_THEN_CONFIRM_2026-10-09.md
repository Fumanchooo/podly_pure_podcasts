# Spot-then-confirm (2026-10-09)

Scout confirm stays **false**. `general_podcast_ads` remains **LOCKED**.
Code default `ENABLE_SPOT_THEN_CONFIRM=false`. Live env **true** after fold.

Oct 8 ~9:36 AM PT Daily Podly morning loop never spawned a cloud agent
(no bcId in 16:00–17:30 UTC). Live `podly.service` was healthy throughout
(detached `e902c89`, disk 48%). Not a Cake crash; left alone.

## Plan critique (before the run)

- Missed ads are unrecoverable if the $0 spotter never fires.
- Mining original-10 gold + cached quotes into the lexicon overfits
  recall on seen data. Holdout names (`the zone`, FanDuel, Cornerpiece,
  TAL partner) were also read while drafting the lexicon.
- Broad cues (`listen to`, `your support`) create extra candidates;
  confirm must reject them or `not_ad` blows up.
- Fail-closed chunk-fallback on a padded window would cut ~100s of show;
  fallback is clamped to the spotted cluster ±2s.
- New prompts miss the old chunk LLM cache (this step spends Gemini).

## Architecture

`$0` keyword/sponsor/URL/ngram spotter on aligned words → cluster hits
(gap 40s) → one Gemini confirm per candidate with 50s pad, one edge
extend. Reuses quote-anchor matcher + lexical_v2 + ±2s silence snap.

## Scorecard vs live `e902c89`

| set | system | catch | openings | not_ad | edge | calls/ep | in tok/ep | out tok/ep | Gemini $ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10 | live e902c89 | 97.68% | 7/7 | 0.00s | 23.31s | 14.2 | 22578 | 314 | $0 |
| original 10 | spot-then-confirm | **97.93%** | **7/7** | **0.00s** | **23.21s** | **8.9** | **11339** | 347 | **$0.098** |
| holdout now-seen | live e902c89 | 98.71% | 3/3 | 0.00s | 10.35s | 20.25 | 30049 | 426 | $0 |
| holdout now-seen | spot-then-confirm | **98.71%** | **3/3** | **0.00s** | **10.35s** | **11.25** | **13808** | 434 | **$0.048** |

Spotter recall: **21/21** original-10 gold ads, **7/7** holdout. No
unrecoverable misses on these sets.

UTC ledger 2026-10-09: **$0.146** of $0.50.

## Claim

On the frozen original 10 this flag cuts more inner-ad seconds than live
at half the input tokens with `not_ad` 0.00s and openings 7/7. Holdout
labeled windows match live catch/edge. Spotter recall was tuned on seen
gold text (original 10 + holdout names read while drafting); do not
treat 100% spotter recall as unseen-sponsor proof.

## Live

Folded: detach `podly.service` to this commit. Keep
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true` and set
`ENABLE_SPOT_THEN_CONFIRM=true`. Rollback this increment:
`git checkout --detach e902c89` and unset `ENABLE_SPOT_THEN_CONFIRM`
(keep the quote-anchor flag). Full quote-anchor rollback remains unset
both flags and `git checkout --detach 20ed084`.

## Next (accuracy → reliability → spend)

1. **Accuracy:** drop holdout-seen names from the lexicon and re-score
   spotter recall; add a never-seen sponsor smoke set.
2. **Reliability:** clamp unmatched-quote fallback harder (one
   `donate.npr.org` chunk-fallback 71–155s); empty-`sane` align `min()`.
3. **Spend:** cap candidates/ep (ATP/Wait Wait still 15–16 calls);
   tighten `listen to` / company-name FPs.
