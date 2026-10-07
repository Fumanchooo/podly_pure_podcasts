# Start-only backward pack (2026-10-07 weekday loop)

Scout confirm stays **false**. `general_podcast_ads` remains **LOCKED**.
No gold/label edits. Gemini **$0**. Groq unused (local Whisper).

## Health (before this change)

Live `podly.service` was `cursor/quote-anchor-2026-10-05@f252799`, detached
in `/home/jdr/cursor/repos/podly_pure_podcasts`, with
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`. Writer + web were active. LAN
`podly.jdrbc.ca` → `:5001`. Three episodes processed after last night’s fold
all completed with cuts (not silent 0-ad):

| feed | title | quotes | note |
| --- | --- | ---: | --- |
| Planet Money | Charles Ponzi's scheme | 6 | one start forward-trimmed 1133→1145 (`After the break`) |
| Indicator | One economist's answer to data center anxiety | 3 | 2.2s EOF stub `Head comes from NPR sponsor AARP`; align interpolated group 307.1–332.6 |
| Prof G Markets | Why The S&P Just Hit A Record High | 4 | preroll snapped to 0 |

## Hypothesis

Start-only quotes in a leftover 1-segment EOF chunk cannot see packed
sister-show audio in the previous chunk. Grow **backward** on the full
episode segment list (gap ≤0.5s, span ≤25s), stop before the prior match
end. Forward pack stays 10s.

Not in this fold: DAI-bed 8–12s flag, short-hole pull <8s, Indicator
holdout 370.785–398.785 (gold is **not_ad** voicemail — label/policy
hypothesis only), empty-`sane` align `min()` crash.

## Gates (vs live f252799)

| set | catch | openings | not_ad | edge | tokens/ep | Gemini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10 live | 97.68% | 7/7 | 0.00s | 23.31s | 22578 | $0 |
| original 10 candidate | **97.68%** | **7/7** | **0.00s** | **23.31s** | **22578** | **$0** |
| holdout live | 98.71% | 3/3 | 0.00s | 10.35s | 30049 | $0 |
| holdout candidate | **98.71%** | **3/3** | **0.00s** | **10.35s** | — | **$0** |

Original-10 and holdout **cut-list identity** vs `pipeline-segment-matched2`
/ `quote-anchor-segment-matched`. Holdout 370.785–398.785 covered **0.00s**.

Unit: leftover EOF AARP chunk grows to the packed Your Turn block
(654.4–675.0) and stops before a prior Up First match.

## Claim

On the frozen original 10 and now-seen holdout this matcher change is a
**no-op**. It is allowed to say that a start-only leftover EOF bumper plus
a packed preceding sister-show (the 2026-10-07 live Indicator fixture) is
cut as one span in unit tests. It does **not** change catch/edge on the
scored 14 episodes, does not relabel Indicator ~6:11, and does not fix
the align empty-`sane` crash.

## Live

Folded: detach `podly.service` to this commit, flag unchanged
(`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`). Today’s Indicator file is not
reprocessed. Rollback this increment: `git checkout --detach f252799`
(keep the env flag). Full quote-anchor rollback is still unset the flag
and `git checkout --detach 20ed084`.

## Next (accuracy → reliability → spend)

1. **Accuracy / labels:** Indicator holdout 370.785–398.785 is frozen
   `not_ad` (listener voicemail praising Throughline). If that should count
   as sister-show underwriting, propose a gold re-judge — do not silently
   flip the label.
2. **Reliability:** `lexicon_words_for_segment` `min()` on empty `sane`
   interpolates a whole ~25s align group (live Indicator 307.1–332.6).
   Fixture that group; do not default to full-segment linear across a bed.
3. **Accuracy, $0:** leftover last LLM chunk (1 segment) is the root of
   the AARP stub. Merging a tiny remainder into the previous chunk needs
   new prompts (Gemini, not $0). Matcher grow is the $0 stand-in.
