# Full-sentence quotes + leftover-chunk merge (2026-10-07)

Scout confirm stays **false**. `general_podcast_ads` remains **LOCKED**.
No gold/label edits. Groq unused (local Whisper). Worktree:
`/home/jdr/cursor/repos/podly_fullsent` on `cursor/fullsent-quotes-2026-10-07`.
Live checkout stays detached `e902c89` until a fold.

## Self-critique (before locking)

- **Holdout leakage:** the holdout is now-seen. A pass here is a no-regression
  check against live `e902c89`, not a claim that the prompt generalizes.
- **Attribution:** prompt v3, leftover merge, and 2-segment overlap land in one
  candidate. Gains cannot be split. v3 is a full LLM cache miss (~14 episodes).
- **Overclaim:** live already word-matches 41/41 original-10 quotes. Full
  sentences help when the model currently quotes a 5–8 word core that misses
  the lead-in/outro, and leftover merge is the real AARP-stub fix (live
  Indicator fixture, not in the frozen 10).
- **not_ad risk:** longer quotes + overlap union can overcut. Fail-closed
  chunk fallback is still last resort; short-quote fallback must fire first.
- **Heuristics snooping:** flags are scored on the same gold. They are
  inspection-only and must not change cuts. Recommendations are “try later”,
  not “will raise catch”.
- **Voice/repeats:** one frozen episode per show. Cross-episode n-grams only
  fire for shared publishers (NPR/BBC). Loudness is RMS, not diarization.
- **Spend:** stale ledger date `2026-10-05` ($0.316) is a new UTC day today
  (2026-10-07). Cap is $0.50. If original-10 already fails gates, skip
  holdout Gemini.

## Flags

`ENABLE_QUOTE_FULL_SENTENCE_QUOTES` default **false**. Code path unchanged
when off (prompt_v2, no leftover merge, no overlap). Live env stays
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true` only unless this experiment folds.

## Gates vs live e902c89

Fold only if catch ≥ live, not_ad stays 0.00s, openings not worse, spend
≤ $0.50 UTC. Rollback if folded: `git checkout --detach e902c89` (keep
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`; unset the full-sentence flag).
