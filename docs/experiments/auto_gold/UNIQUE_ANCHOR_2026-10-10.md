# Unique-anchor (2026-10-10)

Scout stays **false**. `general_podcast_ads` remains **LOCKED**. Local Whisper
only. Worktree `/home/jdr/cursor/repos/podly_unique_anchor` on
`cursor/unique-anchor-2026-10-10` (`45ed735`+). Live `podly.service` stays
detached `1b7728d`.

## Self-critique (before the run)

- **Prompt-only already failed twice** on Planet Money 1217–1311: sentence
  quotes (2026-10-07) and spot-v2 v3 (2026-10-10) both stopped at the first
  “listen on the NPR app” CTA (~1263). Telling the model the CTA is not unique
  is not enough; **verification** has to reject it. If the model cannot emit
  “wherever you get Fresh Air” after one re-ask, we fall back to first-best
  and recreate the same truncation whenever merge packs the stack into one
  window.
- **Uniqueness is excerpt-local.** A 50s pad can put four NPR CTAs in one
  confirm. A slightly unique closer that still fuzzy-matches another CTA at
  0.72 will re-ask then fall back. A too-long unique string that walks into
  show copy fails `not_ad`.
- **Unique-alone vs unique+merge.** Live `1b7728d` already catches 1217–1311
  with unmerged candidates. Unique-alone may match live without fixing
  anything. Unique+merge is the actual test: one window, unique end required.
- **Re-asks spend.** Remaining UTC Gemini was ~$0.27 of $0.50. v4 misses the
  v2/v3 cache. Unique-alone holdout was the likely skip if the cap tightened.
- **Leakage:** the Planet Money closer was read while drafting. The prompt
  example uses “wherever you get Show B”, not the gold Fresh Air line.

## Flag (default off)

| flag | purpose |
| --- | --- |
| `ENABLE_QUOTE_UNIQUE_ANCHORS` | start_anchor / end_anchor must be unique in the excerpt; one re-ask; then first-best fallback |

Compose with `ENABLE_SPOT_MERGE_CALLS` in the unique+merge ablation. Sentence /
richer / fallback-bound stay off.

## Fold gate vs live `1b7728d`

catch ≥ live, `not_ad` 0.00s, openings not worse, Gemini ≤ $0.50 UTC.
Rollback: `git checkout --detach 1b7728d` and keep
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true` `ENABLE_SPOT_THEN_CONFIRM=true`;
unset `ENABLE_QUOTE_UNIQUE_ANCHORS` (and merge if it was set).

## Result (did not fold)

UTC 2026-10-10 Gemini **$0.493** of $0.50 (this run **$0.268**). Live
`podly.service` remains detached `1b7728d` with scout off, Whisper local,
both live flags true. Unique-alone holdout skipped (cap).

### Metrics vs live `1b7728d`

| set | system | catch | openings | not_ad | edge | calls/ep | in tok/ep | out tok/ep | re-ask | Gemini $ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10 | live 1b7728d | 97.93% | 7/7 | 0.00s | 23.21s | 8.9 | 11339 | 347 | — | $0 |
| original 10 | merge-only (spot-v2) | 97.79% | 7/7 | 0.00s | 32.34s | **6.6** | **8760** | 244 | — | $0.075 |
| original 10 | unique | 97.68% | 7/7 | 0.00s | 21.56s | 9.9 | 15648 | 466 | 10.1% | $0.135 |
| original 10 | unique+merge | 97.42% | 7/7 | 0.00s | 21.67s | 7.1 | 11449 | 243 | 7.0% | $0.095 |
| holdout | live 1b7728d | 98.71% | 3/3 | 0.00s | 10.35s | 11.25 | 13808 | 434 | — | $0 |
| holdout | unique | — | — | — | — | — | — | — | — | skipped |
| holdout | unique+merge | 97.75% | 3/3 | 0.00s | 12.55s | **7.0** | 11081 | 312 | 0.0% | $0.038 |

ATP orig10 calls: live 17 → unique 23 (6 fallbacks) / unique+merge 16.
Wait Wait: 16 → unique 16 / unique+merge 10.

Planet Money stacked Ted Radio Hour + Fresh Air is **1217–1311** on unique
(1 re-ask, 0 fallback) and unique+merge (0 re-ask) — same as live, not the
sentence-variant **1217–1263**. The catch miss is elsewhere (ATP unmatched
fallbacks; Crime Junkie Freestyle still chunk-falls back 962–1021).

## Claim

Unique-anchor **does** bound the Planet Money stack through Fresh Air, even
in one merged window, but orig10 catch is 97.68% / 97.42% vs live 97.93%
(`not_ad` 0.00s, openings 7/7). Fold fails. Leave live.

## Live

Unchanged: detached `1b7728d`, `ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`,
`ENABLE_SPOT_THEN_CONFIRM=true`, scout off, local Whisper. No deploy.
Rollback is a no-op (never left `1b7728d`). If a later fold happens:
`git checkout --detach 1b7728d` and unset `ENABLE_QUOTE_UNIQUE_ANCHORS`.

## Next (accuracy → reliability → spend)

1. **Accuracy:** $0 grow when uniqueness fails: last unique n-gram inside
   the spotted cluster (do not first-best a repeated CTA, do not chunk-fallback
   60s of Freestyle). Re-score from the v4 LLM cache.
2. **Reliability:** 9 unique-alone fallbacks (ATP 6) are the catch leak.
   Re-ask listing is not enough when the model still emits a non-unique CTA.
3. **Spend:** unique+merge is the call win (Wait Wait 16→10, 7.1 calls/ep)
   once catch is recovered; unique-alone spends more than live (8.9→9.9).
