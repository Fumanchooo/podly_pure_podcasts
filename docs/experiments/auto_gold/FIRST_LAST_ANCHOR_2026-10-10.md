# First/last-anchor (2026-10-10)

Scout stays **false**. `general_podcast_ads` remains **LOCKED**. Local Whisper
only. Worktree `/home/jdr/cursor/repos/podly_first_last_anchor` on
`cursor/first-last-anchor-2026-10-10`. Live `podly.service` stays detached
`1b7728d`. Gemini UTC-day ledger was **$0.493 / $0.50**; this run made **no**
live Gemini calls.

## Self-critique (before the run)

- Unique-anchor failed catch because re-asks then first-best fallback. Dropping
  uniqueness and snapping **start=first / end=last** inside the spotted window
  should grow stacked CTAs (Planet Money 1263→1311) from cached v2/v4 replies
  without spend.
- Last-in-window can swallow show if the same phrase appears in content inside
  the spotted/merged cluster. Report `not_ad` (labeled) **and** overcut.
- ATP unique fallbacks and Crime Junkie Freestyle 962–1021 are quote **misses**.
  First/last cannot invent a match; expect those to stay fallback unless the
  cached reply already had a hittable string.
- Live/v4 Planet Money is already 1217–1311. This ablation is a matcher change
  on those replies, not a new stacked-break prompt.

## Flag (default off)

| flag | purpose |
| --- | --- |
| `ENABLE_QUOTE_FIRST_LAST_ANCHORS` | start quote/anchor → first hit; end quote/anchor → last hit; inside spotted/merged window; skip uniqueness re-asks |

Compose with `ENABLE_SPOT_MERGE_CALLS` and (for v4 replies)
`ENABLE_QUOTE_UNIQUE_ANCHORS` (re-asks off when first/last is on). Sentence /
richer / fallback-bound stay off.

## Fold gate vs live `1b7728d`

catch ≥ live, `not_ad` 0.00s, openings not worse, Gemini $0 this experiment.
Rollback: `git checkout --detach 1b7728d` and keep
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true` `ENABLE_SPOT_THEN_CONFIRM=true`;
unset `ENABLE_QUOTE_FIRST_LAST_ANCHORS` (and unique/merge if they were set).

## Result (did not fold)

UTC 2026-10-10 Gemini **$0.493** of $0.50 (this run **$0**). Live
`podly.service` remains detached `1b7728d` with scout off, Whisper local,
both live flags true. Holdout unique-alone and holdout merge-only skipped (no
cached confirms).

### Metrics vs live `1b7728d`

| set | system | catch | openings | not_ad | edge | calls/ep | in tok/ep | out tok/ep | Gemini $ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10 | live 1b7728d | 97.93% | 7/7 | 0.00s | 23.21s | 8.9 | 11339 | 347 | $0 |
| original 10 | merge-only (spot-v2) | 97.79% | 7/7 | 0.00s | 32.34s | **6.6** | **8760** | 244 | $0.075 |
| original 10 | unique | 97.68% | 7/7 | 0.00s | 21.56s | 9.9 | 15648 | 466 | $0.135 |
| original 10 | unique+merge | 97.42% | 7/7 | 0.00s | 21.67s | 7.1 | 11449 | 243 | $0.095 |
| original 10 | first/last | 96.56% | 7/7 | 0.00s | 56.67s | 8.4 | 10632 | 320 | **$0** |
| original 10 | first/last+merge | 95.89% | 7/7 | 0.00s | 60.77s | 6.4 | 8457 | 235 | **$0** |
| original 10 | first/last+unique | 96.30% | 7/7 | 0.00s | 55.03s | 8.4 | 12976 | 370 | **$0** |
| original 10 | first/last+unique+merge | 95.52% | 7/7 | 0.00s | 54.52s | 6.4 | 10217 | 206 | **$0** |
| holdout | live 1b7728d | 98.71% | 3/3 | 0.00s | 10.35s | 11.25 | 13808 | 434 | $0 |
| holdout | unique+merge | 97.75% | 3/3 | 0.00s | 12.55s | 7.0 | 11081 | 312 | $0.038 |
| holdout | first/last | 93.24% | 3/3 | 0.00s | 40.28s | 10.75 | 13150 | 395 | **$0** |
| holdout | first/last+unique+merge | 92.16% | 3/3 | 0.00s | 24.97s | 7.0 | 11081 | 312 | **$0** |

Labeled `not_ad` stayed **0.00s**. Overcut (cut past gold inner span) is the
real swallow: orig10 13.3s live → 33–41s first/last. Catch fails the gate
on every ablation.

### Cases

**Planet Money stacked NPR 1217–1311.** Live, merge, unique, unique+merge, and
all four first/last ablations cut **1217.1–1311.6**. First/last did not
recreate the sentence-variant 1263 stop; these cached v2/v4 replies already
bound the stack. No win here.

**ATP unique fallbacks.** Unique had 6 re-asks / 6 fallbacks and dropped live’s
310.7–347.0. First/last skipped re-asks (ATP 23→16 calls) but did **not**
recover the fallbacks: word hits collapsed (4→1), chunk fallbacks 3→5, and
cuts lost 144.4–171.3 and 310.7–347.0. Last-in-window stretched the closer
6211–6242 → **6215–6277**.

**Crime Junkie Freestyle 962–1021.** Still chunk-fallback on every system
(live through first/last). First/last also split the later midroll (live
1884–1969 → 1864–1887 + 1928–2006) and nicked the opening (0–32 → 10–34).
Word hits 4→1.

## Claim

First/last snap on cached v2/v4 replies keeps Planet Money 1217–1311 and
labeled `not_ad` at 0.00s, but orig10 catch is 96.56%/95.89%/96.30%/95.52% vs
live 97.93% with overcut 13s→33–41s (ATP/Crime Junkie fallbacks not saved).
Fold fails. Leave live.

## Live

Unchanged: detached `1b7728d`, `ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`,
`ENABLE_SPOT_THEN_CONFIRM=true`, scout off, local Whisper. No deploy.
Rollback is a no-op (never left `1b7728d`). If a later fold happens:
`git checkout --detach 1b7728d` and unset `ENABLE_QUOTE_FIRST_LAST_ANCHORS`.

## Next (accuracy → reliability → spend)

1. **Accuracy:** last-occurrence must not walk into post-ad copy that sits
   inside the spotted cluster (ATP closer 6242→6277; Crime Junkie split
   midroll). Prefer last hit **before** a content gap, not last in the
   whole candidate.
2. **Reliability:** unmatched quotes still chunk-fallback (Freestyle 962–1021,
   ATP 6 unique fallbacks). First/last is a matcher for hits, not a miss
   repair. Grow-from-spot or a tighter fallback bound is the next lever.
3. **Spend:** merge still cuts calls (Wait Wait 16→10, 6.4/ep) once catch is
   recovered; first/last itself is $0 matcher-side and dropped unique re-asks.
