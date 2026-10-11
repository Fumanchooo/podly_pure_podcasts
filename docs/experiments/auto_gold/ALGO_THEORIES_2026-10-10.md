# Algo theories (2026-10-10)

Scout stays **false**. `general_podcast_ads` remains **LOCKED**. Local Whisper
only. Worktree `/home/jdr/cursor/repos/podly_algo_theories` on
`cursor/algo-theories-2026-10-10`. Live `podly.service` stays detached `1b7728d`
until a fold.

## Per-failure (live `1b7728d`, original 10)

Remaining miss is **9.90s undercut / 474.1s inner = 2.07%**, not missed ads.
Spotter still 21/21. Edge 23.21s = 9.90s under + 13.31s over. Openings 7/7,
`not_ad` 0.00s. Chunk fallbacks: ATP 3, Crime Junkie 1 (Freestyle 962–1021),
Up First 1, Serial 1.

| window | under | over | mechanism |
| --- | ---: | ---: | --- |
| Up First 0–26.5 membership | 2.53s | 1.08s | end snaps to 1.91s pause after “thank you.”; gold inner continues ~2.5s into “A flight bound for Israel” (news). Stacked promo start 71.42 vs 72.5 eats the show handoff. |
| Wait Wait open 0–25.7 | 1.80s | 0 | start 0.54 (silence snap later); end 24.44 before “From NPR and WBEZ”. |
| Wait Wait 1888.5 Austin Tice | 1.75s | 0 | quote starts 1890.25, 2.69s pause after “NPO.” |
| Planet Money 1216 stack | 1.08s | 4.30s | 1s late start; 4.3s “over” is the DAI bed into the next labeled window (1217–1311 still covers Ted Radio Hour + Fresh Air). |
| Planet Money open 0–25 | 0.87s | 0 | cut ends at donate.npr.org; gold includes “This is Planet Money”. |
| WTF open 0–44.5 | 0 | 2.86s | cut 0–47.36 includes “Okay, let's do the show.” |
| Darknet Doppel 581–593 | 0 | 2.33s | cut to 595.33 includes “Okay, this is the epic story” after a 3.37s pause. |
| plus <0.6s bits | ~1.7s | ~2.7s | CJ/Serial/Wait Wait/ThreatLocker openings |

Crime Junkie Freestyle 962–1021 is a **chunk fallback that covers gold 963–991**;
window-local overcut is 0 because 991–1021 sits outside that gold window (still
Freestyle CTA). ATP 144–171 / 310–347 are **caught on live** via fallback; unique
/ first-last leaked them.

## Hypotheses (before the run)

1. **H1 `ENABLE_QUOTE_RESUME_TRIM`** — trim the last sentence if it is a host
   resume (`okay let's do the show` / `okay this is the epic…`), not a network
   credit. **Catch:** hold. **Overcut:** −~5s (WTF + Doppel). **Spend:** $0.
   **Fail:** “This is NPR/Planet Money” credit trimmed (must not; separate regex).
2. **H2 `ENABLE_QUOTE_BACKFILL_LEADIN`** — grow start backward ≤4s to a ≥0.8s
   pause (stop at resume). **Catch:** +Wait Wait 1888 / tiny PM stack start.
   **Fail:** pull into pre-ad chat (4s + pause cap; first/last had no cap).
3. **H3 `ENABLE_QUOTE_CLOSER_TAIL`** — after a CTA URL, include a short
   `This is NPR/Planet Money` sentence if the gap is <0.8s and ≤4s. **Catch:**
   +PM open 0.87s. **Fail:** crossing the Up First “thank you.” 1.91s pause into
   news (pause stop).
4. **H4 `ENABLE_SPOT_WEAK_DROP`** — drop company-name-only candidates (spot-v2
   weak-drop without 70s merge). **Spend:** ATP/Wait Wait calls down. **Catch:**
   hold if every gold ad has a cue/URL. **Fail:** brand-only host-read.
5. **H5 `ENABLE_SNAP_EARLIER_START`** — silence-snap **starts** only earlier,
   never later; ends stay nearest. **Catch:** Wait Wait 0.54→0. **Fail:** inward
   -only already lost catch; this is start-only, not both edges.

## Self-critique (before the run)

- Orig10 and holdout are **seen**. These windows were read while drafting.
  A pass is a no-regression check, not unseen-sponsor proof. Fresh Cake
  episodes (Planet Money / Indicator / Prof G) only if remaining Gemini
  allows a judge pass; matcher flags themselves are $0 cache replay.
- H1+H3 are opposite levers (trim vs extend). Discriminator is resume vs
  network credit. If the regex leaks, overcut or undercut moves the wrong way.
- H2 is **not** first/last-in-window (that overcut 13s→33–41s). It is
  pause-bounded 4s.
- H4 was bundled with merge in spot-v2 (97.79% catch, 32s overcut). Alone it
  should not widen windows.
- H5 is not inward-only (eval lost catch). End snap stays nearest.
- Sentence quotes / unique / first-last / Whistle / 70s merge are **not**
  re-run. Strong-core fallback skipped: live ATP 144–171/310–347 depend on
  cluster fallback.
- `--require-llm-cache`; stop rather than spend on a miss.

## Fold gate vs live `1b7728d`

catch ≥ live, `not_ad` 0.00s, openings not worse, overcut not worse, Gemini
≤ $0.50 UTC. Rollback: `git checkout --detach 1b7728d` and keep
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true` `ENABLE_SPOT_THEN_CONFIRM=true`; unset
the five new flags.

## Result

(filled after scoring)

## Live

Unchanged unless a variant folds.

## Next (accuracy → reliability → spend)

(filled after scoring)
