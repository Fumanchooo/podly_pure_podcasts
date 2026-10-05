# Opening FN 2026-10-05 (policy fold)

User policy (2026-10-05): self-promo, network/sister-show promos, first-party
app plugs, and membership asks **count as ads** anywhere in the episode.
Scout confirm **OFF**. Live `podly.service` untouched. `general_podcast_ads`
remains **LOCKED**.

## Scorecard (same script, 10/12 shows)

Denominators change after relabel (Planet Money 0–92s and Up First 868–896s
became is_ad; Bill Simmons 0–92s flipped but is not in the scored 10).

| metric | Friday baseline | this morning H1 (preroll-only keep) | now (policy folded + relabel + 12s hole cap) |
| --- | ---: | ---: | ---: |
| is_ad seconds caught (candidate) | 28.2% (254.1 / 900) | 49.7% (447.3 / 900) | 45.7% (466.5 / 1020) |
| is_ad seconds caught (inner ad_span) | 57.7% (242.8 / 421) | 88.8% (373.7 / 421) | **96.2% (456.2 / 474.1)** |
| is_ad windows ≥50% candidate | 5/17 | 6/17 | 6/19 |
| opening seconds caught (∩ 0–90s) | 18.1% (97.7 / 540) | 54.1% (292.3 / 540) | 39.6% (249.7 / 630) |
| opening windows ≥50% candidate | 0/6 | 2/6 | 0/7 |
| opening windows ≥50% inner ad_span | 2/6 | 6/6 | **7/7** |
| not_ad seconds cut (FP) | 2.7 / 1224 | 27.7 / 1224 | **2.7 / 1104** |
| label flips (not_ad → is_ad) | — | — | **3/3** |
| Gemini spend | $0.4924 | $0.00 | **$0.0027** |

Candidate ≥50% and opening-seconds % dropped vs H1 because H1 glued Up First
0–99s through the headlines. Inner-span recall is the honest ad metric.

## Label flips (re-judge only `not_ad` + `educational/self_promo`)

| window | old | new | new inner span |
| --- | --- | --- | --- |
| npr_up_first 868–896 (Sources & Methods) | not_ad | is_ad | 868–896 |
| planet_money 0–92 (NPR membership) | not_ad | is_ad | 0–25 |
| bill_simmons 0–92 (Ringer network; not in scored 10) | not_ad | is_ad | 0–44.5 |

Other labels copied unchanged. $0.0027 Gemini.

## What changed in code

1. **`_adjust_confidence`:** do not demote `educational/self_promo`. A 1.00
   self_promo segment stays 1.00 and is cut. `technical_discussion` still −0.25.
   Removed the eval-only preroll flag.
2. **Gold judge prompt:** membership / network / sister-show / first-party
   plugs are is_ad; taxonomy may still say `educational/self_promo`.
3. **Merge hole cap 12s:** `min(60s separation, 12s content gap)` so two ads
   49s apart are not glued through show content. End-of-episode 60s extend
   unchanged. Up First is now 0–24s and 73–148s (membership + stacked NPR
   Politics / Embedded promos); headlines 24–73s stay.

## Verdict

**Folded on this branch** (not deployed to live `podly.service`). Inner-span
opening 7/7; not_ad FP back to Friday’s 2.7s; inner is_ad 96.2%. Candidate
90s-window ≥50% remains a poor fit when gold windows include show open.

## Claim

`general_podcast_ads` remains **LOCKED** (silver Gemini judge votes only; no human spot-check).
