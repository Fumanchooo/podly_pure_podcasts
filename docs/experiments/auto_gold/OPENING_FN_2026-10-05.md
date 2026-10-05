# Opening FN 2026-10-05

Eval-only attack on Friday’s production-vs-auto-gold opening/preroll misses.
Scout confirm stays **OFF**. Live `podly.service` untouched. Labels frozen.
`general_podcast_ads` remains **LOCKED**.

## Scorecard (same script + labels, 10/12 shows)

| metric | Friday baseline | today (H1 replay) |
| --- | ---: | ---: |
| is_ad seconds caught (candidate window) | 28.2% (254.1s / 900.0s) | 49.7% (447.3s / 900.0s) |
| is_ad seconds caught (inner ad_span) | 57.7% (242.8s / 421.0s) | 88.8% (373.7s / 421.0s) |
| is_ad windows ≥50% candidate | 29.4% (5/17) | 35.3% (6/17) |
| opening seconds caught (∩ t=0–90s) | 18.1% (97.7s / 540.0s) | 54.1% (292.3s / 540.0s) |
| opening windows ≥50% candidate | 0/6 | 2/6 |
| opening windows ≥50% inner ad_span | 2/6 | **6/6** |
| not_ad seconds cut (FP) | 2.7s / 1224.0s | 27.7s / 1224.0s |
| Gemini spend | $0.4924 | **$0.00** (cached ModelCalls) |

Friday `docs/experiments/auto_gold/SCORECARD.md` is unchanged. Today’s raw
score is under gitignored `docs/experiments/auto_gold/runs/opening-fn-2026-10-05/`.

## Diagnosis ($0, cached sqlite + transcripts)

t=0 transcript segments exist on all six opening `is_ad` windows. The first
chunk (seq 0–59) was classified. The LLM **did** emit ad_segments:

| show | first-chunk `content_type` | min conf in 0–90s | identifications t<100s | Friday cut |
| --- | --- | ---: | ---: | --- |
| npr_up_first | educational/self_promo | 0.85 | 0 | empty |
| npr_wait_wait | educational/self_promo | 0.85 | 0 | starts 726s |
| serial | educational/self_promo | 0.95 | 0 | starts 431s |
| crime_junkie | educational/self_promo | 0.88 | 0 | starts 882s |
| wtf_maron | promotional_external | 0.95 | 11 | 0–43.7s |
| darknet_diaries | promotional_external | 0.99 | 16 | 0–54s |

Code path: `AdClassifier._create_identifications` → `_adjust_confidence`
(`ad_classifier.py`) subtracts **0.25** for `educational/self_promo`. Config:
`OUTPUT_MIN_CONFIDENCE = 0.8` (`shared/defaults.py`). Even confidence 1.0
becomes 0.75, so self-promo is a **hard drop**, not a soft demotion. Prompt
(`src/system_prompt.txt` rule 3) tells the model to use that taxonomy.

WTF/Darknet are true external sponsor prerolls and were already cut to the
inner `ad_span`. They fail the 90s-candidate ≥50% bar because the candidate
window still contains show open after the ad (metric, not a silent miss).

## Plan (H1) — ran

1. **H1 (high confidence, opening recall / no silent preroll miss):** eval-only
   flag `enable_keep_preroll_self_promo` skips the 0.25 demotion for
   `educational/self_promo` when `segment_offset ∈ [0, 90]`. Replay Friday’s
   cached `ModelCall` rows ($0 Gemini). Do not lower `min_confidence`. Do not
   retag labels. Do not add shows.
2. **H2 (dominated, not run):** rewrite the prompt so network sister-show promo
   is `promotional_external`. Crime Junkie was already predicted as ads; it died
   on demotion. Needs new LLM spend.

## Critique (recorded before run; still holds)

- **Overfitting to these 6 windows.** Planet Money’s opening is the same NPR
  membership pitch as Up First, but the frozen judge labeled it `is_ad=false`.
  H1 cut 25.0s of that `not_ad` window (the entire FP delta).
- **Label circularity.** Judge and classifier are both Gemini; the judge marked
  `educational/self_promo` as `is_ad=true` on Up First / Wait Wait / Serial and
  `is_ad=false` on Planet Money.
- **Policy (ask the user, do not silently decide):** should NPR membership,
  NPR Politics cross-promo, NYT app self-promo, and Audiochuck sister-show
  (Park Predators) count as ads to cut? Production prompt currently says no.
- **FP risk on not_ad openings.** ATP/BBC first chunks returned empty
  `ad_segments` and stayed clean. `min_ad_segment_separation=60s` merged Up
  First’s two membership CTAs through the headlines (cut 0–99s). Serial
  expand_neighbors stretched 0–28.9s inner span to 0–46s into the story.

## Verdict

**Not fold-eligible.** Inner-span opening recall is now 6/6, but (1) the
production self-promo guardrail is a policy choice, not a bug in t=0
chunking; (2) the same NPR membership is gold-`is_ad` on Up First and
gold-`not_ad` on Planet Money; (3) candidate-window ≥50% is still 2/6
because the 90s gold windows include show open — passing that bar means
cutting cold opens; (4) Up First’s 60s merge ate ~50s of news. Flag
defaults **false**; live Podly service was not changed.

## Claim

`general_podcast_ads` remains **LOCKED** (silver Gemini judge votes only; no human spot-check).
