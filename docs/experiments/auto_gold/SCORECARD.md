# Production remover vs auto-gold scorecard

Production Podly `AdClassifier` (scout confirm **OFF**) scored against this-audio re-judge windows on the frozen auto-gold timestamps. 10/12 episodes. Overlap scoring only; no second judge.

## Metrics

| metric | value |
| --- | ---: |
| is_ad audio caught (candidate-window seconds) | 28.2% (254.1s / 900.0s) |
| is_ad audio missed (candidate-window seconds) | 71.8% (645.9s) |
| is_ad audio caught (judge inner ad_span seconds) | 57.7% (242.8s / 421.0s) |
| is_ad windows caught (≥50% of candidate window) | 29.4% (5/17) |
| is_ad windows missed | 12/17 |
| opening/preroll caught (seconds overlapping t=0–90s) | 18.1% (97.7s / 540.0s) |
| opening/preroll missed (t=0–90s) | 81.9% (442.3s) |
| opening/preroll caught (inner ad_span ∩ t=0–90s) | 41.5% (97.7s / 235.4s) |
| opening windows caught (≥50% of candidate) | 0.0% (0/6) |
| opening windows caught (≥50% of inner ad_span) | 33.3% (2/6) |
| t=0 windows caught (≥50% of candidate) | 0.0% (0/6) |
| t=0 windows caught (≥50% of inner ad_span) | 33.3% (2/6) |
| not_ad seconds also cut (FP-ish) | 2.7s / 1224.0s |
| shows scored | 10/12 |
| Gemini spend | $0.4924 (cap $0.50) |
| `enable_bow_scout_gemini_confirm` | `False` |

## Claim

`general_podcast_ads` remains **LOCKED** (silver Gemini judge votes only; no human spot-check).

## Notes

- FN opening `npr_up_first` 0-92s inner ad_span 0.0/46.0s (0.0%); candidate window 0.0/92.0s
- FN opening `npr_wait_wait` 0-92s inner ad_span 0.0/25.7s (0.0%); candidate window 0.0/92.0s
- FN opening `serial` 0-92s inner ad_span 0.0/28.9s (0.0%); candidate window 0.0/92.0s
- Opening `wtf_maron` inner ad_span caught (98.2%) but 90s candidate window only 47.5%
- FN opening `crime_junkie` 0-92s inner ad_span 0.0/32.6s (0.0%); candidate window 0.0/92.0s
- Opening `darknet_diaries` inner ad_span caught (90.5%) but 90s candidate window only 44.3%

- Original per-chunk 29 is_ad votes were gitignored under `docs/experiments/auto_gold/runs/`. Headline uses this-audio re-judge on the frozen BASELINE_REPORT timestamps: 17 is_ad / 58 judged / 1 skipped (bbc_global_news 1751-1779s, no Whisper). DAI rotated openings vs that report: Planet Money EZKader→NPR membership (not is_ad here); Bill Simmons PNC→Ringer intro; Marketplace Odoo→show open; also Wait Wait Carvana→NPR Politics, Serial YouTube Premium→NYT app, Crime Junkie parenthood→Park Predators, BBC Whole Foods/Schwab→UK show open.
- Hit $0.50 Gemini cap after 10/12 shows ($0.4924 spent). Audio exists for bill_simmons, marketplace but no production cuts; this-audio re-judge is_ad=0 on those 10 windows, so is_ad recall would not change.
- Opening FNs: NPR membership/cross-promo, Serial NYT self-promo, Crime Junkie network promo (0s cut). WTF Withings and Darknet NetSuite prerolls were cut to the inner ad_span but fail the 90s-candidate 50% bar. Midroll host-reads on Planet Money / Crime Junkie / Darknet mostly caught; later Wait Wait underwriting mostly missed.
