# Quote-anchor live deploy gate (2026-10-07)

`enable_quote_anchor_boundaries` stays **false**. Scout confirm stays **false**.
`general_podcast_ads` remains **LOCKED**. Live `podly.service` left at `20ed084`.

Structural path (this table): LLM sees **segment** Whisper text (no timestamps).
Words are matcher-only, with fail-closed **segment-edge** then chunk fallback when
the quote is missing from the word stream. 8–12s DAI-bed pull is **off** (no
original-10 lift; holdout structural identical with or without it).

Holdout is **now-seen** (debug + label fix + structural LLM source) and does
not reopen the original-10 gate.

## Scorecard (one table)

| set | system | inner catch | openings | not_ad | edge | input tok/ep |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original 10 | baseline | 96.52% | 7/7 | 2.72s | 26.43s | 23526 |
| original 10 | quote-anchor (structural) | 95.09% | 7/7 | 0.00s | 32.34s | 22578 |
| holdout now-seen, corrected label | baseline | 98.93% | 3/3 | 0.00s | 22.38s | 29174 |
| holdout now-seen, corrected label | quote-anchor (structural) | 98.71% | 3/3 | 0.00s | 10.35s | 30049 |

Original-10 gate (catch ≥96.5%, openings 7/7, not_ad ≤2.7s, edge ≤26.4s, tokens ≤ baseline): **fail** (inner −1.43 pp, edge +5.91s). Tokens OK.

Holdout gate (no worse than baseline catch / not_ad within 1 pp): **pass** (−0.22 pp catch; not_ad tied at 0s / 617s pool). Flagged now-seen.

Holdout per-show inner (corrected label; structural; DAI-bed off):

| show | baseline | quote-anchor | Δ |
| --- | ---: | ---: | ---: |
| marketplace | 0.00% (no labeled ads) | 0.00% | 0.00 |
| bill_simmons | 99.36% | 98.95% | −0.41 |
| npr_indicator | 97.80% | 97.52% | −0.28 |
| this_american_life | 99.68% | 99.68% | 0.00 |

Quotes never silently dropped. Original 10: 41 quotes (35 word / 5 segment_fallback / 1 chunk_fallback). Holdout: 21 (15 / 6 / 0). NPR Indicator Wise+donate is `segment_fallback` **404.36–459.36**, cut **400.08–459.28** (covers inner 422.2–461.1).

## Word-drop audit ($0; 14 shows)

Holes ≥5s where segment Whisper has speech and `word_timestamps=True` has none:

| | holes | segment-speech in holes | holes overlapping labeled ads | labeled-ad audio in holes |
| --- | ---: | ---: | ---: | ---: |
| all 14 | 98 | 646.0s | 6 | 34.43s |
| original 10 | 46 | 313.3s | 3 | 27.19s |
| holdout | 52 | 332.7s | 3 | 7.24s |

Word-mode dropping speech is **systemic** (98 holes / 646s). Labeled-ad overlap is sparse. Largest labeled hit: Up First **865.4–887.1** (19.1s). Indicator Wise+donate **401.24–421.36** (19.1s segment speech) is a word hole but **unlabeled** (gold inner starts 422.2); the 9.18s hole **426.80–435.98** overlaps 6.56s of labeled inner.

## Label fix ($0)

Indicator **384.8–398.8** (window 370.8–398.8) listener voicemail (Janis Beck praising Throughline) is **not_ad** under `general_podcast_ads`. Reason recorded on the window in holdout `labels.json` (`label_reason` + `original_judge`). After the fix, not_ad pool is 617s and not_ad cut is 0s on both systems. Word-path after the fix was 98.93 vs 96.76 (−2.16); structural recovers to −0.22.

## Spend / flags

| item | Gemini $ |
| --- | ---: |
| word-drop audit | $0 |
| label fix + rescore | $0 |
| original 10 structural (UTC 2026-10-07) | $0.178 |
| original 10 DAI-bed A/B | $0 (cache; identical) |
| holdout structural nodai | $0 (cache) |
| UTC-day remaining after original 10 | ~$0.322 |

DAI-bed 8–12s (closer exemption) **does not help** the original 10, so production default is **false**. Flag remains false. Live remains `20ed084`.

## Verdict

Do **not** enable `ENABLE_QUOTE_ANCHOR_BOUNDARIES` on live. Rollback identity unchanged: `20ed084`.
`general_podcast_ads` remains **LOCKED**.
