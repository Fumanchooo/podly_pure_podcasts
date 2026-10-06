# Quote-anchor live deploy gate (2026-10-05/06)

`enable_quote_anchor_boundaries` stays **false**. Scout confirm stays **false**.
`general_podcast_ads` remains **LOCKED**. Live `podly.service` left at `20ed084`.

Holdout constants were **not** retuned.

## Original 10 (integrated pipeline)

Cached words + `llm_cache_v2`. Cutter path: prompt_v2 → in-order match → lexical_v2 → ±2s snap.

| | inner catch | openings | not_ad | edge | input tok/ep |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 96.52% | 7/7 | 2.72s | 26.43s | 23525.9 |
| quote-anchor | 96.52% | 7/7 | 0.00s | 25.95s | 20473.2 |

Original-10 gate: **pass** (catch ≥96.5%, openings 7/7, not_ad ≤2.7s, edge ≤26.4s, tokens ≤ baseline).

Replay wall (no Whisper): 1.98s/episode.

## Holdout (untouched; 4 shows)

Shows: Marketplace (underwriting), Bill Simmons (DAI + host-read), The Indicator (NPR underwriting/membership), This American Life (membership + DAI). How I Built This download failed (`IncompleteRead`); not scored.

| | inner catch | openings | not_ad | edge |
| --- | ---: | ---: | ---: | ---: |
| baseline | 93.39% | 3/3 | 0.00s | 36.38s |
| quote-anchor | 87.58% | 3/3 | 0.00s | 38.35s |

Holdout gate (no worse than baseline catch / not_ad within 1 pp): **fail** (−5.82 pp catch; not_ad tied at 0s).

Per-show inner catch (untouched):

| show | baseline | quote-anchor | Δ |
| --- | ---: | ---: | ---: |
| marketplace | 0.00% (no labeled ads) | 0.00% | 0.00 |
| bill_simmons | 99.36% | 98.95% | −0.41 |
| npr_indicator | 83.63% | 68.86% | −14.77 |
| this_american_life | 99.68% | 99.68% | 0.00 |

The miss is NPR Indicator: a 14s membership/underwriting span at 370.8–398.8s was uncut; 422–475s was undercut. Quotes never silently dropped (20 quotes, 15 word / 5 segment-edge fallback).

## Cake CPU wall (holdout, local `base` Whisper)

| show | duration | baseline wall | quote-anchor wall | word-Whisper |
| --- | ---: | ---: | ---: | ---: |
| marketplace | 26 min | 221s | 618s | 601s |
| bill_simmons | 123 min | 3093s | 4010s | 3841s |
| npr_indicator | 57 min | 675s | 46s (words cached) | (prior pass) |
| this_american_life | 62 min | 476s | 2261s | 2210s |

Live path is one word-timestamp Whisper pass (not two). Word timestamps are slower than segment Whisper on Cake CPU (~3× on a 26 min episode). Background processing can absorb that; it is not why we blocked deploy.

## Spend / flags

UTC 2026-10-06 Gemini: judge $0.016 + pipeline $0.361 = **$0.377** (cap $0.50). Scout off.

## Verdict

Do **not** enable `ENABLE_QUOTE_ANCHOR_BOUNDARIES` on live. Rollback identity unchanged: `20ed084`.
