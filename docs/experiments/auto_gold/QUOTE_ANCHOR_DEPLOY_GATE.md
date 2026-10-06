# Quote-anchor live deploy gate (2026-10-05/06)

`enable_quote_anchor_boundaries` stays **false**. Scout confirm stays **false**.
`general_podcast_ads` remains **LOCKED**. Live `podly.service` left at `20ed084`.

Holdout constants were **not** retuned for the deploy gate. The 8–12s DAI-bed
pull below is **now-seen** (motivated by this holdout debug) and does not
reopen the gate.

## Original 10 (integrated pipeline)

Cached words + `llm_cache_v2`. Cutter path: prompt_v2 → in-order match → lexical_v2 → ±2s snap.

| | inner catch | openings | not_ad | edge | input tok/ep |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 96.52% | 7/7 | 2.72s | 26.43s | 23525.9 |
| quote-anchor | 96.52% | 7/7 | 0.00s | 25.95s | 20473.2 |

Original-10 gate: **pass** (catch ≥96.5%, openings 7/7, not_ad ≤2.7s, edge ≤26.4s, tokens ≤ baseline). Unchanged after the 8–12s DAI-bed pull (Planet Money 1921 closer exemption holds; start stays 1921.38).

Replay wall (no Whisper): 1.88s/episode. Gemini **$0**.

## Holdout (untouched; 4 shows) — deploy gate

Shows: Marketplace (underwriting), Bill Simmons (DAI + host-read), The Indicator (NPR underwriting/membership), This American Life (membership + DAI). How I Built This download failed (`IncompleteRead`); not scored.

| | inner catch | openings | not_ad | edge |
| --- | ---: | ---: | ---: | ---: |
| baseline | 93.39% | 3/3 | 0.00s | 36.38s |
| quote-anchor | 87.58% | 3/3 | 0.00s | 38.35s |

Holdout gate (no worse than baseline catch / not_ad within 1 pp): **fail** (−5.82 pp catch; not_ad tied at 0s).

Per-show inner catch (**untouched**, pre DAI-bed):

| show | baseline | quote-anchor | Δ |
| --- | ---: | ---: | ---: |
| marketplace | 0.00% (no labeled ads) | 0.00% | 0.00 |
| bill_simmons | 99.36% | 98.95% | −0.41 |
| npr_indicator | 83.63% | 68.86% | −14.77 |
| this_american_life | 99.68% | 99.68% | 0.00 |

Differentiator is NPR Indicator 422.2–461.1 inner (not 370.8–398.8). Quotes never silently dropped.

## NPR Indicator debug (cached words + LLM; $0)

### 422.2–474.9 (judge inner 422.2–461.1) — the real loss

Segment Whisper (baseline / live LLM source) has the full stacked break:

- 404.36–419.36 Wise underwriting
- 419.36–433.36 donate.npr.org membership
- 439.36–459.36 Wait Wait sister-show (Peter Sagal)

Word-timestamp Whisper **drops** Wise+donate: 20.12s hole 401.24–421.36, then misplaced `Thank you so much. Bye-bye.` at 421.36–426.80, then 9.18s hole 426.80–435.98, then Peter at 435.98.

Cached Gemini outputs for this chunk:

| LLM source | `start_quote` | `end_quote` |
| --- | --- | --- |
| word-derived segs (eval path that scored the miss) | `Hey it's me Peter Seggel host` | `or wherever you get your podcasts` |
| segment transcript (also cached) | `This message comes from Wise, the smart` | `wherever you get your podcasts` |

Peter quote **matcher before lexical_v2**: word hit `Hey, it's me, Peter Seggel. host` **435.98–440.48** through `or wherever you get your podcasts.` **455.62–457.76**. Method `word`. Span **435.98–457.76**.

Per-rule after that span:

| rule | applied? | span |
| --- | --- | --- |
| pause / lead-in forward-trim | **no** (`is_lead=False`, no `LEADIN_RE` after the 2.88s pause at 439.30) | 435.98–457.76 |
| 12s hole pull | **no** (pre-quote hole 9.18s < 12s) | 435.98–457.76 |
| CTA tail | **no** | 435.98–457.76 |
| ±2s silence snap | start 435.98 → **436.23** | cut 436.23–459.282 |

Root cause: **LLM quoted mid-ad** because the word-seg excerpt omitted Wise+donate. Not a matcher miss. **Not** the Wait Wait forward-trim.

Baseline cut 404.336–459.282 covers 37.08s of the 38.9s inner. Quote-anchor 436.23 start covers ~23s (−14s lead-in).

If the Wise quote is matched against **segment** spans in-chunk, fail-closed `segment_fallback` is **404.36–459.36** (covers the inner). Word-stream in-chunk cannot find `This message comes from Wise` (that copy only exists at post-roll ~3340s).

### 370.8–398.8 — both miss; label is wrong

Judge: `is_ad=true`, `educational/self_promo`, inner **384.8–398.8**. Baseline 0s, quote-anchor 0s.

Transcript (word stream): *Yes, hello, my name is Janis Beck. I live in Yvokuni, Japan. True line is one of the best. … I wait each week for a new episode, and I'm never disappointed. Keep on doing it, please.*

This is **listener voicemail praising Throughline**, not membership, underwriting, or a sister-show promo. Neither classifier flagged it because it is show content. Policy counts self-promo / network promos / membership asks as ads; listener mail does not qualify. The gold inner is over-inclusive.

## Proposed fixes

1. **422 (tested $0 lexical):** pull start across an 8–12s wordless hole unless the previous sentence matches `CLOSER_RE` (Planet Money 1921 `thanks for listening` + 9.9s). Recovers 435.98 → 426.80 (the DAI bed). Remaining inner miss is 422.2–426.8 (~4.6s of donate still absent from the word stream).
2. **422 (structural, wired, not $0-replayed on original 10 prompts):** keep LocalWhisper at `word_timestamps=False` so the LLM sees DAI copy; load words only in `QuoteAnchorClassifier.load_words`; fail-closed segment-edge when the word stream omits the midroll. Original-10 `llm_cache_v2` keys are word-seg prompts, so this path was not replayed there.
3. **370:** no cutter change. Do not treat this window as a quote-anchor miss.

## Holdout **NOW-SEEN** (DAI-bed 8–12s; not a deploy-gate)

| | inner catch | openings | not_ad | edge |
| --- | ---: | ---: | ---: | ---: |
| baseline | 93.39% | 3/3 | 0.00s | 36.38s |
| quote-anchor | 91.35% | 3/3 | 0.00s | 28.91s |

Still fails the 1 pp catch gate (−2.04 pp). Gemini **$0**.

Per-show inner (now-seen):

| show | baseline | quote-anchor | Δ |
| --- | ---: | ---: | ---: |
| marketplace | 0.00% | 0.00% | 0.00 |
| bill_simmons | 99.36% | 98.95% | −0.41 |
| npr_indicator | 83.63% | 78.63% | −5.00 |
| this_american_life | 99.68% | 99.68% | 0.00 |

NPR 422 inner coverage after DAI-bed: baseline 37.08s, quote-anchor 32.48s (cut 426.8–459.282). NPR 370 inner: 0s both.

## Spend / flags

This debug: Gemini **$0** (cached words + `llm_cache_v2`). Flag remains false. Live remains `20ed084`.

## Verdict

Do **not** enable `ENABLE_QUOTE_ANCHOR_BOUNDARIES` on live. Rollback identity unchanged: `20ed084`.
