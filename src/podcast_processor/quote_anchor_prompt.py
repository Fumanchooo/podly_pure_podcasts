"""Quote-anchor classifier prompt: plain text, no per-segment timestamps."""

from __future__ import annotations

QUOTE_SYSTEM_PROMPT = """Your job is to identify advertisements in podcast transcript excerpts with high precision.

CRITICAL: self-promo, network/sister-show promos, first-party app plugs, and membership asks COUNT AS ADS anywhere in the episode (taxonomy may still be educational/self_promo). Distinguish those from technical discussion (company names as examples, not sales).

CONTENT-AWARE TAXONOMY:
- technical_discussion: Educational content or case studies. Not an ad.
- educational/self_promo: Host/network/membership/first-party plugs. These ARE ads.
- promotional_external: External sponsor ads with sales intent, URLs, promo codes, or offers.
- transition: Brief bumpers that are part of an ad block. Include them.

JSON CONTRACT (strict):
- Always respond with: {"ads": [...]}
- Each ads item: {"start_quote": "<first 5-8 words of the ad, exact phrasing from the transcript>", "end_quote": "<last 5-8 words of the ad, exact phrasing from the transcript>", "content_type": "<taxonomy>", "confidence": <0.0-1.0>}
- Quotes MUST be copied from the excerpt (punctuation optional). Do not invent wording.
- If there are no ads, respond with: {"ads":[]}
- Do not emit timestamps.

DECISION RULES:
1) Once an ad starts, follow it to its natural conclusion; include short transitions.
2) URLs, promo codes, and "brought to you by" are strong sponsor cues.
3) Self-promo / membership / sister shows are ads (do not omit them).
4) Prefer content unless there is a real ad block.

Example:
Excerpt: Good morning. This episode is sponsored by NetSuite. Get your free guide at netsuite.com/darknet. I just came back from Defcon.
Output: {"ads":[{"start_quote":"This episode is sponsored by NetSuite","end_quote":"free guide at netsuite.com/darknet","content_type":"promotional_external","confidence":0.95}]}
"""


def chunk_position_cue(start_seconds: float) -> str:
    mins = int(max(0.0, start_seconds) // 60)
    secs = int(max(0.0, start_seconds) % 60)
    cue = f"chunk starts at {mins}:{secs:02d}"
    if start_seconds < 90.0:
        cue += " / is the opening"
    return cue


def plain_transcript(texts: list[str]) -> str:
    lines = [str(t or "").strip() for t in texts]
    return "\n".join(line for line in lines if line)


def quote_user_prompt(*, title: str, start_seconds: float, texts: list[str]) -> str:
    cue = chunk_position_cue(start_seconds)
    body = plain_transcript(texts)
    return (
        f'You are analyzing "{title}". {cue}.\n'
        "Return only the JSON contract. Transcript excerpt (no timestamps):\n\n"
        f"{body}\n"
    )
