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


QUOTE_SYSTEM_PROMPT_V2 = """Your job is to identify complete advertisement BLOCKS in podcast transcript excerpts.

CRITICAL: self-promo, network/sister-show promos, first-party app plugs, and membership asks COUNT AS ADS anywhere in the episode (taxonomy may still be educational/self_promo). Distinguish those from technical discussion (company names as examples, not sales).

CONTENT-AWARE TAXONOMY:
- technical_discussion: Educational content or case studies. Not an ad.
- educational/self_promo: Host/network/membership/first-party plugs. These ARE ads.
- promotional_external: External sponsor ads with sales intent, URLs, promo codes, or offers.
- transition: Brief bumpers that are part of an ad block. Include them.

JSON CONTRACT (strict):
- Always respond with: {"ads": [...]}
- Each ads item: {"start_quote": "<first 5-8 words>", "end_quote": "<last 5-8 words>", "content_type": "<taxonomy>", "confidence": <0.0-1.0>}
- Quotes MUST be copied from the excerpt (punctuation optional). Do not invent wording.
- If there are no ads, respond with: {"ads":[]}
- Do not emit timestamps.

QUOTE THE FULL AD BREAK, NOT JUST THE SPONSOR SENTENCE:
- start_quote = the first words of the transition INTO the ad. Prefer host handoff / preroll lines such as "support for this show comes from", "this episode is brought to you by", "this message comes from", "we'll be right back", "after the break", "today's episode is sponsored by", or a similar bumper. If the excerpt starts already inside the ad, use the first 5-8 words present.
- end_quote = the last words BEFORE the show resumes, including outro or CTA tail: URL, promo code, "back to the show", "and we're back", "thanks to our sponsor". Do not stop at the first CTA if more ad copy follows.

DECISION RULES:
1) Once an ad break starts, follow it to its natural conclusion; include short transitions on both sides.
2) URLs, promo codes, and "brought to you by" are strong sponsor cues.
3) Self-promo / membership / sister shows are ads (do not omit them).
4) Prefer content unless there is a real ad block.
5) Adjacent reads for the same break are one ad; a later separate break is another ad.

Example:
Excerpt: Good morning. Support for this show comes from NetSuite. Get your free guide at netsuite.com/darknet. And we're back. I just came back from Defcon.
Output: {"ads":[{"start_quote":"Support for this show comes from NetSuite","end_quote":"And we're back","content_type":"promotional_external","confidence":0.95}]}
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


def quote_user_prompt(
    *,
    title: str,
    start_seconds: float,
    texts: list[str],
    prompt_version: str = "v1",
) -> str:
    cue = chunk_position_cue(start_seconds)
    body = plain_transcript(texts)
    extra = ""
    if prompt_version == "v2":
        extra = (
            " Quote the handoff into each ad break and the last words before "
            "the show resumes (outro/CTA tail).\n"
        )
    return (
        f'You are analyzing "{title}". {cue}.{extra}\n'
        "Return only the JSON contract. Transcript excerpt (no timestamps):\n\n"
        f"{body}\n"
    )


def system_prompt_for(version: str) -> str:
    if version == "v2":
        return QUOTE_SYSTEM_PROMPT_V2
    return QUOTE_SYSTEM_PROMPT
