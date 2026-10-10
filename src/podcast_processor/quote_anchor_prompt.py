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


QUOTE_SYSTEM_PROMPT_V3 = """Your job is to identify complete advertisement BLOCKS in podcast transcript excerpts.

CRITICAL: self-promo, network/sister-show promos, first-party app plugs, and membership asks COUNT AS ADS anywhere in the episode (taxonomy may still be educational/self_promo). Distinguish those from technical discussion (company names as examples, not sales).

CONTENT-AWARE TAXONOMY:
- technical_discussion: Educational content or case studies. Not an ad.
- educational/self_promo: Host/network/membership/first-party plugs. These ARE ads.
- promotional_external: External sponsor ads with sales intent, URLs, promo codes, or offers.
- transition: Brief bumpers that are part of an ad block. Include them.

JSON CONTRACT (strict):
- Always respond with: {"ads": [...]}
- Each ads item: {"start_quote": "<verbatim first full sentence of the ad break>", "end_quote": "<verbatim last full sentence of the WHOLE stacked ad break>", "content_type": "<taxonomy>", "confidence": <0.0-1.0>}
- start_quote MUST be the entire first sentence of the ad block, copied from the excerpt, including lead-ins such as "support for this podcast comes from…", "this episode is brought to you by…", "this week on…".
- end_quote MUST be the entire last sentence BEFORE the show resumes. For stacked sister-show / network promos, that is the last sentence of the LAST promo in the stack, not the first CTA.
- If the ad is a single sentence, start_quote and end_quote are that same sentence.
- Quotes MUST be copied from the excerpt (punctuation optional). Do not invent or paraphrase.
- If there are no ads, respond with: {"ads":[]}
- Do not emit timestamps.

STACKED BREAKS (do not truncate):
- Adjacent ads separated by only a few seconds with no show content are ONE break. Union them.
- NPR / network stacks (Ted Radio Hour then Fresh Air, Up First then Politics, etc.) are one break.
- Do NOT stop end_quote at the first "listen on the NPR app" / "wherever you get your podcasts" CTA if another sister-show promo still follows.
- end_quote = last sentence of the WHOLE stacked break.

QUOTE THE FULL AD BREAK:
- The first sentence is the handoff INTO the ad when that lead-in is present. If the excerpt starts already inside the ad, use the first full sentence present.
- The last sentence is the last ad copy BEFORE the show resumes.

DECISION RULES:
1) Once an ad break starts, follow it to its natural conclusion; include short transitions on both sides.
2) URLs, promo codes, and "brought to you by" are strong sponsor cues.
3) Self-promo / membership / sister shows are ads (do not omit them).
4) Prefer content unless there is a real ad block.
5) Adjacent reads for the same break are one ad; a later separate break is another ad.

Example:
Excerpt: Good morning. Support for this podcast comes from NetSuite. Get your free guide at netsuite.com/darknet. Terms apply. I just came back from Defcon.
Output: {"ads":[{"start_quote":"Support for this podcast comes from NetSuite.","end_quote":"Terms apply.","content_type":"promotional_external","confidence":0.95}]}

Stacked example:
Excerpt: This week on Ted Radio Hour, listen on the NPR app. And this week on Fresh Air, Terry talks with a guest. Listen on the NPR app or wherever you get your podcasts. After the break we continue.
Output: {"ads":[{"start_quote":"This week on Ted Radio Hour, listen on the NPR app.","end_quote":"Listen on the NPR app or wherever you get your podcasts.","content_type":"educational/self_promo","confidence":0.95}]}
"""


QUOTE_SYSTEM_PROMPT_V4 = """Your job is to identify complete advertisement BLOCKS in podcast transcript excerpts.

CRITICAL: self-promo, network/sister-show promos, first-party app plugs, and membership asks COUNT AS ADS anywhere in the episode (taxonomy may still be educational/self_promo). Distinguish those from technical discussion (company names as examples, not sales).

CONTENT-AWARE TAXONOMY:
- technical_discussion: Educational content or case studies. Not an ad.
- educational/self_promo: Host/network/membership/first-party plugs. These ARE ads.
- promotional_external: External sponsor ads with sales intent, URLs, promo codes, or offers.
- transition: Brief bumpers that are part of an ad block. Include them.

JSON CONTRACT (strict):
- Always respond with: {"ads": [...]}
- Each ads item: {"start_anchor": "<shortest unique verbatim string at the very start of the ad break>", "end_anchor": "<shortest unique verbatim string at the very end of the WHOLE break>", "content_type": "<taxonomy>", "confidence": <0.0-1.0>}
- start_anchor = the shortest verbatim string at the very start of the ad break that occurs ONLY ONCE in this excerpt. Copy phrasing from the excerpt (punctuation optional).
- end_anchor = the shortest verbatim string at the very end of the WHOLE stacked break (after ALL stacked promos/CTAs, last words before the show resumes) that occurs ONLY ONCE in this excerpt.
- If the ad is a single unique phrase, start_anchor and end_anchor may be that same string.
- Quotes MUST be copied from the excerpt. Do not invent or paraphrase.
- If there are no ads, respond with: {"ads":[]}
- Do not emit timestamps.

UNIQUENESS (mandatory):
- Repeated CTAs are NOT unique and MUST NOT be used as anchors: "listen on the NPR app", "on the NPR app", URLs that appear more than once, "wherever you get your podcasts", "wherever you listen".
- If a short string repeats, EXTEND it with neighboring verbatim words until the whole string occurs only once.
- Prefer a unique closer of the LAST promo (e.g. a show name that appears only in the final CTA) over a network-wide CTA.

STACKED BREAKS (do not truncate):
- Adjacent ads separated by only a few seconds with no show content are ONE break.
- NPR / network stacks (Show A then Show B then Show C) are one break. end_anchor is the last unique words of the LAST promo, not the first CTA.

DECISION RULES:
1) Once an ad break starts, follow it to its natural conclusion; include short transitions on both sides.
2) URLs, promo codes, and "brought to you by" are strong sponsor cues.
3) Self-promo / membership / sister shows are ads (do not omit them).
4) Prefer content unless there is a real ad block.
5) Adjacent reads for the same break are one ad; a later separate break is another ad.

Example:
Excerpt: Good morning. Support for this podcast comes from Acme. Get your free guide at acme.example/show. Terms apply at acme.example/legal. I just came back from Defcon.
Output: {"ads":[{"start_anchor":"Support for this podcast comes from Acme","end_anchor":"Terms apply at acme.example/legal","content_type":"promotional_external","confidence":0.95}]}

Stacked example (repeated CTA is illegal as end_anchor):
Excerpt: This week on Show A. Listen on the NPR app or wherever you get your podcasts. And this week on Show B. Listen on the NPR app or wherever you get Show B. After the break we continue.
Output: {"ads":[{"start_anchor":"This week on Show A","end_anchor":"wherever you get Show B","content_type":"educational/self_promo","confidence":0.95}]}
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
    elif prompt_version == "v3":
        extra = (
            " Quote the FULL first sentence of the break and the FULL last "
            "sentence of the WHOLE stacked break (do not stop at the first "
            "'listen on the NPR app' CTA if another promo follows).\n"
        )
    elif prompt_version == "v4":
        extra = (
            " Return start_anchor and end_anchor: the shortest UNIQUE verbatim "
            "strings at the very start and very end of the WHOLE stacked break. "
            "Repeated CTAs ('listen on the NPR app', URLs, 'wherever you get "
            "your podcasts') are not unique — extend until unique.\n"
        )
    return (
        f'You are analyzing "{title}". {cue}.{extra}\n'
        "Return only the JSON contract. Transcript excerpt (no timestamps):\n\n"
        f"{body}\n"
    )


def confirm_user_prompt(
    *,
    title: str,
    start_seconds: float,
    texts: list[str],
    prompt_version: str = "v2",
) -> str:
    """One-candidate confirm: smaller window, ads:[] if the region is not an ad."""
    cue = chunk_position_cue(start_seconds)
    body = plain_transcript(texts)
    extra = ""
    if prompt_version == "v2":
        extra = (
            " Quote the handoff into each ad break and the last words before "
            "the show resumes (outro/CTA tail)."
        )
    elif prompt_version == "v3":
        extra = (
            " Quote the FULL first sentence and the FULL last sentence of the "
            "WHOLE stacked break. Adjacent sister-show/network promos with only "
            "a few seconds and no show content between them are one break. "
            "Do not stop end_quote at the first 'listen on the NPR app' CTA."
        )
    elif prompt_version == "v4":
        extra = (
            " Return start_anchor / end_anchor: shortest UNIQUE verbatim strings "
            "at the very start and very end of the WHOLE stacked break after all "
            "sister-show promos. Repeated CTAs ('listen on the NPR app', URLs, "
            "'wherever you get your podcasts') are not unique; extend until unique."
        )
    return (
        f'You are analyzing "{title}". {cue}. This excerpt is a CANDIDATE ad '
        "region plus padding — it may be show content. If there is no ad "
        f"(including no self-promo / sister-show / app / membership plug), "
        f'return {{"ads":[]}}.{extra}\n'
        "If there is an ad break, quote the FULL break (not just the cue). "
        "If the ad continues past the start or end of this excerpt, still "
        "quote the first/last sentence (or words) that ARE present.\n"
        "Return only the JSON contract. Transcript excerpt (no timestamps):\n\n"
        f"{body}\n"
    )


def system_prompt_for(version: str) -> str:
    if version == "v4":
        return QUOTE_SYSTEM_PROMPT_V4
    if version == "v3":
        return QUOTE_SYSTEM_PROMPT_V3
    if version == "v2":
        return QUOTE_SYSTEM_PROMPT_V2
    return QUOTE_SYSTEM_PROMPT
