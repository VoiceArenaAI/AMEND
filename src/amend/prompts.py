BASE = """You are evaluating the output of a dictation tool. You will be given the INTENDED text
(what the user wanted to end up with) and the DRAFT (what the dictation tool produced).

Your job: find what the user would actually have to fix in the DRAFT so that it serves
the same purpose as the INTENDED text, and fix only that, using the fewest possible
word edits. The DRAFT does not need the same wording as the INTENDED text. If a reader
would understand the same thing and act the same way, leave it as it is.

Fix these (they change the meaning, or the user would have to retype them):
- Misrecognized or wrong words ("weight" -> "wait", "offensive" -> "defensive",
  "chronic" -> "cron")
- Missing or extra content of any length. Add anything the INTENDED text says that the
  DRAFT is missing, and delete anything the tool added that is not in the INTENDED text
  (for example an answer it generated, invented content, or repeated text)
- Filler words ("like", "you know", "um") that the INTENDED text left out
- Wrong names, numbers or dates, including misspelled names
- Anything that must be written exactly: code identifiers, file names, commands, field
  and config values, product names, emails and URLs. Make them match the INTENDED text
  exactly ("calculate prorated amount" -> "calculateProratedAmount",
  "start date" -> "start_date", "billing ts" -> "billing.ts",
  "node 20 alpine" -> "node:20-alpine", "PG Bouncer" -> "PgBouncer")
- A wrong unit, or an informal/slang unit ("512 megs" -> "512 MB")

Do NOT fix these (same meaning, or only formatting):
- Different wording with the same meaning: articles, singular/plural, tense or
  contraction changes that keep the meaning ("a" vs "the", "note" vs "notes",
  "that handles" vs "that'd handle"), synonyms, paraphrases and word-order changes
  ("I'll mention" vs "I want to mention")
- Punctuation, capitalization, line breaks, bullets, numbered lists, markdown
- Numbers written as digits vs words, or as decimals vs fractions ("4" vs "four",
  "1" vs "one", "$1,450" vs "fourteen fifty", "1.5" vs "1 1/2")
- Standard abbreviations ("tbsp" vs "tablespoons", "lb" vs "pound",
  "mph" vs "miles an hour", "Dr." vs "Doctor")
- Hyphenation and compounding of ordinary words ("short-staffed" vs "short staffed")
- Contractions vs full forms ("we're" vs "we are")
- US vs UK spelling

Rules:
- Leave every part of the DRAFT that does not need fixing exactly as it is.
- When you insert or replace words, match the formatting style of the surrounding
  DRAFT text.
- Make each fix as small as possible: replace one word rather than rewriting the
  sentence around it. But fix a missing or extra passage in full.
- If the DRAFT is empty, "fixed" is the INTENDED text.
{lenient}
{spoken}<intended>
{intended}
</intended>

<draft>
{draft}
</draft>

Respond with JSON whose "fixed" field holds only the corrected draft text, with no
explanation, notes or reasoning in it. If no changes are needed, "fixed" is the DRAFT
exactly as given."""

LENIENT = """
You are also given SPOKEN: a verbatim transcript of what the user actually said,
including any self-corrections. A self-correction can replace something ("Tuesday,
no wait, Wednesday", "ten AM, sorry, ten thirty", "five times, no, make it three")
or retract something ("get milk and eggs, oh wait, we have eggs, take that off").

Self-corrections are lenient. If a difference between the DRAFT and the INTENDED
text exists only because the DRAFT kept a self-correction from SPOKEN (for example
it kept both the original and the corrected version, kept a retracted item together
with the retraction, or kept cue words such as "no wait", "sorry", "actually no",
"no, make it", "I mean", "scratch that", "take that off"), leave it unchanged.
It is not an error.

This takes priority over the rule about extra content. Keep a kept self-correction
whole: the original, the cue words and the correction. This applies even if the cue
words were misheard ("wait now" for "wait no", "no way" for "no wait"). Never delete
only part of it.

But if the DRAFT ends up with the wrong final value, fix it. For example, it kept
"Tuesday" and dropped the correction to "Wednesday", or it mixed the two up.
Everything else follows the normal rules above.
"""

VERSIONS = ("v1",)


def build_prompt(intended: str, pred: str, spoken: str, version: str = "v1") -> str:
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}, expected one of {VERSIONS}")
    if spoken is None:
        raise ValueError("the spoken text is required")
    return BASE.format(
        lenient=LENIENT,
        spoken=f"<spoken>\n{spoken}\n</spoken>\n\n",
        intended=intended or "",
        draft=pred or "",
    )
