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
- Filler words ("like", "you know", "um", "I mean") that the INTENDED text left out,
  including discourse words at the start of a sentence ("Okay", "OK", "So", "Oh",
  "Well", "Oh, and") when the INTENDED text does not start that sentence with them.
  Only remove fillers: never add a filler or discourse word, even if the INTENDED
  text has one that the DRAFT left out
- Wrong names, including misspelled names, and numbers or dates with the wrong value
  (a different number or date, not the same one written differently)
- Anything that must be written exactly: code identifiers, file names, commands, field
  and config values, versions in code, emails, URLs, and product and proper names.
  Make them match the INTENDED text exactly, including any digits in them
  ("calculate prorated amount" -> "calculateProratedAmount",
  "start date" -> "start_date", "billing ts" -> "billing.ts",
  "PG Bouncer" -> "PgBouncer", "node twenty alpine" -> "node:20-alpine")
- A wrong unit, or an informal/slang unit ("512 megs" -> "512 MB")

Do NOT fix these (same meaning, or only formatting):
- Different wording with the same meaning: articles, singular/plural, tense or
  contraction changes that keep the meaning ("a" vs "the", "note" vs "notes",
  "that handles" vs "that'd handle"), synonyms, paraphrases and word-order changes
  ("I'll mention" vs "I want to mention")
- Small grammar slips and spelling variants that do not change what the reader
  understands: plural vs possessive ("league" vs "league's", "deadlines" vs
  "deadline's"), "they" vs "they'd", "we" vs "we'd", "logs" vs "log", "holiday" vs
  "holidays", and a missing or extra courtesy word ("please")
- Punctuation, capitalization, line breaks, bullets, numbered lists, markdown
- How ordinary numbers are written. Some dictation tools cannot write digits, so
  numbers in words are acceptable, as are digits, decimals vs fractions, and spoken
  separators ("4" vs "four", "$1,450" vs "fourteen fifty", "1.5" vs "1 1/2",
  "$618.40" vs "618 dollars and 40 cents", "10:30 AM" vs "ten thirty AM"). This covers
  amounts, quantities, dates, times, phone numbers, and order, account and ticket
  numbers ("415-555-0147" vs "four one five, five five five, zero one four seven",
  "#ER-3312" vs "ER dash three three one two"), as long as every digit and letter is
  right. A wrong digit is still an error. It does not cover the exact-text items
  above (code, file names, commands, versions in code, emails, URLs, product and
  proper names), where digits must match.
- Standard abbreviations ("tbsp" vs "tablespoons", "lb" vs "pound",
  "mph" vs "miles an hour", "Dr." vs "Doctor")
- Hyphenation and compounding of ordinary words ("short-staffed" vs "short staffed")
- Contractions vs full forms ("we're" vs "we are")
- US vs UK spelling

Rules:
- Leave every part of the DRAFT that does not need fixing exactly as it is.
- When you insert or replace words, match the formatting style of the surrounding
  DRAFT text, including writing numbers as words if the DRAFT does.
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
"no, make it", "scratch that", "take that off"), leave it unchanged.
It is not an error.

"I mean" is a self-correction cue only when it introduces a correction ("Tuesday,
I mean Wednesday"). Anywhere else ("and I mean, fix every import") it is a filler
and follows the filler rule above.

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
