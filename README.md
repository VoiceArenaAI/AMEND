# AMEND

**AMEND** (Aggregate Minimum Edits Needed per Draft) measures how much a user would have to fix in a dictation tool's output before it is usable.

Word error rate (WER) compares a transcript with a single reference, so it penalises every legitimate difference: digits vs words, a resolved self-correction, a reworded phrase. AMEND instead asks a different question: *given what the user meant to write, what is the smallest set of edits that makes this draft acceptable?*

## Inspiration: HTER

AMEND is inspired by **HTER** (Human-targeted Translation Edit Rate) from machine translation evaluation ([Snover et al., 2006](https://aclanthology.org/2006.amta-papers.25/)).

In HTER, a human annotator takes a system's translation and edits it, as little as possible, until it is fluent and means the same as the reference. That edited version becomes a *targeted reference* for that output. The score is the edit rate between the system output and its own targeted reference. Because the target is built from the output itself, a correct translation that is worded differently from the reference costs nothing. Only edits a human actually needed are counted.

AMEND applies the same idea to dictation, with an LLM judge in the role of the human annotator:

| | HTER | AMEND |
|---|---|---|
| Output being scored | MT translation | Dictation tool's draft |
| Ground truth | Reference translation | Intended text, plus the spoken text |
| Who builds the targeted reference | Human annotator | LLM judge (`claude-opus-5-5`) |
| Edit rate | TER (word level, with shifts) | CER (character level, via `jiwer`) |

## How it works

For each draft:

1. The judge receives the **spoken text** (a verbatim transcript of what the user said), the **intended text** and the **draft**.
2. It returns the draft with the **minimum edits** needed for it to serve the same purpose as the intended text.
3. Both the draft and the judge's version are normalised: Unicode NFKC, lowercased, sentence punctuation at word edges stripped, whitespace collapsed.
4. The character edits (substitutions + deletions + insertions) needed to turn the draft into the judge's version are counted with `jiwer`, along with the number of characters in the judge's version. Spaces count as characters.

Two numbers summarise a set of drafts:

- **AMEND**: corpus-level character edit rate, pooled over all drafts. Lower is better; 0 means nothing needed fixing.

  ```
  AMEND = total character edits over all drafts / total characters in all judge versions
  ```

  Pooling means long drafts weigh more than short ones, as with corpus-level WER and TER.
- **AMEND@0**: the share of drafts that needed zero edits, i.e. drafts the user could send as-is.

Character edits are used rather than word edits because they track the user's typing effort: fixing "test" to "tests" costs 1 character, not a whole word.

### What the judge fixes

- Misrecognised or wrong words ("weight" → "wait", "chronic" → "cron")
- Missing or extra content of any length, including generated or hallucinated text
- Filler words ("like", "you know", "um") that the intended text left out, including sentence-initial "Okay", "So", "Oh", "Well"
- Wrong names, including misspelled names, and numbers or dates with the wrong value
- Anything that must be typed exactly, including any digits in it: code identifiers, file names, commands, field and config values, versions in code, emails, URLs, product and proper names ("calculate prorated amount" → `calculateProratedAmount`, "node twenty alpine" → `node:20-alpine`)
- Wrong or slang units ("512 megs" → "512 MB")

### What the judge leaves alone

- Different wording with the same meaning: articles, singular/plural, tense, contractions, synonyms, word order
- Punctuation, capitalisation, line breaks, bullets, numbered lists, markdown
- How ordinary numbers are written, since some dictation tools cannot write digits: words vs digits, decimals vs fractions, spoken separators, for amounts, quantities, dates, times, phone numbers and order/account/ticket numbers ("618 dollars and 40 cents" vs `$618.40`, "ER dash three three one two" vs `#ER-3312`). A wrong digit is still an error. Exact-text items above still need their digits.
- Standard abbreviations ("tbsp", "lb", "mph", "Dr.")
- Hyphenation and compounding of ordinary words, US vs UK spelling

### Self-corrections

Self-corrections are not penalised. The judge also receives the spoken text, so if the draft keeps "Tuesday, no wait, Wednesday" and the intended text says "Wednesday", that is not an error. A draft that ends up with the wrong final value ("Tuesday") is still fixed.

The full prompt is in [`src/amend/prompts.py`](src/amend/prompts.py).

### Normalisation

Only sentence-level punctuation at the start or end of a word is removed (`. , ; : ! ?`, quotes, brackets, `*`, `` ` ``, `_`, `~`, bullets, dashes, ellipses). Punctuation that carries meaning is kept: `%`, `/`, `$`, `@`, `#`, `&`, `+`, `=`, leading dots and dashes (`.env`, `--verbose`), and anything inside a token (`24.9`, `1,450`, `2:15`, `node:20-alpine`, `start_date`, `https://acme.com/docs`).

| Input | Normalised |
|---|---|
| `Send it to ops@northwind.io, or visit https://acme.com/docs.` | `send it to ops@northwind.io or visit https://acme.com/docs` |
| `It's 24.9% APR, about $1,450/month (roughly).` | `it's 24.9% apr about $1,450/month roughly` |
| `Wait... what? — no, it's fine.` | `wait what no it's fine` |

## Installation

Requires Python 3.11+.

```bash
pip install amend-va
# or
uv pip install amend-va
```

The package installs as `amend-va` and is imported as `amend`.

For development from a clone:

```bash
git clone https://github.com/VoiceArenaAI/AMEND.git
cd AMEND
uv sync
```

### Authentication

The judge calls the Anthropic API. Set your key before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Python

One draft:

```python
from amend import score

result = score(
    pred="Want to grab brunch Sunday? No way, Saturday. I'm free all mourning.",
    intended="Want to grab brunch Saturday? I'm free all morning.",
    spoken="Want to grab brunch Sunday, no wait, Saturday? I'm free all morning.",
)
# {'pred': '...', 'fixed': "... I'm free all morning.", 'edits': <int>, 'chars': <int>, 'amend': <float>}
```

Three arrays:

```python
from amend import score_arrays

results = score_arrays(preds, intendeds, spokens, mode="batch")
# one dict per draft: {'pred', 'fixed', 'edits', 'chars', 'amend'}
```

A pandas DataFrame, with one or more prediction columns:

```python
import pandas as pd
from amend import score_dataframe

df = pd.read_csv("all_preds.csv")
out = score_dataframe(
    df,
    pred_cols=["apple", "wispr"],
    intended_col="intended_text",
    spoken_col="spoken_text",
    mode="sync",
)
# adds <col>_fixed, <col>_edits, <col>_chars, <col>_amend for each column
```

The metric alone, without the judge:

```python
from amend import amend, char_edits, pooled

char_edits("the weight was long", "The wait was long.")  # (3, 17)
amend("the weight was long", "The wait was long.")       # 0.1765
pooled(out["apple_edits"], out["apple_chars"])            # corpus-level AMEND
```

### Command line

```bash
amend data.csv -p apple claude wispr -o scored.csv
```

| Option | Default | Meaning |
|---|---|---|
| `-p, --pred-col` | required | one or more prediction columns |
| `-i, --intended-col` | `intended_text` | intended text column |
| `-s, --spoken-col` | `spoken_text` | spoken text column |
| `-v, --version` | `v1` | prompt version |
| `-m, --mode` | `sync` | `sync` or `batch` |
| `-o, --out` | required | output CSV |
| `--model` | `claude-opus-5-5` | judge model |
| `--effort` | `medium` | judge effort level |
| `--workers` | `8` | parallel requests in sync mode |
| `--limit` | all rows | score only the first N rows |
| `--retries` | `3` | sync re-runs of failed drafts after the main run |

The output CSV is the input plus, for each prediction column:

| Column | Meaning |
|---|---|
| `<col>_fixed` | the judge's minimally edited draft |
| `<col>_edits` | character edits between the draft and `<col>_fixed` |
| `<col>_chars` | characters in the normalised `<col>_fixed` |
| `<col>_amend` | that draft's own edit rate, `edits / chars` |

A summary is printed, with AMEND pooled over all rows:

```
  pred_col  rows  failed  char_edits  chars  AMEND  AMEND@0
     wispr   750       0        1464 185012 0.0079   0.6627
     apple   749       1        4985 188412 0.0265   0.3044
```

### Sync vs batch

| | Sync | Batch |
|---|---|---|
| How | parallel requests (`--workers`) | Anthropic Message Batches API |
| Price | standard | 50% of standard |
| Speed | minutes | usually under an hour, up to 24 hours |
| Refusals | rerun on a fallback model automatically | marked as failed |

In both modes identical prompts are sent once, so the same draft from two tools, or for two speakers, is only paid for once.

## Failed rows

A draft is marked failed, with empty `fixed`, `edits`, `chars` and `amend`, when the judge refuses, the response is truncated or not valid JSON, or the batch request errors or expires. After the main run, in either mode, failed drafts are re-sent in sync mode up to `retries` times (default 3; `--retries` on the CLI, `retries=` in Python). Drafts that still fail are excluded from AMEND and AMEND@0 and counted in the `failed` column.

If the judge's version is empty, the draft contributes its own length as edits and 0 characters to the pooled AMEND, and its per-draft `amend` is 0 when the draft is also empty and 1 otherwise.

## Limitations

- **Judge noise.** The judge's model doesn't accept `temperature`, so repeated runs can differ slightly. Measure run-to-run variance on a sample before reading small gaps between tools.
- **Casing is ignored.** Text is lowercased, so an identifier that differs only in case (`pgbouncer` vs `PgBouncer`) is not penalised.
- **CER, not TER.** Unlike HTER, there is no shift operation: moving a phrase counts as a deletion plus an insertion.
- **Deleting text costs every character.** Removing a long hallucinated passage is one select-and-delete for a person, but counts as one edit per character.
- **A draft's `amend` can exceed 1** when the draft is much longer than the judge's version, for example when a tool appends a long generated answer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
