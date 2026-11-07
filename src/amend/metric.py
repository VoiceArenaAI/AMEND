import re
import unicodedata

import jiwer

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_LEAD = "\"'([{<*_`~•·…—–,;:!?"
_TRAIL = "\"')]}>*_`~•·…—–,;:!?.-"
_JUNK = set(_LEAD) | set(_TRAIL) | set("#")
_ELLIPSIS = re.compile(r"\.{2,}")


def _clean(token: str) -> str:
    token = _ELLIPSIS.sub(" ", token).strip()
    if not token or all(ch in _JUNK for ch in token):
        return ""
    return token.lstrip(_LEAD).rstrip(_TRAIL)


def normalize(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    tokens = (_clean(t) for t in _ELLIPSIS.sub(" ", text).split())
    return " ".join(t for t in tokens if t)


def char_edits(pred: str | None, fixed: str | None) -> tuple[int, int]:
    ref = normalize(fixed)
    hyp = normalize(pred)
    if not ref:
        return len(hyp), 0
    if not hyp:
        return len(ref), len(ref)
    o = jiwer.process_characters(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions


def amend(pred: str | None, fixed: str | None) -> float:
    edits, chars = char_edits(pred, fixed)
    if not chars:
        return 0.0 if not edits else 1.0
    return edits / chars


def pooled(edits, chars) -> float:
    edits, chars = sum(edits), sum(chars)
    return edits / chars if chars else 0.0
