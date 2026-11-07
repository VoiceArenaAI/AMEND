from typing import Sequence

import pandas as pd

from .judge import Judge
from .metric import char_edits
from .prompts import build_prompt


def _judge(judge: Judge | None, **kwargs) -> Judge:
    return judge or Judge(**kwargs)


def _log(judge, msg: str) -> None:
    if hasattr(judge, "_log"):
        judge._log(msg)


def score_arrays(
    preds: Sequence[str],
    intendeds: Sequence[str],
    spokens: Sequence[str],
    version: str = "v1",
    mode: str = "sync",
    judge: Judge | None = None,
    retries: int = 3,
    **judge_kwargs,
) -> list[dict]:
    if not (len(preds) == len(intendeds) == len(spokens)):
        raise ValueError("preds, intendeds and spokens must have the same length")
    judge = _judge(judge, **judge_kwargs)
    prompts = [build_prompt(i, p, s, version) for p, i, s in zip(preds, intendeds, spokens)]
    fixed = judge.fix_many(prompts, mode=mode)

    def failed() -> list[int]:
        return [k for k, f in enumerate(fixed) if f is None]

    for attempt in range(1, retries + 1):
        bad = failed()
        if not bad:
            break
        _log(judge, f"retry {attempt}/{retries}: re-running {len(bad)} failed drafts in sync mode")
        for k, f in zip(bad, judge.fix_many([prompts[k] for k in bad], mode="sync")):
            fixed[k] = f
    bad = failed()
    if bad:
        _log(judge, f"{len(bad)} drafts still failed after {retries} retries")
    out = []
    for p, f in zip(preds, fixed):
        if f is None:
            out.append({"pred": p, "fixed": None, "edits": None, "chars": None, "amend": None})
            continue
        e, c = char_edits(p, f)
        out.append({"pred": p, "fixed": f, "edits": e, "chars": c, "amend": e / c if c else float(e > 0)})
    return out


def score(
    pred: str,
    intended: str,
    spoken: str,
    version: str = "v1",
    judge: Judge | None = None,
    retries: int = 3,
    **judge_kwargs,
) -> dict:
    return score_arrays([pred], [intended], [spoken], version, "sync", judge, retries, **judge_kwargs)[0]


def score_dataframe(
    df: pd.DataFrame,
    pred_cols: str | Sequence[str],
    intended_col: str = "intended_text",
    spoken_col: str = "spoken_text",
    version: str = "v1",
    mode: str = "sync",
    judge: Judge | None = None,
    retries: int = 3,
    **judge_kwargs,
) -> pd.DataFrame:
    cols = [pred_cols] if isinstance(pred_cols, str) else list(pred_cols)
    out = df.copy()
    n = len(df)
    preds, intendeds, spokens = [], [], []
    for c in cols:
        preds += df[c].fillna("").astype(str).tolist()
        intendeds += df[intended_col].fillna("").astype(str).tolist()
        spokens += df[spoken_col].fillna("").astype(str).tolist()
    results = score_arrays(preds, intendeds, spokens, version, mode, _judge(judge, **judge_kwargs), retries)
    for k, c in enumerate(cols):
        part = results[k * n : (k + 1) * n]
        out[f"{c}_fixed"] = [r["fixed"] for r in part]
        out[f"{c}_edits"] = [r["edits"] for r in part]
        out[f"{c}_chars"] = [r["chars"] for r in part]
        out[f"{c}_amend"] = [r["amend"] for r in part]
    return out
