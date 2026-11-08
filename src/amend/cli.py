import argparse
import sys

import pandas as pd

from .api import score_dataframe
from .judge import EFFORT, MAX_TOKENS, MODEL, Judge
from .metric import pooled


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="amend",
        description="AMEND: an LLM judge applies the minimum edits to each prediction, "
        "then AMEND (CER between the prediction and the judge output) is reported per row.",
    )
    ap.add_argument("csv", help="input CSV")
    ap.add_argument("-p", "--pred-col", nargs="+", required=True, help="prediction column(s)")
    ap.add_argument("-i", "--intended-col", default="intended_text")
    ap.add_argument("-s", "--spoken-col", default="spoken_text", help="spoken text column")
    ap.add_argument("-v", "--version", choices=["v1"], default="v1", help="prompt version")
    ap.add_argument("-m", "--mode", choices=["sync", "batch"], default="sync")
    ap.add_argument("-o", "--out", required=True, help="output CSV")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--workers", type=int, default=8, help="parallel requests in sync mode")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    ap.add_argument("--retries", type=int, default=3, help="sync re-runs of failed drafts after the main run")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="output token limit per request")
    ap.add_argument("--resume", nargs="+", metavar="BATCH_ID", default=None,
                    help="collect results of already submitted batch(es) instead of submitting new ones; "
                    "use the same inputs and options as the original run")
    a = ap.parse_args(argv)
    if a.resume:
        a.mode = "batch"

    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
    if a.spoken_col not in df.columns:
        ap.error(f"spoken text column {a.spoken_col!r} not found (set --spoken-col)")
    if a.limit:
        df = df.head(a.limit)
    judge = Judge(model=a.model, effort=a.effort, max_tokens=a.max_tokens, workers=a.workers,
                  verbose=True, resume_batches=a.resume)
    out = score_dataframe(df, a.pred_col, a.intended_col, a.spoken_col, a.version, a.mode, judge, a.retries)
    out.to_csv(a.out, index=False)

    summary = pd.DataFrame(
        {
            "pred_col": c,
            "rows": out[f"{c}_amend"].notna().sum(),
            "failed": out[f"{c}_amend"].isna().sum(),
            "char_edits": int(out[f"{c}_edits"].sum()),
            "chars": int(out[f"{c}_chars"].sum()),
            "AMEND": pooled(out[f"{c}_edits"].dropna(), out[f"{c}_chars"].dropna()),
            "AMEND@0": (out[f"{c}_edits"].dropna() == 0).mean(),
        }
        for c in a.pred_col
    ).sort_values("AMEND")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
