"""Command-line inference.

Examples:
    python -m src.predict "الفندق نظيف والموظفين محترمين"
    python -m src.predict "Arrived broken, very disappointed" --model baseline
    python -m src.predict --file feedback.csv --column text --out feedback_scored.csv
"""
import argparse
import sys

import pandas as pd

from src.service import SentimentService


def main():
    ap = argparse.ArgumentParser(description="Arabic / English customer feedback sentiment")
    ap.add_argument("texts", nargs="*", help="one or more feedback texts")
    ap.add_argument("--model", choices=["transformer", "baseline"], default="transformer")
    ap.add_argument("--file", help="CSV file with one feedback per row")
    ap.add_argument("--column", default="text", help="text column in the CSV")
    ap.add_argument("--out", help="where to save the scored CSV")
    a = ap.parse_args()

    svc = SentimentService()
    if a.model == "transformer" and not svc.has_transformer:
        print("Fine-tuned transformer not found (run python -m src.transformer). Using the baseline.", file=sys.stderr)

    if a.file:
        df = pd.read_csv(a.file)
        res = pd.DataFrame(svc.predict(df[a.column].astype(str).tolist(), model=a.model))
        df[["language", "sentiment", "confidence"]] = res[["language", "label", "confidence"]].values
        out = a.out or a.file.replace(".csv", "_scored.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(df["sentiment"].value_counts().to_string())
        print("saved", out)
        return

    if not a.texts:
        ap.error("give some text or --file")
    for r in svc.predict(a.texts, model=a.model):
        print(f"[{r['language']}] {r['label']:<8} {r['confidence']:.0%}  ({r['model']})  {r['text']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
