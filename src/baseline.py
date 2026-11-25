"""Classical baselines: TF-IDF features + a linear classifier.

One model handles both languages. Features are word 1-2 grams plus character
2-5 grams; character n-grams help Arabic, where one word can appear with many
prefixes and suffixes (الفندق / بالفندق / وفندقكم).

Candidates are compared on the validation set (macro-F1); the winner is saved.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def features(char=True):
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True))]
    if char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                              max_features=300_000, sublinear_tf=True)))
    return FeatureUnion(parts)


def candidates():
    yield "Naive Bayes (words)", Pipeline([("tfidf", features(char=False)), ("clf", ComplementNB(alpha=0.5))])
    for C in [0.5, 1, 2, 4]:
        yield f"Logistic regression C={C}", Pipeline([("tfidf", features()), ("clf", LogisticRegression(C=C, max_iter=2000))])
    for C in [0.1, 0.3]:
        yield f"Linear SVM C={C}", Pipeline([("tfidf", features()), ("clf", LinearSVC(C=C))])


def main():
    train, val = pd.read_parquet(DATA / "train.parquet"), pd.read_parquet(DATA / "val.parquet")
    rows, best = [], None
    for name, model in candidates():
        t = time.time()
        model.fit(train["clean"], train["label"])
        pred = model.predict(val["clean"])
        row = {"model": name, "val_macro_f1": f1_score(val["label"], pred, average="macro"),
               "fit_seconds": round(time.time() - t, 1)}
        for lang in ["ar", "en"]:
            m = val["lang"] == lang
            row[f"val_f1_{lang}"] = f1_score(val.loc[m, "label"], pred[m.values], average="macro")
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        # SVM has no predict_proba; we prefer LR when scores are close, for the confidence in the demo
        if best is None or row["val_macro_f1"] > best[1] + (0.003 if "SVM" in name else 0):
            if "Naive" not in name:
                best = (name, row["val_macro_f1"], model)

    pd.DataFrame(rows).to_csv(ROOT / "reports" / "baseline_selection.csv", index=False)
    name, f1, model = best
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "baseline_tfidf.joblib", compress=3)
    (MODELS / "baseline_info.json").write_text(json.dumps({"model": name, "val_macro_f1": f1}, indent=2))
    print("selected:", name, round(f1, 4))


if __name__ == "__main__":
    main()
