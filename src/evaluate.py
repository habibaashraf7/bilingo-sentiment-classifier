"""Final evaluation on the held-out test set (looked at once, after all choices were made on validation).

Outputs (reports/):
  results.csv                macro-F1 / accuracy / per-class F1 per model and language
  significance.json          paired bootstrap CI for transformer - baseline
  error_tags.csv             error rate on negation, contrast, short / long texts...
  test_predictions.csv       every test prediction (for the notebook and error analysis)
  error_examples.csv         hand-pickable examples of each kind of mistake
  figures/*.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.service import LABELS, SentimentService
from src.transformer import MAX_LEN, ZERO_SHOT_MODEL

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, FIG = ROOT / "data" / "processed", ROOT / "reports", ROOT / "reports" / "figures"
MODELS = {"zero_shot": "Pretrained, no fine-tuning", "baseline": "TF-IDF + Logistic Regression",
          "transformer": "Fine-tuned XLM-RoBERTa"}
COLORS = {"zero_shot": "#C8D9E6", "baseline": "#F0A9BA", "transformer": "#2F4156"}  # BILINGO palette

CMAP = LinearSegmentedColormap.from_list("bilingo", [(0, "#C8D9E6"), (0.3, "#8EA8C0"), (0.6, "#4E6A87"), (1, "#2F4156")])

NEGATION = re.compile(r"\b(not|no|never|nothing|n't|don't|didn't|doesn't|isn't|wasn't|won't|cannot)\b|"
                      r"(^|\s)(لا|لم|لن|ليس|ليست|ما|مش|مو|غير|بدون)(\s|$)", re.I)
CONTRAST = re.compile(r"\b(but|however|although|though|except|yet)\b|(^|\s)(لكن|ولكن|بس|الا ان|إلا أن|رغم|مع ذلك)(\s|$)", re.I)


def zero_shot_predict(texts):
    import torch
    from transformers import pipeline
    clf = pipeline("text-classification", model=ZERO_SHOT_MODEL, device=0 if torch.cuda.is_available() else -1,
                   truncation=True, max_length=MAX_LEN)
    return [o["label"].lower().replace("neutral", "mixed") for o in clf(list(texts), batch_size=64)]


def predict_all(df, svc):
    df = df.copy()
    pb = svc.baseline_proba(df["text"].tolist())
    pt = svc.transformer_proba(df["text"].tolist())
    df["baseline"] = [LABELS[i] for i in pb.argmax(1)]
    df["baseline_conf"] = pb.max(1)
    df["transformer"] = [LABELS[i] for i in pt.argmax(1)]
    df["transformer_conf"] = pt.max(1)
    df["zero_shot"] = zero_shot_predict(df["text"])
    return df


def scores(df, model, subset):
    y, p = df["label"], df[model]
    per_class = f1_score(y, p, labels=LABELS, average=None)
    return {"model": MODELS[model], "key": model, "subset": subset, "n": len(df),
            "macro_f1": f1_score(y, p, average="macro"), "accuracy": accuracy_score(y, p),
            **{f"f1_{l}": v for l, v in zip(LABELS, per_class)}}


def paired_bootstrap(df, a="transformer", b="baseline", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, pa, pb = df["label"].values, df[a].values, df[b].values
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro"), "ci95": [lo, hi]}


def plot_confusions(test):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for r, lang in enumerate(["ar", "en"]):
        d = test[test["lang"] == lang]
        for c, m in enumerate(MODELS):
            cm = confusion_matrix(d["label"], d[m], labels=LABELS)
            pct = cm / cm.sum(1, keepdims=True)
            ax = axes[r, c]
            ax.imshow(pct, cmap=CMAP, vmin=0, vmax=1)
            ax.set_xticks(np.arange(-.5, 3), minor=True)
            ax.set_yticks(np.arange(-.5, 3), minor=True)
            ax.grid(which="minor", color="#F7C9D4", linewidth=2)
            ax.tick_params(which="minor", length=0)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center",
                            color="white" if pct[i, j] > 0.3 else "#222", fontsize=10)
            ax.set_xticks(range(3), LABELS)
            ax.set_yticks(range(3), LABELS)
            ax.set_xlabel("predicted")
            if c == 0:
                ax.set_ylabel(f"{'Arabic' if lang == 'ar' else 'English'}\n\ntrue")
            ax.set_title(f"{MODELS[m]}\nmacro-F1 {f1_score(d['label'], d[m], average='macro'):.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_f1(results):
    subsets = ["Arabic", "English", "Arabic SHEIN (new domain)"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.26
    for k, m in enumerate(MODELS):
        vals = [results[(results["key"] == m) & (results["subset"] == s)]["macro_f1"].iloc[0] for s in subsets]
        bars = ax.bar(np.arange(3) + (k - 1) * w, vals, w, label=MODELS[m], color=COLORS[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(3), subsets)
    ax.set_ylabel("macro-F1 (test)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1.png", dpi=130)
    plt.close(fig)


def error_tags(test):
    words = test["text"].str.split().str.len()
    tags = {"all": pd.Series(True, index=test.index),
            "has negation": test["text"].str.contains(NEGATION),
            "has contrast (but / لكن)": test["text"].str.contains(CONTRAST),
            "short (< 8 words)": words < 8,
            "long (> 120 words)": words > 120}
    rows = []
    for lang in ["ar", "en"]:
        for name, mask in tags.items():
            d = test[mask & (test["lang"] == lang)]
            rows.append({"lang": lang, "tag": name, "n": len(d),
                         **{f"error_{m}": float((d[m] != d["label"]).mean()) for m in MODELS}})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "error_tags.csv", index=False)
    return out


def error_examples(test):
    t = test.assign(words=test["text"].str.split().str.len())
    t = t[(t["words"] >= 4) & (t["words"] <= 60)]
    picks = []
    kinds = {
        "both wrong, transformer confident": (t["baseline"] != t["label"]) & (t["transformer"] != t["label"]),
        "transformer fixed baseline": (t["baseline"] != t["label"]) & (t["transformer"] == t["label"]),
        "baseline right, transformer wrong": (t["baseline"] == t["label"]) & (t["transformer"] != t["label"]),
    }
    for kind, mask in kinds.items():
        for lang in ["ar", "en"]:
            d = t[mask & (t["lang"] == lang)].sort_values("transformer_conf", ascending=False).head(15)
            picks.append(d.assign(kind=kind))
    cols = ["kind", "lang", "label", "baseline", "transformer", "transformer_conf", "zero_shot", "text"]
    pd.concat(picks)[cols].to_csv(REPORTS / "error_examples.csv", index=False, encoding="utf-8-sig")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    svc = SentimentService()
    test = predict_all(pd.read_parquet(DATA / "test.parquet"), svc)
    shein = predict_all(pd.read_parquet(DATA / "shein_ood.parquet"), svc)
    test.drop(columns="clean").to_csv(REPORTS / "test_predictions.csv", index=False, encoding="utf-8-sig")
    shein.drop(columns="clean").to_csv(REPORTS / "shein_predictions.csv", index=False, encoding="utf-8-sig")

    rows = []
    for m in MODELS:
        rows += [scores(test, m, "All test"), scores(test[test["lang"] == "ar"], m, "Arabic"),
                 scores(test[test["lang"] == "en"], m, "English"), scores(shein, m, "Arabic SHEIN (new domain)")]
    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "results.csv", index=False)
    print(results.round(3).to_string())

    sig = {"all": paired_bootstrap(test),
           "ar": paired_bootstrap(test[test["lang"] == "ar"]),
           "en": paired_bootstrap(test[test["lang"] == "en"]),
           "shein": paired_bootstrap(shein)}
    (REPORTS / "significance.json").write_text(json.dumps(sig, indent=2))
    print(json.dumps(sig, indent=1))

    plot_confusions(test)
    plot_f1(results)
    print(error_tags(test).round(3).to_string())
    error_examples(test)


def replot():
    """Redraw the figures from saved predictions (no models needed)."""
    test = pd.read_csv(REPORTS / "test_predictions.csv")
    plot_confusions(test)
    plot_f1(pd.read_csv(REPORTS / "results.csv"))


if __name__ == "__main__":
    import sys
    replot() if "--replot" in sys.argv else main()
